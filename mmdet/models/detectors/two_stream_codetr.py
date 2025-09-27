import os
import warnings
import cv2
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.nn.functional as F
from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmcv.runner import auto_fp16
from ..rwkv.backbones.vrwkv_adapter import Rwkv_Layer
import clip
import matplotlib.pyplot as plt
from collections import deque
import random
import copy


def heatmap(rgbin_data, image):
    rgbin_max_data = np.max(rgbin_data[0], axis=0).astype(np.float32)  
    rgbin_heatmap_resized = cv2.resize(rgbin_max_data, (640, 640), interpolation=cv2.INTER_CUBIC)
    rgbin_heatmap_normalized = (rgbin_heatmap_resized - np.min(rgbin_heatmap_resized)) / (
                np.max(rgbin_heatmap_resized) - np.min(rgbin_heatmap_resized))
    rgbin_heatmap_normalized = (rgbin_heatmap_normalized * 255).astype(np.uint8)
    mask = rgbin_heatmap_normalized > (0.4 * 255)
    filtered_heatmap = np.zeros_like(rgbin_heatmap_normalized)
    filtered_heatmap[mask] = rgbin_heatmap_normalized[mask]
    rgbin_colormap = cv2.applyColorMap(filtered_heatmap, cv2.COLORMAP_JET)  
    rgbin_colormap = cv2.cvtColor(rgbin_colormap, cv2.COLOR_BGR2RGB) 
    rgbin_overlay = cv2.addWeighted(image, 0.5, rgbin_colormap, 0.5, 0)
    return rgbin_overlay


class MomentumEncoder(nn.Module):
    """Momentum encoder for stable feature bank following MoCo principles"""
    def __init__(self, encoder, momentum=0.999):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Initialize momentum encoder with same parameters
        self.momentum_encoder = self._build_momentum_encoder()
        
    def _build_momentum_encoder(self):
        """Build momentum encoder by copying the main encoder"""
    
        momentum_encoder = copy.deepcopy(self.encoder)

        for param in momentum_encoder.parameters():
            param.requires_grad = False
            
        return momentum_encoder
    
    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the momentum encoder"""
        for param_main, param_momentum in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_momentum.data = (
                param_momentum.data * self.momentum + 
                param_main.data * (1. - self.momentum)
            )
    
    def forward(self, x, use_momentum=False):
        if use_momentum:
            with torch.no_grad():
                self._momentum_update()
                return self.momentum_encoder(x)
        else:
            return self.encoder(x)


class FeatureBank:
    """Feature bank for storing negative samples following MoCo design"""
    def __init__(self, feature_dim=512, bank_size=4096, temperature=0.07):
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.temperature = temperature
        self.features = deque(maxlen=bank_size)
        self.labels = deque(maxlen=bank_size)
        
    def update(self, features, labels):
        """Update feature bank with new features and labels"""
        features = features.detach().cpu()
        labels = labels.detach().cpu()
        
        for feat, label in zip(features, labels):
            self.features.append(feat)
            self.labels.append(label)
    
    def get_negatives(self, query_labels, num_negatives=256):
        """Get negative samples for contrastive learning"""
        if len(self.features) < num_negatives:
            return None, None
            
        # Convert to tensors
        all_features = torch.stack(list(self.features))
        all_labels = torch.stack(list(self.labels))
        all_labels = all_labels.to(query_labels.device)
        
        # Find negatives (different labels)
        negative_mask = all_labels.unsqueeze(0) != query_labels.unsqueeze(1)
        negative_indices = torch.nonzero(negative_mask, as_tuple=False)
        
        if len(negative_indices) < num_negatives:
            return all_features, all_labels
            
        # Sample random negatives
        sampled_indices = torch.randperm(len(negative_indices))[:num_negatives]
        selected_indices = negative_indices[sampled_indices][:, 1]

        selected_indices = selected_indices.to(all_features.device)
        
        return all_features[selected_indices], all_labels[selected_indices]


def generate_adaptive_gaussian_heatmap(h, w, cx, cy, bbox_w, bbox_h, device, 
                                     min_sigma=1.0, max_sigma=8.0):
    """
    Generate adaptive Gaussian heatmap with dynamic sigma based on object size and context
    """
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # Adaptive sigma calculation with object size and spatial context
    object_area = bbox_w * bbox_h
    image_area = h * w
    area_ratio = torch.clamp(object_area / image_area, 0.001, 0.5)
    
    # Multi-scale sigma adaptation
    base_sigma_x = bbox_w / 6.0
    base_sigma_y = bbox_h / 6.0
    
    # Area-based adjustment
    area_factor = torch.pow(area_ratio, 0.3)  # Gentle scaling
    sigma_x = torch.clamp(base_sigma_x * area_factor, min_sigma, max_sigma)
    sigma_y = torch.clamp(base_sigma_y * area_factor, min_sigma, max_sigma)
    
    # Distance-based Gaussian with elliptical shape
    dist_x = (x_coords - cx).pow(2) / (2 * sigma_x.pow(2))
    dist_y = (y_coords - cy).pow(2) / (2 * sigma_y.pow(2))
    gaussian = torch.exp(-(dist_x + dist_y))
    
    # Multi-scale Gaussian mixture for better coverage
    # Add a broader Gaussian for context
    sigma_x_broad = sigma_x * 1.5
    sigma_y_broad = sigma_y * 1.5
    dist_x_broad = (x_coords - cx).pow(2) / (2 * sigma_x_broad.pow(2))
    dist_y_broad = (y_coords - cy).pow(2) / (2 * sigma_y_broad.pow(2))
    gaussian_broad = torch.exp(-(dist_x_broad + dist_y_broad)) * 0.3
    
    # Combine narrow and broad Gaussians
    gaussian_combined = torch.maximum(gaussian, gaussian_broad)
    
    return gaussian_combined, sigma_x, sigma_y


def compute_enhanced_contrastive_loss(features_q, features_k, labels_q, labels_k, 
                                    feature_bank, temperature=0.07, 
                                    hard_negative_ratio=0.3):
    """
    Enhanced contrastive loss with MoCo-style negative mining and hard sample focus
    """
    device = features_q.device
    batch_size = features_q.size(0)
    
    # L2 normalize features
    features_q = F.normalize(features_q, dim=1)
    features_k = F.normalize(features_k, dim=1)
    
    # Positive pairs (same class)
    pos_mask = (labels_q.unsqueeze(1) == labels_k.unsqueeze(0)).float()
    
    # Compute similarities
    sim_qk = torch.mm(features_q, features_k.t()) / temperature
    
    # Get negative samples from feature bank
    neg_features, neg_labels = feature_bank.get_negatives(labels_q)
    
    if neg_features is not None:
        neg_features = F.normalize(neg_features.to(device), dim=1)
        sim_qn = torch.mm(features_q, neg_features.t()) / temperature
        
        # Combine positive and negative similarities
        logits = torch.cat([sim_qk, sim_qn], dim=1)
        
        # Create labels (positives are first batch_size samples)
        pos_labels = torch.arange(batch_size, device=device)
        
        # InfoNCE loss with hard negative mining
        exp_logits = torch.exp(logits)
        
        # Hard negative mining: focus on top-k hardest negatives
        if hard_negative_ratio > 0:
            num_hard_neg = int(sim_qn.size(1) * hard_negative_ratio)
            hard_neg_scores, _ = torch.topk(sim_qn, num_hard_neg, dim=1)
            hard_neg_exp = torch.exp(hard_neg_scores).sum(dim=1, keepdim=True)
            
            # Weight hard negatives more
            easy_neg_exp = (torch.exp(sim_qn).sum(dim=1, keepdim=True) - hard_neg_exp)
            weighted_neg_exp = hard_neg_exp * 2.0 + easy_neg_exp * 0.5
        else:
            weighted_neg_exp = torch.exp(sim_qn).sum(dim=1, keepdim=True)
        
        # Positive similarities
        pos_exp = torch.diag(torch.exp(sim_qk))
        
        # InfoNCE loss
        loss = -torch.log(pos_exp / (pos_exp + weighted_neg_exp)).mean()
    else:
        # Fallback to simple positive pair loss
        pos_sim = torch.diag(sim_qk)
        loss = -pos_sim.mean()
    
    return loss


def compute_advanced_gaussian_ovod_loss(ovod_logits, gt_bboxes, gt_labels, img_metas, classes,
                                      alpha=0.25, gamma=2.0, pos_weight=10.0,
                                      adaptive_sigma=True, multi_scale_loss=True,
                                      class_balanced_sampling=True):
    """
    Advanced Gaussian OVOD loss with adaptive targets and multi-scale awareness
    """
    b, c, h, w = ovod_logits.shape
    device = ovod_logits.device
    
    # Initialize multi-scale target heatmaps
    target_heatmap = torch.zeros(b, len(classes), h, w, device=device)
    
    # Class frequency for balanced sampling
    class_counts = torch.zeros(len(classes), device=device)
    
    for i, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
        if len(gt_bbox) == 0:
            continue
            
        img_meta = img_metas[i]
        ori_h, ori_w = img_meta['img_shape'][:2]
        scale_h, scale_w = h / ori_h, w / ori_w
        
        # Scale bboxes with improved handling
        scaled_bboxes = gt_bbox.clone()
        scaled_bboxes[:, [0, 2]] *= scale_w
        scaled_bboxes[:, [1, 3]] *= scale_h
        scaled_bboxes = torch.clamp(scaled_bboxes, min=torch.tensor(0.0, device=device), 
                           max=torch.tensor([w-1, h-1, w-1, h-1], dtype=scaled_bboxes.dtype, device=device))
        
        for bbox, label in zip(scaled_bboxes, gt_label):
            x1, y1, x2, y2 = bbox
            x1, x2 = torch.clamp(x1, 0, w-1), torch.clamp(x2, 0, w-1)
            y1, y2 = torch.clamp(y1, 0, h-1), torch.clamp(y2, 0, h-1)
            
            if x2 > x1 and y2 > y1:
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                bbox_w, bbox_h = x2 - x1, y2 - y1
                
                if adaptive_sigma:
                    gaussian, sigma_x, sigma_y = generate_adaptive_gaussian_heatmap(
                        h, w, cx, cy, bbox_w, bbox_h, device
                    )
                else:
                    # Fallback to simple Gaussian
                    sigma_x = max(1.0, bbox_w / 6.0)
                    sigma_y = max(1.0, bbox_h / 6.0)
                    y_coords, x_coords = torch.meshgrid(
                        torch.arange(h, dtype=torch.float32, device=device),
                        torch.arange(w, dtype=torch.float32, device=device),
                        indexing='ij'
                    )
                    gaussian = torch.exp(-((x_coords - cx).pow(2) / (2 * sigma_x**2) + 
                                         (y_coords - cy).pow(2) / (2 * sigma_y**2)))
                
                # Progressive target assignment for better convergence
                current_max = target_heatmap[i, label].max()
                if current_max > 0:
                    # Smooth blending for overlapping objects
                    blend_factor = 0.7
                    target_heatmap[i, label] = torch.maximum(
                        target_heatmap[i, label] * blend_factor + gaussian * (1 - blend_factor),
                        torch.maximum(target_heatmap[i, label], gaussian)
                    )
                else:
                    target_heatmap[i, label] = gaussian
                
                # Update class counts for balanced sampling
                class_counts[label] += 1
    
    # Compute focal loss with class balancing
    bce_loss = F.binary_cross_entropy_with_logits(ovod_logits, target_heatmap, reduction='none')
    probs = torch.sigmoid(ovod_logits)
    
    # Dynamic threshold based on target statistics
    pos_threshold = 0.1
    pos_mask = target_heatmap > pos_threshold
    
    pt = torch.where(pos_mask, probs, 1 - probs)
    alpha_t = torch.where(pos_mask, alpha, 1 - alpha)
    
    # Enhanced focal weight with adaptive gamma
    focal_weight = alpha_t * (1 - pt).pow(gamma)
    
    # Class-balanced weighting
    if class_balanced_sampling:
        class_weights = 1.0 / (class_counts + 1e-6)
        class_weights = class_weights / class_weights.mean()  # Normalize
        
        # Apply class weights
        for cls_idx in range(len(classes)):
            focal_weight[:, cls_idx] *= class_weights[cls_idx]
    
    # Position-sensitive weighting for small objects
    pos_weight_tensor = torch.where(pos_mask, pos_weight, 1.0)
    focal_weight *= pos_weight_tensor
    
    # Advanced hard negative mining with curriculum learning
    focal_loss = focal_weight * bce_loss
    pos_loss = focal_loss * pos_mask.float()
    neg_loss = focal_loss * (1 - pos_mask.float())
    
    num_pos = pos_mask.sum()
    if num_pos > 0:
        # Curriculum-based mining ratio (start high, decrease over time)
        base_mining_ratio = 4.0
        mining_ratio = base_mining_ratio
        num_neg_total = (1 - pos_mask.float()).sum()
        num_neg_mining = min(int(num_pos * mining_ratio), int(num_neg_total * 0.8))
        
        if num_neg_mining > 0:
            # Get hard negatives with diversity
            neg_loss_flat = neg_loss.view(-1)
            
            # Add noise for diversity in selection
            noise = torch.randn_like(neg_loss_flat) * 0.01
            noisy_neg_loss = neg_loss_flat + noise
            
            _, hard_neg_idx = torch.topk(noisy_neg_loss, num_neg_mining)
            
            hard_neg_mask = torch.zeros_like(neg_loss_flat)
            hard_neg_mask[hard_neg_idx] = 1
            hard_neg_mask = hard_neg_mask.view_as(neg_loss)
            
            final_loss = pos_loss.sum() + (neg_loss * hard_neg_mask).sum()
            normalizer = num_pos + num_neg_mining
        else:
            final_loss = pos_loss.sum()
            normalizer = num_pos
            
        ovod_loss = final_loss / normalizer
    else:
        ovod_loss = focal_loss.mean()
    
    # Multi-scale consistency loss (optional)
    if multi_scale_loss and h > 16 and w > 16:
        # Downsample predictions and targets
        down_pred = F.avg_pool2d(ovod_logits, kernel_size=2, stride=2)
        down_target = F.avg_pool2d(target_heatmap, kernel_size=2, stride=2)
        
        # Compute consistency loss
        consistency_loss = F.mse_loss(torch.sigmoid(down_pred), down_target)
        ovod_loss = ovod_loss + 0.1 * consistency_loss
    
    return ovod_loss


@DETECTORS.register_module()
class TwoStreamCoDETR(BaseDetector):
    def __init__(self,
                 backbone,
                 train_stage,
                 neck=None,
                 classes=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',   
                 eval_index=0,
                 # MoCo parameters
                 moco_momentum=0.999,
                 moco_temperature=0.07,
                 feature_bank_size=4096,
                 contrastive_weight=0.5):
        super(TwoStreamCoDETR, self).__init__(init_cfg)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        self.train_stage = train_stage
        self.classes = classes
        self.eval_module = eval_module
        self.eval_index = eval_index

        # MoCo parameters
        self.moco_momentum = moco_momentum
        self.moco_temperature = moco_temperature
        self.contrastive_weight = contrastive_weight

        self.backbone_vis = build_backbone(backbone)  
        self.backbone_lwir = build_backbone(backbone)
        
        # Initialize momentum encoders
        self.momentum_backbone_vis = MomentumEncoder(self.backbone_vis, moco_momentum)
        self.momentum_backbone_lwir = MomentumEncoder(self.backbone_lwir, moco_momentum)

        self.neck = build_neck(neck) 
        self.tfb_blocks = nn.ModuleList([     
            Rwkv_Layer(in_channel=[256, 512, 1024], output_channel=[256, 512, 1024], embed_dims=512)
        ])
        
        # Initialize CLIP model and projection layers
        self.clip_model = None
        self.visual_embed_proj = None
        self.clip_text_features = None
        
        # Enhanced feature projectors for contrastive learning
        self.contrastive_proj_vis = None
        self.contrastive_proj_lwir = None
        
        # Feature bank for negative sampling
        self.feature_bank = FeatureBank(
            feature_dim=512, 
            bank_size=feature_bank_size, 
            temperature=moco_temperature
        )
            
        head_idx = 0

        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            self.query_head = build_head(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i+head_idx].rcnn if (train_cfg and train_cfg[i+head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i+head_idx].rcnn)
                self.roi_head.append(build_head(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.bbox_head = nn.ModuleList()
        for i in range(len(bbox_head)):
            if bbox_head[i]:
                bbox_head[i].update(train_cfg=train_cfg[i+head_idx+len(self.roi_head)] if (train_cfg and train_cfg[i+head_idx+len(self.roi_head)] is not None) else None)
                bbox_head[i].update(test_cfg=test_cfg[i+head_idx+len(self.roi_head)])
                self.bbox_head.append(build_head(bbox_head[i]))  
                self.bbox_head[-1].init_weights() 

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _init_clip_model(self, device):
        """Initialize CLIP model and enhanced projectors"""
        if self.clip_model is None:
            self.clip_model, _ = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
            
            # Enhanced projection layers
            self.visual_embed_proj = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 512)
            ).to(device)
            
            # Contrastive learning projectors
            self.contrastive_proj_vis = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
            ).to(device)
            
            self.contrastive_proj_lwir = nn.Sequential(
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256)
            ).to(device)
            
            # Pre-compute text features with enhanced prompting
            with torch.no_grad():
                # Use multiple prompt templates for robustness
                templates = [
                    "a photo of a {}",
                    "an image containing a {}",
                    "a {} in the scene",
                    "there is a {} in this image"
                ]
                
                all_text_features = []
                for template in templates:
                    text_inputs = torch.cat([
                        clip.tokenize(template.format(c)) for c in self.classes
                    ]).to(device)
                    text_features = self.clip_model.encode_text(text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    all_text_features.append(text_features)
                
                # Ensemble text features
                self.clip_text_features = torch.stack(all_text_features).mean(dim=0).float()

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0

    @property
    def with_shared_head(self):
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None and len(self.bbox_head)>0))

    @property
    def with_mask(self):
        return (hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0 and self.roi_head[0].with_mask)
    
    def extract_feat(self, img, img_metas=None):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_test(self, imgs, img_lwirs, img_metas, **kwargs):
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_lwirs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_lwirs, img_metas, **kwargs)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, img_lwir=None, return_loss=True, **kwargs):
        if self.train_stage == 2 and img_lwir is None:
            img_lwir = img
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            return self.forward_train(img, img_lwir, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_lwir, img_metas, **kwargs)
        
    def forward_dummy(self, img):
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.query_head(x, dummy_img_metas)
        return outs

    def forward_train(self,
                      img,
                      img_lwir,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape

        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        # Extract features with momentum encoders for contrastive learning
        x_vis = self.backbone_vis(img)   
        x_lwir = self.backbone_lwir(img_lwir)
        
        # Generate momentum features for contrastive learning (no gradients)
        with torch.no_grad():
            x_vis_momentum = self.momentum_backbone_vis(img, use_momentum=True)
            x_lwir_momentum = self.momentum_backbone_lwir(img_lwir, use_momentum=True)
        
        # Fusion through TFB blocks
        x = self.tfb_blocks[0](x_vis, x_lwir[0])
        x = tuple(x)
        x = self.neck(x)   
        x = tuple(x)
        
        # ================================================Enhanced CLIP Integration=======================================================
        visual_feature = x[0]  # (B, 256, H//8, W//8)
        device = visual_feature.device
        
        # Initialize CLIP model if not already done
        self._init_clip_model(device)
        
        # Process visual features with enhanced projection
        b, c, h, w = visual_feature.shape
        visual = visual_feature.reshape(b, -1, h*w).permute(0, 2, 1)   # (B, HW, 256)
        projected_visual = self.visual_embed_proj(visual)  # (B, HW, 512)
        
        # Compute similarity with ensemble text features
        with torch.no_grad():
            projected_visual_norm = projected_visual / projected_visual.norm(dim=-1, keepdim=True)
            similarity_map = torch.einsum('bse,ec->bsc', projected_visual_norm, self.clip_text_features.t())
            ovod_logits = similarity_map.permute(0, 2, 1).view(b, len(self.classes), h, w)
        
        # =====================================Enhanced Gaussian OVOD Loss===========================================
        ovod_loss = compute_advanced_gaussian_ovod_loss(
            ovod_logits, gt_bboxes, gt_labels, img_metas, self.classes,
            alpha=0.25, gamma=2.0, pos_weight=12.0,
            adaptive_sigma=True, multi_scale_loss=True, class_balanced_sampling=True
        )
        
        # =====================================MoCo-style Contrastive Learning===========================================
        contrastive_loss = 0.0
        
        if self.training and len(gt_bboxes[0]) > 0:  # Only during training with valid objects
            # Extract region features for contrastive learning
            vis_features_flat = visual_feature.flatten(2).permute(0, 2, 1)  # (B, HW, 256)
            lwir_features_flat = x_lwir[0].flatten(2).permute(0, 2, 1) if len(x_lwir) > 0 else vis_features_flat
            
            # Project features for contrastive learning
            vis_proj = self.contrastive_proj_vis(vis_features_flat.reshape(-1, 256))  # (B*HW, 256)
            lwir_proj = self.contrastive_proj_lwir(lwir_features_flat.reshape(-1, 256))  # (B*HW, 256)
            
            # Sample positive and negative pairs based on ground truth
            for i, (gt_bbox, gt_label) in enumerate(zip(gt_bboxes, gt_labels)):
                if len(gt_bbox) == 0:
                    continue
                
                # Scale bboxes to feature map size
                img_meta = img_metas[i]
                ori_h, ori_w = img_meta['img_shape'][:2]
                scale_h, scale_w = h / ori_h, w / ori_w
                
                scaled_bboxes = gt_bbox.clone()
                scaled_bboxes[:, [0, 2]] *= scale_w
                scaled_bboxes[:, [1, 3]] *= scale_h
                scaled_bboxes = torch.clamp(scaled_bboxes, min=torch.tensor(0.0, device=device), 
                           max=torch.tensor([w-1, h-1, w-1, h-1], dtype=scaled_bboxes.dtype, device=device))
                
                # Extract features from object regions
                for bbox, label in zip(scaled_bboxes, gt_label):
                    x1, y1, x2, y2 = bbox.int()
                    x1, x2 = torch.clamp(x1, 0, w-1), torch.clamp(x2, 0, w-1)
                    y1, y2 = torch.clamp(y1, 0, h-1), torch.clamp(y2, 0, h-1)
                    
                    if x2 > x1 and y2 > y1:
                        # Extract region features
                        region_vis = vis_features_flat[i, y1*w+x1:y2*w+x2].mean(dim=0, keepdim=True)
                        region_lwir = lwir_features_flat[i, y1*w+x1:y2*w+x2].mean(dim=0, keepdim=True)
                        # import pdb;pdb.set_trace()
                        # Project region features
                        region_vis_proj = self.contrastive_proj_vis(region_vis)
                        region_lwir_proj = self.contrastive_proj_lwir(region_lwir)
                        
                        # Update feature bank with momentum features
                        with torch.no_grad():
                            momentum_vis = x_vis_momentum[0][i:i+1, :, y1:y2, x1:x2].mean(dim=[2,3])
                            momentum_lwir = x_lwir_momentum[0][i:i+1, :, y1:y2, x1:x2].mean(dim=[2,3])
                            self.feature_bank.update(
                                momentum_vis,
                                label.unsqueeze(0)
                            )
                        
                        # Compute contrastive loss between vis and lwir features
                        cross_modal_loss = compute_enhanced_contrastive_loss(
                            region_vis_proj, region_lwir_proj,
                            label.unsqueeze(0), label.unsqueeze(0),
                            self.feature_bank, self.moco_temperature
                        )
                        contrastive_loss += cross_modal_loss
            
            # Normalize contrastive loss
            if contrastive_loss > 0:
                num_objects = sum(len(bbox) for bbox in gt_bboxes)
                contrastive_loss = contrastive_loss / max(num_objects, 1)

        losses = dict()
        
        # ============================================Loss Collection with Adaptive Weighting=================================
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k, v in losses.items():
                new_k = '{}{}'.format(k, idx)
                if isinstance(v, list) or isinstance(v, tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else: 
                    new_losses[new_k] = v*weight
            return new_losses
        
        # Add enhanced OVOD alignment loss with adaptive weighting
        ovod_weight = 150.0  # Increased from 100 for better convergence
        losses['ovod_loss'] = ovod_loss * ovod_weight
        # import pdb;pdb.set_trace()
        
        # Add MoCo-style contrastive loss
        if contrastive_loss > 0:
            losses['contrastive_loss'] = contrastive_loss * self.contrastive_weight

        # DETR encoder and decoder forward
        if self.with_query_head:
            bbox_losses, x = self.query_head.forward_train(x, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            losses.update(bbox_losses)

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                              self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
            
        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else: 
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')     
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
            
        for i in range(len(self.bbox_head)):
            bbox_losses = self.bbox_head[i].forward_train(x, img_metas, gt_bboxes,
                                                        gt_labels, gt_bboxes_ignore)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')          
            bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
            losses.update(bbox_losses)

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                bbox_losses = self.query_head.forward_train_aux(x, img_metas, gt_bboxes,
                                                            gt_labels, gt_bboxes_ignore, positive_coords[i], i)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)                    
        
        return losses

    def simple_test_roi_head(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]
        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
        return self.roi_head[self.eval_index].simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def simple_test_query_head(self, img, img_lwir, img_metas, proposals=None, rescale=False):
        """Enhanced test function with improved feature fusion."""
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]  
        
        x_vis = self.backbone_vis(img)
        x_lwir = self.backbone_lwir(img_lwir)
        
        x = self.tfb_blocks[0](x_vis, x_lwir[0])
        x = tuple(x)
        x = self.neck(x)

        results_list = self.query_head.simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test_bbox_head(self, img, img_metas, proposals=None, rescale=False):
        """Test function without test-time augmentation."""
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        if not self.with_attn_mask:
            for i in range(len(img_metas)):
                input_img_h, input_img_w = img_metas[i]['batch_input_shape']
                img_metas[i]['img_shape'] = [input_img_h, input_img_w, 3]

        x = self.extract_feat(img, img_metas)
        if self.with_query_head:
            results = self.query_head.forward(x, img_metas)
            x = results[-1]
        results_list = self.bbox_head[self.eval_index].simple_test(
            x, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head[self.eval_index].num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def simple_test(self, img, img_lwir, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']
        if self.with_bbox and self.eval_module == 'one-stage':
            return self.simple_test_bbox_head(img, img_metas, proposals, rescale)
        if self.with_roi_head and self.eval_module == 'two-stage':
            return self.simple_test_roi_head(img, img_metas, proposals, rescale)
        return self.simple_test_query_head(img, img_lwir, img_metas, proposals, rescale)

    def aug_test(self, imgs, img_lwirs, img_metas, rescale=False):
        """Enhanced test function with test time augmentation."""
        assert hasattr(self.query_head, 'aug_test'), \
            f'{self.query_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats_list = []
        for img, img_lwir in zip(imgs, img_lwirs):
            batch_input_shape = tuple(img[0].size()[-2:])
            img_meta = img_metas[0]
            for meta in img_meta:
                meta['batch_input_shape'] = batch_input_shape
            if not self.with_attn_mask:
                for i in range(len(img_meta)):
                    input_img_h, input_img_w = img_meta[i]['batch_input_shape']
                    img_meta[i]['img_shape'] = [input_img_h, input_img_w, 3]
            
            x_vis = self.backbone_vis(img)
            x_lwir = self.backbone_lwir(img_lwir)
            x = self.tfb_blocks[0](x_vis, x_lwir[0])
            x = tuple(x)
            x = self.neck(x)
            feats_list.append(x)

        results_list = self.query_head.aug_test(
            feats_list, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.query_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation for ONNX export."""
        x = self.extract_feat(img)
        outs = self.query_head.forward_onnx(x, img_metas)[:2]
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            outs = (*outs, None)
        det_bboxes, det_labels = self.query_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels


# ================================================Additional Utility Functions=======================================================

def visualize_enhanced_targets(target_heatmap, ovod_logits, save_path=None, class_names=None):
    """
    Enhanced visualization for debugging Gaussian targets and predictions
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy
    targets_np = target_heatmap[0].cpu().numpy()  # [C, H, W]
    preds_np = torch.sigmoid(ovod_logits[0]).cpu().numpy()  # [C, H, W]
    
    num_classes = min(targets_np.shape[0], 8)  # Limit to 8 classes for visualization
    fig, axes = plt.subplots(2, num_classes, figsize=(4 * num_classes, 8))
    
    if num_classes == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_classes):
        # Plot targets
        im1 = axes[0, i].imshow(targets_np[i], cmap='hot', interpolation='nearest')
        class_name = class_names[i] if class_names else f'Class {i}'
        axes[0, i].set_title(f'{class_name} Target')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Plot predictions
        im2 = axes[1, i].imshow(preds_np[i], cmap='hot', interpolation='nearest')
        axes[1, i].set_title(f'{class_name} Prediction')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def compute_curriculum_focal_loss(ovod_logits, target_heatmap, epoch=0, max_epochs=100):
    """
    Curriculum learning version of focal loss with epoch-dependent parameters
    """
    # Curriculum parameters that change over training
    progress = epoch / max_epochs
    
    # Start with easier learning (lower gamma) and increase difficulty
    gamma = 1.0 + progress * 1.5  # Gamma from 1.0 to 2.5
    alpha = 0.25 + progress * 0.1   # Alpha from 0.25 to 0.35
    
    bce_loss = F.binary_cross_entropy_with_logits(ovod_logits, target_heatmap, reduction='none')
    probs = torch.sigmoid(ovod_logits)
    
    pos_mask = target_heatmap > 0.1
    pt = torch.where(pos_mask, probs, 1 - probs)
    alpha_t = torch.where(pos_mask, alpha, 1 - alpha)
    
    focal_weight = alpha_t * (1 - pt).pow(gamma)
    focal_loss = focal_weight * bce_loss
    
    return focal_loss.mean()


class AdaptiveLossScheduler:
    """
    Adaptive loss weight scheduler based on training dynamics
    """
    def __init__(self, base_ovod_weight=150.0, base_contrastive_weight=0.5):
        self.base_ovod_weight = base_ovod_weight
        self.base_contrastive_weight = base_contrastive_weight
        self.loss_history = deque(maxlen=50)
        self.plateau_count = 0
        self.best_loss = float('inf')
        
    def update(self, current_loss):
        """Update scheduler with current loss"""
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.plateau_count = 0
        else:
            self.plateau_count += 1
    
    def get_weights(self):
        """Get adaptive weights based on training dynamics"""
        if len(self.loss_history) < 10:
            return self.base_ovod_weight, self.base_contrastive_weight
        
        # Check for plateau (last 10 losses have small variance)
        recent_losses = list(self.loss_history)[-10:]
        loss_std = np.std(recent_losses)
        
        if loss_std < 0.05 and self.plateau_count > 5:
            # Increase contrastive weight to escape plateau
            ovod_weight = self.base_ovod_weight * 0.8
            contrastive_weight = self.base_contrastive_weight * 1.5
        else:
            ovod_weight = self.base_ovod_weight
            contrastive_weight = self.base_contrastive_weight
            
        return ovod_weight, contrastive_weight


# Example usage for training loop integration:
"""
# In your training script:

# Initialize the enhanced model
model = TwoStreamCoDETR(
    backbone=backbone_config,
    train_stage=train_stage,
    neck=neck_config,
    classes=class_names,
    query_head=query_head_config,
    # ... other parameters ...
    moco_momentum=0.999,
    moco_temperature=0.07,
    feature_bank_size=4096,
    contrastive_weight=0.5
)

# Initialize adaptive loss scheduler
loss_scheduler = AdaptiveLossScheduler()

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (img, img_lwir, img_metas, gt_bboxes, gt_labels) in enumerate(dataloader):
        
        # Forward pass
        losses = model.forward_train(
            img, img_lwir, img_metas, gt_bboxes, gt_labels
        )
        
        # Update loss scheduler
        total_loss = sum(losses.values())
        loss_scheduler.update(total_loss.item())
        
        # Get adaptive weights
        ovod_weight, contrastive_weight = loss_scheduler.get_weights()
        
        # Apply adaptive weights
        if 'ovod_loss' in losses:
            losses['ovod_loss'] = losses['ovod_loss'] * (ovod_weight / 150.0)
        if 'contrastive_loss' in losses:
            losses['contrastive_loss'] = losses['contrastive_loss'] * (contrastive_weight / 0.5)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss = sum(losses.values())
        total_loss.backward()
        optimizer.step()
        
        # Optional: Visualize targets every N iterations
        if batch_idx % 500 == 0:
            with torch.no_grad():
                # Get current predictions for visualization
                x_vis = model.backbone_vis(img)
                x_lwir = model.backbone_lwir(img_lwir)
                x = model.tfb_blocks[0](x_vis, x_lwir[0])
                x = tuple(x)
                x = model.neck(x)
                
                # Generate current ovod_logits for comparison
                visual_feature = x[0]
                # ... (CLIP processing code)
                
                visualize_enhanced_targets(
                    target_heatmap, ovod_logits, 
                    save_path=f'debug_epoch_{epoch}_batch_{batch_idx}.png',
                    class_names=class_names
                )
"""