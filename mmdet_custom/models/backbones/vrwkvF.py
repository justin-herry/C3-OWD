# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from .base.vrwkv import VRWKV
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs
from mmdet_custom.models.utils import resize_pos_embed

_logger = logging.getLogger(__name__)

class VrwkvFusion(VRWKV):
    def __init__(self, 
                 in_channels: int = 3,
                 embed_dims: int = 256,
                 num_scales: int = 3,
                 pretrain_size: int = 224, 
                 conv_inplane: int = 64, 
                 n_points: int = 4, 
                 deform_num_heads: int = 8,
                 init_values_inject: float = 0., 
                 interaction_indexes: List[int] = [0, 2, 3, 4], 
                 with_cffn: bool = True, 
                 cffn_ratio: float = 0.25,
                 deform_ratio: float = 1.0, 
                 add_rwkv_feature: bool = True, 
                 use_extra_extractor: bool = True, 
                 with_cp: bool = False, 
                 *args, **kwargs):
        
        # Set configurable embed_dims before parent initialization
        self._embed_dims = embed_dims
        super().__init__(embed_dims=embed_dims, with_cp=with_cp, *args, **kwargs)
        
        # Configuration parameters
        self.in_channels = in_channels
        self.num_scales = num_scales
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes or [0, 2, 3, 4]
        self.add_rwkv_feature = add_rwkv_feature
        
        # Level embedding for multi-scale features
        self.level_embed = nn.Parameter(torch.zeros(num_scales, embed_dims))
        
        # Separate SPM modules for RGB and IR inputs
        self.spm_rgb = SpatialPriorModule(
            inplanes=conv_inplane, 
            embed_dim=embed_dims,
        )
        self.spm_ir = SpatialPriorModule(
            inplanes=conv_inplane, 
            embed_dim=embed_dims,
        )
        
        # Interaction blocks for cross-modal fusion
        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dims, 
                num_heads=deform_num_heads, 
                n_points=n_points,
                init_values=init_values_inject, 
                drop_path=self.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, 
                deform_ratio=deform_ratio,
                extra_extractor=((i == len(interaction_indexes) - 1) and use_extra_extractor),
                with_cp=with_cp)
            for i in range(len(self.interaction_indexes))
        ])
        
        # Normalization layers for output features
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(embed_dims) for _ in range(num_scales)
        ])

        # Initialize weights
        self.spm_rgb.apply(self._init_weights)
        self.spm_ir.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    @property
    def embed_dims(self) -> int:
        """Access embed_dims as property"""
        return self._embed_dims

    def _init_weights(self, m: nn.Module):
        """Weight initialization for different layer types"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Resize position embedding to match feature resolution"""
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, 
            size=(H, W), 
            mode='bicubic', 
            align_corners=False
        ).reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m: nn.Module):
        """Initialize weights for deformable attention modules"""
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Add level-specific embeddings to multi-scale features"""
        return [feat + self.level_embed[i] for i, feat in enumerate(features)]

    def _reshape_features(self, features: List[torch.Tensor], resolutions: List[Tuple[int, int]]) -> List[torch.Tensor]:
        """Reshape features to spatial dimensions"""
        reshaped = []
        for feat, (H, W) in zip(features, resolutions):
            reshaped.append(
                feat.transpose(1, 2).view(feat.size(0), self.embed_dims, H, W).contiguous()
            )
        return reshaped

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass for dual-modal input
        RGB as Query, IR as Key/Value
        Outputs features at multiple scales with configurable resolutions
        """
        # Separate RGB and IR inputs
        rgb_fea, ir_fea = x
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape
        
        # Generate deformable attention inputs from IR
        deform_inputs1_ir, deform_inputs2_ir = deform_inputs(ir_fea)

        # Process both modalities through Spatial Prior Modules (SPM)
        # Get multi-scale features and their resolutions
        rgb_features, rgb_resolutions = self.spm_rgb(rgb_fea, return_resolutions=True)
        ir_features, ir_resolutions = self.spm_ir(ir_fea, return_resolutions=True)
        
        # Verify we have the expected number of scales
        assert len(rgb_features) == self.num_scales
        assert len(ir_features) == self.num_scales
        
        # Add level embeddings to multi-scale features
        rgb_features = self._add_level_embed(rgb_features)
        ir_features = self._add_level_embed(ir_features)
        
        # Concatenate features for interaction
        c_rgb = torch.cat(rgb_features, dim=1)
        c_ir = torch.cat(ir_features, dim=1)

        # Patch embedding for both modalities
        x_rgb, patch_resolution = self.patch_embed(rgb_fea)  # RGB as Query
        x_ir, _ = self.patch_embed(ir_fea)                   # IR as Key/Value
        
        bs, n, dim = x_rgb.shape
        H, W = patch_resolution
        
        # Add position embeddings
        pos_embed = self._get_pos_embed(self.pos_embed, H, W)
        x_rgb = x_rgb + pos_embed
        x_ir = x_ir + pos_embed
        x_rgb = self.drop_after_pos(x_rgb)
        x_ir = self.drop_after_pos(x_ir)

        # Cross-modal interaction blocks
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            # RGB as Query, IR as Key/Value
            x_rgb, c_rgb = layer(
                x_rgb,  # Query features (RGB)
                c_ir,   # Key/Value features (IR)
                self.layers[indexes[0]:indexes[-1] + 1],
                deform_inputs1_ir,  # Deformable params from IR
                deform_inputs2_ir, 
                H, W
            )

        # Split output features
        start_idx = 0
        output_features = []
        for feat in rgb_features:
            end_idx = start_idx + feat.size(1)
            output_features.append(c_rgb[:, start_idx:end_idx, :])
            start_idx = end_idx
        
        # Reshape to spatial dimensions using resolutions from SPM
        output_features = self._reshape_features(output_features, rgb_resolutions)

        # Optional: Add original VRWKV features
        if self.add_rwkv_feature:
            # Base feature at patch resolution
            base_feature = x_rgb.transpose(1, 2).view(bs, dim, H, W).contiguous()
            
            # Create multi-scale features from base
            base_features = []
            for (target_H, target_W) in rgb_resolutions:
                scale_factor = target_H / H
                base_features.append(F.interpolate(
                    base_feature, 
                    scale_factor=scale_factor, 
                    mode='bilinear', 
                    align_corners=False
                ))
            
            # Add to output features
            output_features = [out + base for out, base in zip(output_features, base_features)]

        # Apply normalization
        normalized_features = [norm(feat) for norm, feat in zip(self.norms, output_features)]
        
        return normalized_features
    
