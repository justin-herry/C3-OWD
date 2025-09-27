# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from .base.vrwkv import VRWKV
from .adapter_modules import SpatialPriorModule, InteractionBlock, DeformInputs
from mmdet_custom.models.utils import resize_pos_embed
from mmcv.cnn.bricks.transformer import PatchEmbed
_logger = logging.getLogger(__name__)


class VRWKV_Encoder(VRWKV):
    def __init__(self, in_channel, output_channel, embed_dims = 512, pretrain_size=640,  n_points=4, deform_num_heads=8, patch_size=2,
                 init_values_inject=0., interaction_indexes=[0, 2, 3, 4], with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_rwkv_feature=True, use_extra_extractor=True, with_cp=False, *args, **kwargs):
        """
        Args:
            in_channel (list): input channel of each feature map, e.g., [256, 512, 1024]
            output_channel (list): output channel of each feature map, e.g., [256, 512, 1024]
            embed_dims (int): embedding dimension, default 512
            pretrain_size (int): input image size, default 640
        """
        super().__init__()
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_rwkv_feature = add_rwkv_feature
        self._embed_dims = embed_dims
        self.embed_dims = embed_dims
        self.ln1 = None  # aviod unused error
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dims))
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dims, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values_inject, drop_path=self.drop_path_rate,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        max_h, max_w = 80, 80
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_h*max_w, self.embed_dims)
)
        self.patch_embed = PatchEmbed(
            in_channels=in_channel[0],
            input_size=pretrain_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        self.input_proj1 = nn.Linear(in_channel[0], embed_dims)
        self.input_proj2 = nn.Linear(in_channel[1], embed_dims)
        self.input_proj3 = nn.Linear(in_channel[2], embed_dims)
        self.output_proj1 = nn.Linear(embed_dims, output_channel[0])
        self.output_proj2 = nn.Linear(embed_dims, output_channel[1])
        self.output_proj3 = nn.Linear(embed_dims, output_channel[2])
        self.norm2 = nn.BatchNorm2d(output_channel[0])
        self.norm3 = nn.BatchNorm2d(output_channel[1])
        self.norm4 = nn.BatchNorm2d(output_channel[2])

        self.interactions.apply(self._init_weights)
        # self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        self.deform_inputs = DeformInputs()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    # def _init_deform_weights(self, m):
    #     if isinstance(m, MSDeformAttn):
    #         m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, feat, x):
        """
        Args:
            x (torch.Tensor): input ir_feat, shape (b, h/8, w/8, c)
            rgb_feat (list): list of features from backbone, e.g., [c2, c3, c4]
                where c2, c3, c4 are features from backbone
                shape: (b, c, h/8, w/8), (b, c, h/16, w/16), (b, c, h/32, w/32)
        Returns:
            list: output features after interaction, e.g., [c2, c3, c4]
                where c2, c3, c4 are features after interaction
                shape: (b, c, h/8, w/8), (b, c, h/4, w/4), (b, c, h/2, w/2)
        """
        # x = x.permute(0, 3, 1, 2).contiguous()  # (b, h/8, w/8, c)
        deform_inputs1, deform_inputs2 = self.deform_inputs(x)
        c2, c3, c4 = feat # c2, c3, c4 are features from backbone
        c2 = c2.permute(0,2,3,1).contiguous()
        c3 = c3.permute(0,2,3,1).contiguous()
        c4 = c4.permute(0,2,3,1).contiguous()
        c2= self.input_proj1(c2.view(c2.shape[0], -1, c2.shape[3]))
        c3 = self.input_proj2(c3.view(c3.shape[0], -1, c3.shape[3]))
        c4 = self.input_proj3(c4.view(c4.shape[0], -1, c4.shape[3]))
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, patch_resolution = self.patch_embed(x)
        bs, n, dim = x.shape
        H, W = patch_resolution
        # import ipdb; ipdb.set_trace()
        pos_embed = self.pos_embed[:, :H*W, :]
        x = x + resize_pos_embed(
            pos_embed,
            patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens).to(x.device)
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes:indexes + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :] + c2
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :] + c3
        c4 = c[:, c2.size(1) + c3.size(1):, :] + c4
        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()

        if self.add_rwkv_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4


        # Final Norm
        f2 = self.norm2(self.output_proj1(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        f3 = self.norm3(self.output_proj2(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        f4 = self.norm4(self.output_proj3(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))

        return [f2, f3, f4]

class VRWKV_Decoder(VRWKV):
    def __init__(self, in_channel, output_channel, embed_dims = 512, pretrain_size=640,  n_points=4, deform_num_heads=8, patch_size=2,
                 init_values_inject=0., interaction_indexes=[0, 2, 3, 4], with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_rwkv_feature=True, use_extra_extractor=True, with_cp=False, *args, **kwargs):
        """
        Args:
            in_channel (list): input channel of each feature map, e.g., [256, 512, 1024]
            output_channel (list): output channel of each feature map, e.g., [256, 512, 1024]
            embed_dims (int): embedding dimension, default 512
         """
        super().__init__()
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_rwkv_feature = add_rwkv_feature
        self._embed_dims = embed_dims
        self.embed_dims = embed_dims
        self.ln1 = None  # aviod unused error
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dims))
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dims, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values_inject, drop_path=self.drop_path_rate,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        max_h, max_w = 80, 80
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_h*max_w, self.embed_dims)
)
        self.patch_embed = PatchEmbed(
            in_channels=in_channel[0],
            input_size=pretrain_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
        self.input_proj1 = nn.Linear(in_channel[0], embed_dims)
        self.input_proj2 = nn.Linear(in_channel[1], embed_dims)
        self.input_proj3 = nn.Linear(in_channel[2], embed_dims)
        self.output_proj1 = nn.Linear(embed_dims, output_channel[0])
        self.output_proj2 = nn.Linear(embed_dims, output_channel[1])
        self.output_proj3 = nn.Linear(embed_dims, output_channel[2])
        self.up = nn.ConvTranspose2d(embed_dims, embed_dims, 2, 2)
        self.norm2 = nn.BatchNorm2d(output_channel[0])
        self.norm3 = nn.BatchNorm2d(output_channel[1])
        self.norm4 = nn.BatchNorm2d(output_channel[2])

        self.up.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        # self.apply(self._init_deform_weights)
        normal_(self.level_embed)
        self.deform_inputs = DeformInputs()
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    # def _init_deform_weights(self, m):
    #     if isinstance(m, MSDeformAttn):
    #         m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, feat):
        """
        Args: 
            feat (list): list of features from backbone, e.g., [c2, c3, c4]
                where c2, c3, c4 are features from backbone
                shape: (b, c, h/8, w/8), (b, c, h/16, w/16), (b, c, h/32, w/32)
        Returns:    
            list: output features after interaction, e.g., [f2, f3, f4]
                where f2, f3, f4 are features after interaction
                shape: (b, c, h/8, w/8), (b, c, h/4, w/4), (b, c, h/2, w/2)
        """
        x = feat[0]
        # x = x.permute(0, 2, 3, 1).contiguous()  # (b, h/8, w/8, c)
        deform_inputs1, deform_inputs2 = self.deform_inputs(x)
        c2, c3, c4 = feat 
        c2 = c2.permute(0,2,3,1).contiguous()
        c3 = c3.permute(0,2,3,1).contiguous()
        c4 = c4.permute(0,2,3,1).contiguous()# c2, c3, c4 are features from backbone
        c2= self.input_proj1(c2.view(c2.shape[0], -1, c2.shape[3]))
        c3 = self.input_proj2(c3.view(c3.shape[0], -1, c3.shape[3]))
        c4 = self.input_proj3(c4.view(c4.shape[0], -1, c4.shape[3]))
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, patch_resolution = self.patch_embed(x)
        bs, n, dim = x.shape
        H, W = patch_resolution
        pos_embed = self.pos_embed[:, :H*W, :]
        x = x + resize_pos_embed(
            pos_embed,
            patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens).to(x.device)
        x = self.drop_after_pos(x)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes:indexes + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()

        if self.add_rwkv_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c2, c3, c4 = c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f2 = self.norm2(self.output_proj1(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        f3 = self.norm3(self.output_proj2(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        f4 = self.norm4(self.output_proj3(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return [f2, f3, f4]

class Rwkv_Layer(nn.Module):
    def __init__(self, in_channel=[256, 512, 1024], output_channel=[256, 512, 1024], embed_dims=512):
        super().__init__()
        self.encoder = VRWKV_Encoder(in_channel=in_channel, output_channel=output_channel, embed_dims=embed_dims)
        self.decoder = VRWKV_Decoder(in_channel=in_channel, output_channel=output_channel, embed_dims=embed_dims)
    def forward(self, x, feat):
        encoder_out = self.encoder(x, feat)
        out = self.decoder(encoder_out)
        return out
        