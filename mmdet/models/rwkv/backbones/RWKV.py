import math
from copy import copy
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# from utils.datasets import letterbox
# from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
# from utils.plots import colors, plot_one_box
# from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath
from mmcv.cnn.bricks.transformer import PatchEmbed
from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

from mmcv.ops import DeformConv2d
from mmcv.ops import DeformConv2dPack

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

from mmdet_custom.models.backbones.base.vrwkv import VRWKV,ChannelSpatialAttention,ChunkedCrossAttention
from mmdet_custom.models.backbones.adapter_modules import  InteractionBlock, DeformInputs,CrossModalExchange
from mmdet_custom.models.utils import resize_pos_embed



class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print("Concat inputs shape", torch.cat(x, self.d).shape)
        return torch.cat(x, self.d)



class NiNfusion(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(NiNfusion, self).__init__()

        self.concat = Concat(dimension=1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.concat(x)
        y = self.act(self.conv(y))

        return y

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv inputs shape", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

_logger = logging.getLogger(__name__)
class VrwkvFusion_improved_1(VRWKV):
    def __init__(self, 
                 in_channels: int = 128,
                 d_model: int = 256,  
                 h: int = 8,  
                 block_exp: int = 4,  
                 n_layer: int = 1,  
                 pretrain_size: int = 224, 
                 conv_inplane: int = 64, 
                 n_points: int = 4, 
                 patch_size: int = 4,
                 deform_num_heads: int = 8,
                 init_values_inject: float = 0., 
                 with_cffn: bool = False, 
                 cffn_ratio: float = 0.25,
                 deform_ratio: float = 1.0, 
                 add_rwkv_feature: bool = True, 
                 use_extra_extractor: bool = True, 
                 with_cp: bool = False, 
                 img_size: tuple = (320, 320),
                 *args, **kwargs):
        
        
        self._d_model = d_model
        super().__init__(in_channel = in_channels, embed_dims = d_model, with_cp=with_cp, *args, **kwargs)
        
        
        self.in_channels = in_channels
        self.h = h
        self.block_exp = block_exp
        self.n_layer = n_layer
        
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.add_rwkv_feature = add_rwkv_feature
        
        
        self.interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
        
        
        self.level_embed = nn.Parameter(torch.zeros(in_channels, d_model))
        self.conv1 = Conv(c1=128, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv2 = Conv(c1=512, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.conv3 = Conv(c1=1024, c2=d_model, k=1, s=1, p=0, g=1, act=True)

        self.conv4 = Conv(c1=d_model, c2=512, k=1, s=1, p=0, g=1, act=True)
        self.conv5 = Conv(c1=d_model, c2=1024, k=1, s=1, p=0, g=1, act=True)

        
        self.interactions_rgb = nn.Sequential(*[
            InteractionBlock(
                dim=d_model, 
                num_heads=deform_num_heads, 
                n_points=n_points,
                init_values=init_values_inject, 
                drop_path=self.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, 
                deform_ratio=deform_ratio,
                extra_extractor=((i == len(self.interaction_indexes) - 1) and use_extra_extractor),
                with_cp=with_cp)
            for i in range(len(self.interaction_indexes))
        ])
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=d_model,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=True)
       
        self.norm = nn.SyncBatchNorm(d_model)  
        self.up = nn.ConvTranspose2d(d_model, d_model, 2, 2)
        self.interactions_ir = nn.Sequential(*[
            InteractionBlock(
                dim=d_model, 
                num_heads=deform_num_heads, 
                n_points=n_points,
                init_values=init_values_inject, 
                drop_path=self.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio, 
                deform_ratio=deform_ratio,
                extra_extractor=((i == len(self.interaction_indexes) - 1) and use_extra_extractor),
                with_cp=with_cp)
            for i in range(len(self.interaction_indexes))
        ])

        self.interactions_rgb.apply(self._init_weights)
        self.interactions_ir.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        nn.init.normal_(self.level_embed)
        self.fusion2 = NiNfusion(
            512,256)
        self.fusion3 = NiNfusion(
            1024,512)
        self.fusion4 = NiNfusion(   
            2048,1024)
        self.deform_inputs = DeformInputs()
    @property
    def d_model(self) -> int:
        
        return self._d_model

    def _init_weights(self, m: nn.Module):
       
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
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
       
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):

        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        rgb_features = x[0:4]
        ir_features=x[4:]
        print(rgb_features[0].shape, ir_features[0].shape)
        deform_inputs1_rgb, deform_inputs2_rgb = self.deform_inputs(rgb_features[0])
        deform_inputs1_ir, deform_inputs2_ir = self.deform_inputs(ir_features[0])
        c1_rgb = self.conv1(rgb_features[0])
        c2_rgb = rgb_features[1]
        c3_rgb = self.conv2(rgb_features[2])
        c4_rgb = self.conv3(rgb_features[3])
        c1_ir = self.conv1(ir_features[0])
        c2_ir = ir_features[1]
        # import ipdb;ipdb.set_trace()    
        c3_ir = self.conv2(ir_features[2])
        c4_ir = self.conv3(ir_features[3])

        bs, dim, _, _ = c1_rgb.shape
        c2_rgb = c2_rgb.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3_rgb = c3_rgb.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4_rgb = c4_rgb.view(bs, dim, -1).transpose(1, 2)  # 32s
        c2_ir = c2_ir.view(bs, dim, -1).transpose(1, 2)
        c3_ir = c3_ir.view(bs, dim, -1).transpose(1, 2)
        c4_ir = c4_ir.view(bs, dim, -1).transpose(1, 2)
        c2_rgb,c3_rgb,c4_rgb = self._add_level_embed(c2_rgb,c3_rgb,c4_rgb)
        c2_ir,c3_ir,c4_ir = self._add_level_embed(c2_ir,c3_ir,c4_ir)
        c_rgb = torch.cat([c2_rgb, c3_rgb, c4_rgb], dim=1)
        c_ir = torch.cat([c2_ir, c3_ir, c4_ir], dim=1)
        
        # Patch Embedding forward
        
        x_rgb, patch_resolution = self.patch_embed(c1_rgb)
        bs, n, dim = x_rgb.shape
        H, W = patch_resolution

        x_rgb = x_rgb + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x_rgb = self.drop_after_pos(x_rgb)
        # Patch Embedding forward
        x_ir, _ = self.patch_embed(c1_ir)
        x_ir = x_ir + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        # Enhance IR/RGB_feature
        for i, layer in enumerate(self.interactions_ir):
            indexes = self.interaction_indexes[i] 
            x_ir, c_ir = layer(x_ir, c_ir, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1_ir, deform_inputs2_ir, H, W)
            x_rgb, c_rgb = layer(x_rgb, c_rgb, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1_ir, deform_inputs2_ir, H, W)
            x_ir, c_rgb = layer(x_ir, c_rgb, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1_ir, deform_inputs2_ir, H, W)
            x_rgb, c_ir = layer(x_rgb, c_ir, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1_ir, deform_inputs2_ir, H, W)
            

        # Split & Reshape
        c2_rgb = c_rgb[:, 0:c2_rgb.size(1), :]
        c3_rgb = c_rgb[:, c2_rgb.size(1):c2_rgb.size(1) + c3_rgb.size(1), :]
        c4_rgb = c_rgb[:, c2_rgb.size(1) + c3_rgb.size(1):, :]

        c2_rgb = c2_rgb.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3_rgb = c3_rgb.transpose(1, 2).view(bs, dim, H , W ).contiguous()
        c4_rgb = c4_rgb.transpose(1, 2).view(bs, dim, H // 2, W //2).contiguous()

        c1_rgb = self.up(c2_rgb) + c1_rgb


        c2_ir = c_ir[:, 0:c2_ir.size(1), :]
        c3_ir = c_ir[:, c2_ir.size(1):c2_ir.size(1) + c3_ir.size(1), :]    
        c4_ir = c_ir[:, c2_ir.size(1) + c3_ir.size(1):, :]
        c2_ir = c2_ir.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3_ir = c3_ir.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4_ir = c4_ir.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1_ir = self.up(c2_ir) + c1_ir

        if self.add_rwkv_feature:
            x3_rgb = x_rgb.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1_rgb = F.interpolate(x3_rgb, scale_factor=4, mode='bilinear', align_corners=False)
            x2_rgb = F.interpolate(x3_rgb, scale_factor=2, mode='bilinear', align_corners=False)
            x4_rgb = F.interpolate(x3_rgb, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1_rgb, c2_rgb, c3_rgb, c4_rgb = c1_rgb + x1_rgb, c2_rgb + x2_rgb, c3_rgb + x3_rgb, c4_rgb + x4_rgb
            x3_ir = x_ir.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1_ir = F.interpolate(x3_ir, scale_factor=4, mode='bilinear', align_corners=False)
            x2_ir = F.interpolate(x3_ir, scale_factor=2, mode='bilinear', align_corners=False)
            x4_ir = F.interpolate(x3_ir, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1_ir, c2_ir, c3_ir, c4_ir = c1_ir + x1_ir, c2_ir + x2_ir, c3_ir + x3_ir, c4_ir + x4_ir
        c3_rgb = self.conv4(c3_rgb)
        c4_rgb = self.conv5(c4_rgb)
        c3_ir = self.conv4(c3_ir)
        c4_ir = self.conv5(c4_ir)

        f2 = self.fusion2([c2_rgb, c2_ir])
        f3 = self.fusion3([c3_rgb, c3_ir])
        f4 = self.fusion4([c4_rgb, c4_ir])

        return [f2, f3, f4]
class Get_feature(nn.Module):
    def __init__(self, layer, d_model):
        super(Get_feature, self).__init__()
        self.layer = layer
    def forward(self, x):
        return x[self.layer]