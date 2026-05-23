# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')

import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from functools import partial

# 导入底层网络积木
from network_modules import *
try:
    from Utils import *
except ImportError:
    pass

class RefineNet(nn.Module):
    def __init__(self, cfg=None, c_in=3, n_view=1):
        """
        医疗定制版 RefineNet (6-DoF 位姿回归)
        @c_in: 默认为 3。严格对应 Mask(1) + Contour(1) + Depth(1)
        """
        super().__init__()
        self.cfg = cfg
        
        # 兼容 dict 和 object 两种 cfg 访问方式
        use_BN = cfg.use_BN if hasattr(cfg, 'use_BN') else cfg.get('use_BN', True)
        rot_rep = cfg.rot_rep if hasattr(cfg, 'rot_rep') else cfg.get('rot_rep', '6d')

        if use_BN:
            norm_layer = nn.BatchNorm2d
            norm_layer1d = nn.BatchNorm1d
        else:
            norm_layer = None
            norm_layer1d = None

        # --- 1. 孪生特征提取网络（A 和 B 共享此权重） ---
        # 输入: c_in=3 -> 输出: 128维特征图
        self.encodeA = nn.Sequential(
            ConvBNReLU(C_in=c_in, C_out=64, kernel_size=7, stride=2, norm_layer=norm_layer),
            ConvBNReLU(C_in=64, C_out=128, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(128, 128, bias=True, norm_layer=norm_layer),
        )

        # --- 2. 特征融合网络 ---
        # 接收 A(128) + B(128) 拼接后的 256 维特征，输出 512 维特征图
        self.encodeAB = nn.Sequential(
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(256, 256, bias=True, norm_layer=norm_layer),
            ConvBNReLU(256, 512, kernel_size=3, stride=2, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
            ResnetBasicBlock(512, 512, bias=True, norm_layer=norm_layer),
        )

        embed_dim = 512
        num_heads = 4
        
        # 扩展序列容量至 10000，完美适配 960x540 分辨率下的序列展平
        self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=10000)

        # --- 3. 平移预测头 (Translation Head) ---
        self.trans_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, 3),
        )

        # --- 4. 旋转预测头 (Rotation Head) ---
        if rot_rep == 'axis_angle':
            rot_out_dim = 3
        elif rot_rep == '6d':
            rot_out_dim = 6
        else:
            raise RuntimeError(f"Unsupported rot_rep: {rot_rep}")
            
        self.rot_head = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
            nn.Linear(512, rot_out_dim),
        )


    def forward(self, A, B):
        """
        前向传播
        @A: (B, 3, H, W) - Source (渲染图 / Base Pose)
        @B: (B, 3, H, W) - Target (术中图 / 观测姿态)
        """
        bs = len(A)
        output = {}

        # 1. 孪生特征提取：将 A 和 B 在 Batch 维度拼接，共用一次 encodeA 进行计算（大幅提升推理效率）
        # x.shape: (2B, 3, H, W)
        x = torch.cat([A, B], dim=0)
        x = self.encodeA(x)   # 输出 shape: (2B, 128, H', W')
        
        # 2. 提取完特征后重新拆分回 a 和 b
        a = x[:bs]  # (B, 128, H', W')
        b = x[bs:]  # (B, 128, H', W')

        # 3. 通道级融合：在 Channel 维度拼接
        ab = torch.cat((a, b), dim=1).contiguous()  # (B, 256, H', W')
        ab = self.encodeAB(ab)                      # (B, 512, H'', W'')

        # 4. 序列化与位置编码
        # 将 2D 特征图展平为 1D 序列: (B, 512, L) -> permute -> (B, L, 512)
        ab = ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1)
        ab = self.pos_embed(ab)

        # 5. Transformer 预测并进行全局平均池化 (mean)
        # self.trans_head(ab) 输出 (B, L, 3)，.mean(dim=1) 后变成 (B, 3)
        output['trans'] = self.trans_head(ab).mean(dim=1)
        output['rot'] = self.rot_head(ab).mean(dim=1)

        return output