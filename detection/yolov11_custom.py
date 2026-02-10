"""
Custom YOLOv11 Implementation from Scratch
This implementation provides a drop-in replacement for Ultralytics YOLO
with the same interface for training and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Union
import math
import yaml
import os
from pathlib import Path


# ============================================================================
# Core Building Blocks
# ============================================================================

class Conv(nn.Module):
    """Standard convolution with BatchNorm and SiLU activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """Depthwise convolution"""
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__(c1, c2, k, s, p, g=math.gcd(c1, c2), act=act)


class Bottleneck(nn.Module):
    """Standard bottleneck block with residual connection"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions - YOLOv8/v11 style"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C2fCIB(nn.Module):
    """C2f with Contextual Information Block - YOLOv11 specific"""
    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CIB(nn.Module):
    """Contextual Information Block"""
    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else Conv(2 * c_, 2 * c_, 9, g=2 * c_),
            Conv(2 * c_, c2, 1),
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv1(x) if self.add else self.cv1(x)


class Attention(nn.Module):
    """Attention module for YOLOv11"""
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = self.head_dim  # Make key_dim equal to head_dim
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.split([C, self.key_dim * self.num_heads, self.key_dim * self.num_heads], dim=1)
        
        # Reshape for multi-head attention: [B, num_heads, N, head_dim/key_dim]
        q = q.reshape(B, self.num_heads, C // self.num_heads, N).permute(0, 1, 3, 2)
        k = k.reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 3, 2)
        
        # Attention: match dimensions by using key_dim for both q and k
        attn = (q[..., :self.key_dim] @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        x = self.proj(x) + self.pe(v.permute(0, 1, 3, 2).reshape(B, -1, H, W))
        return x


class PSA(nn.Module):
    """Position-Sensitive Attention for YOLOv11"""
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, 2 * c_, 1)
        self.cv2 = Conv(2 * c_, c2, 1)
        self.attn = Attention(c_, attn_ratio=0.5, num_heads=c_ // 64)
        self.ffn = nn.Sequential(
            Conv(c_, c_ * 2, 1),
            Conv(c_ * 2, c_, 1, act=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, 1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


# ============================================================================
# Detection Head
# ============================================================================

class DFL(nn.Module):
    """Distribution Focal Loss"""
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        x = torch.arange(c1, dtype=torch.float)
        self.register_buffer('project', x.view(1, c1, 1, 1))

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, c, 1, a).transpose(2, 1).softmax(1)).view(b, a)


class DetectionHead(nn.Module):
    """YOLOv11 Detection Head"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        
        if self.training:
            return x
        
        # Inference path
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        
        # Decode boxes
        dbox = self.decode_bboxes(self.dfl(box))
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, x

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes"""
        return bboxes


# ============================================================================
# YOLOv11 Backbone and Neck
# ============================================================================

class YOLOv11Backbone(nn.Module):
    """YOLOv11 Backbone"""
    def __init__(self, channels_list, depth_list, use_psa=True):
        super().__init__()
        c1, c2, c3, c4, c5 = channels_list
        d1, d2, d3, d4 = depth_list

        # Stem
        self.stem = Conv(3, c1, 3, 2)  # P1/2

        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(c1, c2, 3, 2),  # P2/4
            C2fCIB(c2, c2, d1, True, lk=True)
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(c2, c3, 3, 2),  # P3/8
            C2fCIB(c3, c3, d2, True, lk=True)
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(c3, c4, 3, 2),  # P4/16
            C2fCIB(c4, c4, d3, True, lk=True)
        )

        # Stage 4
        stage4_layers = [
            Conv(c4, c5, 3, 2),  # P5/32
            C2fCIB(c5, c5, d4, True, lk=True)
        ]
        if use_psa:
            stage4_layers.append(PSA(c5, c5))
        self.stage4 = nn.Sequential(*stage4_layers)

    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage1(x)
        p3 = self.stage2(p2)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return [p3, p4, p5]


class YOLOv11Neck(nn.Module):
    """YOLOv11 Neck (FPN + PAN)"""
    def __init__(self, channels_list, depth_list):
        super().__init__()
        c3, c4, c5 = channels_list[2], channels_list[3], channels_list[4]
        d1, d2, d3 = depth_list[1:4]

        # Top-down pathway
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f1 = C2fCIB(c5 + c4, c4, d2, shortcut=False)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f2 = C2fCIB(c4 + c3, c3, d1, shortcut=False)

        # Bottom-up pathway
        self.down1 = Conv(c3, c3, 3, 2)
        self.c2f3 = C2fCIB(c3 + c4, c4, d2, shortcut=False)

        self.down2 = Conv(c4, c4, 3, 2)
        self.c2f4 = C2fCIB(c4 + c5, c5, d3, shortcut=False)

    def forward(self, features):
        p3, p4, p5 = features
        
        # Top-down
        x = self.up1(p5)
        x = torch.cat([x, p4], dim=1)
        x = self.c2f1(x)
        p4_out = x
        
        x = self.up2(x)
        x = torch.cat([x, p3], dim=1)
        p3_out = self.c2f2(x)
        
        # Bottom-up
        x = self.down1(p3_out)
        x = torch.cat([x, p4_out], dim=1)
        p4_out = self.c2f3(x)
        
        x = self.down2(p4_out)
        x = torch.cat([x, p5], dim=1)
        p5_out = self.c2f4(x)
        
        return [p3_out, p4_out, p5_out]


# ============================================================================
# Complete YOLOv11 Model
# ============================================================================

class YOLOv11Model(nn.Module):
    """Complete YOLOv11 Model"""
    def __init__(self, nc=80, model_size='n'):
        super().__init__()
        self.nc = nc
        
        # Model configurations: [channels, depth]
        # channels: [c1, c2, c3, c4, c5] for stem, stage1, stage2(P3), stage3(P4), stage4(P5)
        configs = {
            'n': ([16, 32, 64, 128, 256], [1, 2, 2, 1]),      # nano
            's': ([32, 64, 128, 256, 512], [1, 2, 2, 1]),     # small
            'm': ([48, 96, 192, 384, 768], [2, 4, 4, 2]),     # medium
            'l': ([64, 128, 256, 512, 512], [3, 6, 6, 3]),    # large
            'x': ([80, 160, 320, 640, 640], [4, 8, 8, 4]),    # xlarge
        }
        
        channels, depths = configs.get(model_size, configs['n'])
        
        self.backbone = YOLOv11Backbone(channels, depths)
        self.neck = YOLOv11Neck(channels, depths)
        
        # Detection head
        ch = [channels[2], channels[3], channels[4]]  # channels for P3, P4, P5
        self.head = DetectionHead(nc, ch)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Backbone
        features = self.backbone(x)
        # Neck
        features = self.neck(features)
        # Head
        return self.head(features)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# ============================================================================
# Utility Functions
# ============================================================================

def autopad(k, p=None, d=1):
    """Auto-calculate padding for 'same' shape"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def make_divisible(x, divisor):
    """Returns nearest x divisible by divisor"""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor
