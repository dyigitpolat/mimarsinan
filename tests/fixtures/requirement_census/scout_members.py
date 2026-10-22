"""Scout network instances copied verbatim for the requirement-census pins (unit tests read no research-tree file).

Definitions from RA/10_wave1/oracles/net_models.py (sha256 a538dbb191e833f5) and req_census.py (sha256 296ff421dcc7088b);
``MEMBERS`` lists the oracle members (``req_census.members()``) whose classes are copied here, same keys and order.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.cifar_models import CifarBasicBlock

NC = 10
VSHAPE = (3, 32, 32)

class ExplicitAttention(nn.Module):
    """timm-style attention: fused qkv Linear, explicit Q K^T and A V products."""

    def __init__(self, d, heads, softmax="method", score_act="softmax"):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.scale = self.dh ** -0.5
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.softmax = softmax
        self.score_act = score_act

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.softmax == "sdpa":
            o = F.scaled_dot_product_attention(q, k, v)
        else:
            a = (q @ k.transpose(-2, -1)) * self.scale
            if self.score_act == "relu":
                a = F.relu(a) * (1.0 / n)          # ReLU attention (softmax replaced, Wortsman et al. 2023)
            elif self.softmax == "method":
                a = a.softmax(dim=-1)
            else:
                a = F.softmax(a, dim=-1)
            o = a @ v
        o = o.transpose(1, 2).reshape(b, n, d)
        return self.proj(o)


class BatchFirstAttention(nn.Module):
    """Separate q, k, v Linears, heads split with view + transpose(1, 2) (batch axis never moves)."""

    def __init__(self, d, heads, score_act="softmax"):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.scale = self.dh ** -0.5
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.score_act = score_act

    def forward(self, x):
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(b, n, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(b, n, self.h, self.dh).transpose(1, 2)
        a = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if self.score_act == "relu":
            a = F.relu(a) * (1.0 / n)
        else:
            a = F.softmax(a, dim=-1)
        o = torch.matmul(a, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(o)


class ExplicitViTBlock(nn.Module):
    def __init__(self, d, heads, mlp=2, softmax="method", norm="ln", act="gelu", score_act="softmax"):
        super().__init__()
        if norm == "ln":
            self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        else:
            from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
            self.n1, self.n2 = ChannelsLastBatchNorm1d(d), ChannelsLastBatchNorm1d(d)
        self.attn = (BatchFirstAttention(d, heads, score_act=score_act) if softmax == "bf"
                     else ExplicitAttention(d, heads, softmax=softmax, score_act=score_act))
        self.fc1 = nn.Linear(d, mlp * d)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        self.fc2 = nn.Linear(mlp * d, d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.fc2(self.act(self.fc1(self.n2(x))))


class ExplicitViT(nn.Module):
    """Patchify conv, positional add, explicit-attention blocks, mean-token head (no cls token)."""

    def __init__(self, d=64, heads=2, depth=2, patch=4, **kw):
        super().__init__()
        n = (32 // patch) ** 2
        self.embed = nn.Conv2d(3, d, patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, n, d) * 0.02)
        self.blocks = nn.Sequential(*[ExplicitViTBlock(d, heads, **kw) for _ in range(depth)])
        self.norm = nn.LayerNorm(d) if kw.get("norm", "ln") == "ln" else nn.Identity()
        self.head = nn.Linear(d, NC)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        x = self.blocks(x)
        return self.head(self.norm(x).mean(dim=1))


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, t, act=nn.ReLU6, groups_dw=True, se=False):
        super().__init__()
        h = cin * t
        self.use_res = stride == 1 and cin == cout
        self.expand = nn.Sequential(nn.Conv2d(cin, h, 1, bias=False), nn.BatchNorm2d(h), act())
        self.dw = nn.Sequential(nn.Conv2d(h, h, 3, stride=stride, padding=1, groups=h if groups_dw else 1,
                                          bias=False), nn.BatchNorm2d(h), act())
        self.se = SqueezeExcite(h) if se else None
        self.project = nn.Sequential(nn.Conv2d(h, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = self.dw(self.expand(x))
        if self.se is not None:
            y = self.se(y)
        y = self.project(y)
        return x + y if self.use_res else y


class SqueezeExcite(nn.Module):
    """EfficientNet SE: global pool, 1x1 reduce, SiLU, 1x1 expand, sigmoid, channel-wise gate (act x act)."""

    def __init__(self, c, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, max(4, c // r), 1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(max(4, c // r), c, 1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        return x * self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))


class MobileNetV2Small(nn.Module):
    def __init__(self, act=nn.ReLU6, groups_dw=True, se=False):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), act())
        self.blocks = nn.Sequential(InvertedResidual(16, 16, 1, 2, act, groups_dw, se),
                                    InvertedResidual(16, 24, 2, 2, act, groups_dw, se),
                                    InvertedResidual(24, 24, 1, 2, act, groups_dw, se))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(24, NC)

    def forward(self, x):
        return self.head(torch.flatten(self.pool(self.blocks(self.stem(x))), 1))


def cbr(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


class UNetBlock(nn.Module):
    """One U-Net level (Ronneberger et al. 2015): two conv-BN-ReLU, max pool, bottleneck, upsample, skip concat."""

    def __init__(self, c=16, up="transpose"):
        super().__init__()
        self.enc = nn.Sequential(cbr(3, c), cbr(c, c))
        self.pool = nn.MaxPool2d(2)
        self.mid = nn.Sequential(cbr(c, 2 * c), cbr(2 * c, 2 * c))
        if up == "transpose":
            self.up = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        elif up == "nearest":
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(2 * c, c, 1))
        elif up == "bilinear":
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                    nn.Conv2d(2 * c, c, 1))
        else:  # pixel shuffle
            self.up = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 1), nn.PixelShuffle(2))
        self.dec = nn.Sequential(cbr(2 * c, c), nn.Conv2d(c, 4, 1))
        self.head = nn.Linear(4 * 32 * 32, NC)

    def forward(self, x):
        s = self.enc(x)
        y = self.up(self.mid(self.pool(s)))
        y = self.dec(torch.cat([s, y], dim=1))
        return self.head(torch.flatten(y, 1))


# req_census.py (its ``M`` is net_models, whose classes are this module's)
M = sys.modules[__name__]


def resnet18_cifar(w=0.25, nc=100):
    widths = [max(8, int(c * w / 8 + 0.5) * 8) for c in (64, 128, 256, 512)]

    class R18(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem_conv = nn.Conv2d(3, widths[0], 3, padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(widths[0])
            self.stem_relu = nn.ReLU()
            blocks, cin = [], widths[0]
            for s, c in enumerate(widths):
                for b in range(2):
                    blocks.append(CifarBasicBlock(cin, c, stride=2 if (s > 0 and b == 0) else 1))
                    cin = c
            self.blocks = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(cin, nc)

        def forward(self, x):
            x = self.stem_relu(self.stem_bn(self.stem_conv(x)))
            return self.fc(self.flatten(self.pool(self.blocks(x))))
    return R18()


def vit_lite(**kw):
    return M.ExplicitViT(d=256, heads=4, depth=7, patch=4, **kw)


MEMBERS = (
    ("resnet18_cifar_w1/4", lambda: resnet18_cifar(0.25), VSHAPE),
    ("mobilenetv2_small_relu6", lambda: MobileNetV2Small(), VSHAPE),
    ("mobilenetv2_small_relu6_dense_twin", lambda: MobileNetV2Small(groups_dw=False), VSHAPE),
    ("efficientnet_mbconv_se", lambda: MobileNetV2Small(act=nn.SiLU, se=True), VSHAPE),
    ("efficientnet_mbconv_se_dense_twin", lambda: MobileNetV2Small(act=nn.SiLU, groups_dw=False, se=True), VSHAPE),
    ("vit_lite_7_4_softmax_ln_gelu", lambda: vit_lite(softmax="bf"), VSHAPE),
    ("vit_lite_7_4_bn_relu_reluattn", lambda: vit_lite(softmax="bf", norm="bn", act="relu", score_act="relu"), VSHAPE),
    ("unet_level_convtranspose", lambda: UNetBlock(up="transpose"), VSHAPE),
)
