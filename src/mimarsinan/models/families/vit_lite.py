"""ViT-Lite with the CCT encoder block (hassani2021escaping): ViT-Lite-7/4 is the ViT gate's network.

Structure of the reference code (SHI-Labs Compact-Transformers ``ViTLite``, the gate's ``gate_models.py``,
``block="cct"``): a Conv2d tokenizer (kernel = stride = patch, bias), a learnable class token and positional embedding,
encoder layers ``x = x + drop_path(attn(pre_norm(x))); x = norm1(x); x = x + drop_path(linear2(gelu(linear1(x))))``
with a fused qkv Linear without bias, a final LayerNorm and the class token into a Linear classifier. The attention
reads its heads from ``qkv.reshape(b, n, 3, h, d).permute(2, 0, 3, 1, 4)`` and forms QK^T and PV as explicit
products of activations; its form is ``softmax`` or ``relu`` (P = relu(QK^T / sqrt(d)) / L, L the token count).
Parameter count of ViT-Lite-7/4: 3,716,874 at 10 classes, 3,740,004 at 100.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_FORMS = ("softmax", "relu")


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth per sample (the timm and CCT form); the identity in eval mode."""
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    mask = (keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)).floor_()
    return x.div(keep) * mask


class CCTAttention(nn.Module):
    def __init__(self, dim: int, heads: int, *, attention: str, attn_drop: float, proj_drop: float):
        super().__init__()
        if attention not in ATTENTION_FORMS:
            raise ValueError(f"unknown attention form {attention!r}; declared {ATTENTION_FORMS}")
        self.heads, self.head_dim, self.attention = heads, dim // heads, attention
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1) if self.attention == "softmax" else F.relu(attn) / n
        out = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class CCTEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, *, attention: str, attn_drop: float, drop: float,
                 drop_path_rate: float):
        super().__init__()
        self.pre_norm = nn.LayerNorm(dim)
        self.self_attn = CCTAttention(dim, heads, attention=attention, attn_drop=attn_drop, proj_drop=drop)
        self.linear1, self.dropout1 = nn.Linear(dim, mlp_dim), nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(dim)
        self.linear2, self.dropout2 = nn.Linear(mlp_dim, dim), nn.Dropout(drop)
        self.drop_path_rate = float(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + drop_path(self.self_attn(self.pre_norm(x)), self.drop_path_rate, self.training))
        y = self.linear2(self.dropout1(F.gelu(self.linear1(x))))
        return x + drop_path(self.dropout2(y), self.drop_path_rate, self.training)


class ViTLiteCCT(nn.Module):
    """ViT-Lite with CCT encoder layers; ``attention`` selects the softmax or the ReLU-attention block."""

    def __init__(self, *, num_classes: int = 100, img_size: int = 32, in_channels: int = 3, patch: int = 4,
                 dim: int = 256, depth: int = 7, heads: int = 4, mlp_ratio: float = 2.0, attention: str = "softmax",
                 attn_drop: float = 0.1, drop: float = 0.0, stochastic_depth: float = 0.1):
        super().__init__()
        if dim % heads or img_size % patch:
            raise ValueError(f"ViT-Lite needs dim % heads == 0 and img_size % patch == 0; got {dim}, {heads}, "
                             f"{img_size}, {patch}")
        tokens = (img_size // patch) ** 2
        self.tokenizer = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch, padding=0, bias=True)
        self.class_emb = nn.Parameter(torch.zeros(1, 1, dim))
        self.positional_emb = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        self.dropout = nn.Dropout(drop)
        rates = [float(r) for r in torch.linspace(0, stochastic_depth, depth)]
        self.blocks = nn.ModuleList(
            CCTEncoderLayer(dim, heads, int(dim * mlp_ratio), attention=attention, attn_drop=attn_drop, drop=drop,
                            drop_path_rate=rates[i]) for i in range(depth))
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_classes)
        self._initialize()

    def _initialize(self) -> None:
        """The reference initialization: Kaiming-normal tokenizer, positional embedding truncated normal (std 0.2),
        Linear weights truncated normal (std 0.02) with zero biases, LayerNorm weight 1 and bias 0."""
        nn.init.kaiming_normal_(self.tokenizer.weight)
        nn.init.trunc_normal_(self.positional_emb, std=0.2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer(x).flatten(2).transpose(1, 2)
        x = torch.cat((self.class_emb.expand(x.shape[0], -1, -1), x), dim=1) + self.positional_emb
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.norm(x)[:, 0])
