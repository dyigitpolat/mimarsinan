"""MHA-leaf vision transformer family for torch_mapping conversion.

The declared conversion subset for the ViT vehicle: LayerNorm pre-norm
encoder blocks whose attention is a single ``nn.MultiheadAttention``
(``batch_first=True``) FX LEAF — the frontend hosts the whole attention as
one ComputeOp instead of tracing through fused-qkv raw ops. Patchify is a
strided Conv2d; tokens carry a learnable cls token and positional embedding;
the MLP is Linear-GELU-Linear; the head reads the cls token after a final
LayerNorm.

Also owns the timm weight-porting shim: ``timm_vit_to_leaf_state_dict`` /
``load_timm_vit_state_dict`` map a timm ``vision_transformer``-layout state
dict (``deit_tiny_patch16_224``-style: fused ``attn.qkv`` Linear,
``patch_embed.proj``, ``mlp.fc1/fc2``) onto this leaf form. The fused qkv
rows land directly on ``in_proj_weight``/``in_proj_bias`` (both stack q, k, v
along dim 0), so the port is pure key remapping — no transposes.
"""

from __future__ import annotations

import re
from typing import Dict, Mapping

import torch
import torch.nn as nn


class LeafViTBlock(nn.Module):
    """Pre-LN transformer encoder block with an MHA-leaf attention."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + attn_out
        return x + self.fc2(self.act(self.fc1(self.norm2(x))))


class LeafVisionTransformer(nn.Module):
    """Vision transformer in MHA-leaf form (conv patchify + cls + pos + blocks + head)."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 192,
        num_heads: int = 3,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size={image_size} must be divisible by patch_size={patch_size}"
            )
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            LeafViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def tiny_test_vit(num_classes: int = 10) -> LeafVisionTransformer:
    """Small fast-test config: 8px image, 4px patches, d=16, 2 heads, 1 block."""
    return LeafVisionTransformer(
        image_size=8, patch_size=4, in_channels=3,
        embed_dim=16, num_heads=2, depth=1, mlp_ratio=2.0,
        num_classes=num_classes,
    )


def deit_tiny_leaf(num_classes: int = 1000) -> LeafVisionTransformer:
    """DeiT-Tiny geometry in leaf form: d=192, 3 heads, 12 blocks, patch 16, 197 tokens."""
    return LeafVisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        embed_dim=192, num_heads=3, depth=12, mlp_ratio=4.0,
        num_classes=num_classes,
    )


# ── timm weight-porting shim ─────────────────────────────────────────────────

_TIMM_DIRECT_KEY_MAP: Dict[str, str] = {
    "cls_token": "cls_token",
    "pos_embed": "pos_embed",
    "patch_embed.proj.weight": "patch_embed.weight",
    "patch_embed.proj.bias": "patch_embed.bias",
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
    "head.weight": "head.weight",
    "head.bias": "head.bias",
}

_TIMM_BLOCK_SUFFIX_MAP: Dict[str, str] = {
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    # timm's fused qkv Linear stacks (q, k, v) along dim 0 — exactly the
    # nn.MultiheadAttention in_proj layout, so the rows copy verbatim.
    "attn.qkv.weight": "attn.in_proj_weight",
    "attn.qkv.bias": "attn.in_proj_bias",
    "attn.proj.weight": "attn.out_proj.weight",
    "attn.proj.bias": "attn.out_proj.bias",
    "norm2.weight": "norm2.weight",
    "norm2.bias": "norm2.bias",
    "mlp.fc1.weight": "fc1.weight",
    "mlp.fc1.bias": "fc1.bias",
    "mlp.fc2.weight": "fc2.weight",
    "mlp.fc2.bias": "fc2.bias",
}

_TIMM_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def timm_vit_to_leaf_state_dict(
    timm_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remap a timm ``vision_transformer``-layout state dict to leaf-form keys.

    Every key must be recognized — an unmapped key raises ``KeyError`` (a
    silently ignored tensor would mean silently dropped weights).
    """
    out: Dict[str, torch.Tensor] = {}
    for key, value in timm_state.items():
        if key in _TIMM_DIRECT_KEY_MAP:
            out[_TIMM_DIRECT_KEY_MAP[key]] = value
            continue
        match = _TIMM_BLOCK_KEY_RE.match(key)
        if match is not None:
            index, suffix = match.group(1), match.group(2)
            mapped = _TIMM_BLOCK_SUFFIX_MAP.get(suffix)
            if mapped is not None:
                out[f"blocks.{index}.{mapped}"] = value
                continue
        raise KeyError(
            f"timm state-dict key {key!r} has no leaf-form mapping; refusing "
            "to silently drop weights. Expected a deit/vit "
            "vision_transformer layout (fused attn.qkv, patch_embed.proj, "
            "mlp.fc1/fc2)."
        )
    return out


def load_timm_vit_state_dict(
    model: LeafVisionTransformer,
    timm_state: Mapping[str, torch.Tensor],
) -> LeafVisionTransformer:
    """Load a timm-layout state dict into a leaf-form model (strict: every
    parameter must be covered, every tensor must land)."""
    model.load_state_dict(timm_vit_to_leaf_state_dict(timm_state), strict=True)
    return model
