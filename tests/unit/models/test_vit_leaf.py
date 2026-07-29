"""MHA-leaf vision transformer: geometry, timm weight-porting shim, registration.

The timm tests construct SYNTHETIC state dicts with timm names/shapes (via a
hand-wired timm-style reference module) — no downloads. Placement is asserted
tensor-by-tensor AND by round-trip forward comparison against the reference.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.vit_leaf import (
    LeafVisionTransformer,
    deit_tiny_leaf,
    load_timm_vit_state_dict,
    timm_vit_to_leaf_state_dict,
    tiny_test_vit,
)


# ── Geometry ─────────────────────────────────────────────────────────────────

class TestLeafVisionTransformerGeometry:
    def test_tiny_config_forward_shape(self):
        torch.manual_seed(0)
        model = tiny_test_vit(num_classes=10).eval()
        with torch.no_grad():
            out = model(torch.randn(4, 3, 8, 8))
        assert out.shape == (4, 10)

    def test_deit_tiny_config_geometry(self):
        model = deit_tiny_leaf()
        assert model.pos_embed.shape == (1, 197, 192)  # (224/16)^2 + cls = 197
        assert model.cls_token.shape == (1, 1, 192)
        assert len(model.blocks) == 12
        assert model.patch_embed.kernel_size == (16, 16)
        assert model.patch_embed.stride == (16, 16)
        first = model.blocks[0]
        assert first.attn.num_heads == 3
        assert first.fc1.out_features == 4 * 192
        assert model.head.out_features == 1000

    def test_attention_is_a_batch_first_mha_leaf(self):
        model = deit_tiny_leaf()
        for block in model.blocks:
            assert isinstance(block.attn, nn.MultiheadAttention)
            assert block.attn.batch_first

    def test_rejects_indivisible_geometry(self):
        with pytest.raises(ValueError, match="divisible"):
            LeafVisionTransformer(image_size=10, patch_size=3)
        with pytest.raises(ValueError, match="divisible"):
            LeafVisionTransformer(embed_dim=10, num_heads=3)


# ── Hand-wired timm-style reference (fused qkv, raw-op attention) ────────────

class _TimmStyleAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _TimmStyleMlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _TimmStyleBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _TimmStyleAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _TimmStyleMlp(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _TimmStylePatchEmbed(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _TimmStyleViT(nn.Module):
    """Attribute names mirror timm's vision_transformer, so ``state_dict()``
    IS a synthetic timm-layout state dict."""

    def __init__(self, *, image_size, patch_size, in_channels, dim, num_heads,
                 depth, mlp_ratio, num_classes):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = _TimmStylePatchEmbed(in_channels, dim, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.blocks = nn.ModuleList(
            _TimmStyleBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def _small_pair(seed=3):
    """Random-weight timm-style reference + matching leaf model geometry."""
    torch.manual_seed(seed)
    reference = _TimmStyleViT(
        image_size=8, patch_size=4, in_channels=3, dim=16, num_heads=2,
        depth=2, mlp_ratio=2.0, num_classes=10,
    ).eval()
    leaf = LeafVisionTransformer(
        image_size=8, patch_size=4, in_channels=3, embed_dim=16, num_heads=2,
        depth=2, mlp_ratio=2.0, num_classes=10,
    ).eval()
    return reference, leaf


# ── Weight porting ───────────────────────────────────────────────────────────

class TestTimmWeightPorting:
    def test_exact_tensor_placement(self):
        reference, leaf = _small_pair()
        timm_state = reference.state_dict()
        load_timm_vit_state_dict(leaf, timm_state)

        assert torch.equal(leaf.cls_token, reference.cls_token)
        assert torch.equal(leaf.pos_embed, reference.pos_embed)
        assert torch.equal(leaf.patch_embed.weight, reference.patch_embed.proj.weight)
        assert torch.equal(leaf.patch_embed.bias, reference.patch_embed.proj.bias)
        for leaf_block, ref_block in zip(leaf.blocks, reference.blocks):
            # fused qkv rows -> in_proj rows: q = [0:d], k = [d:2d], v = [2d:3d]
            d = ref_block.norm1.weight.shape[0]
            assert torch.equal(leaf_block.attn.in_proj_weight, ref_block.attn.qkv.weight)
            assert torch.equal(
                leaf_block.attn.in_proj_weight[0:d], ref_block.attn.qkv.weight[0:d]
            )
            assert torch.equal(
                leaf_block.attn.in_proj_weight[2 * d:3 * d],
                ref_block.attn.qkv.weight[2 * d:3 * d],
            )
            assert torch.equal(leaf_block.attn.in_proj_bias, ref_block.attn.qkv.bias)
            assert torch.equal(
                leaf_block.attn.out_proj.weight, ref_block.attn.proj.weight
            )
            assert torch.equal(leaf_block.attn.out_proj.bias, ref_block.attn.proj.bias)
            assert torch.equal(leaf_block.norm1.weight, ref_block.norm1.weight)
            assert torch.equal(leaf_block.norm2.bias, ref_block.norm2.bias)
            assert torch.equal(leaf_block.fc1.weight, ref_block.mlp.fc1.weight)
            assert torch.equal(leaf_block.fc2.bias, ref_block.mlp.fc2.bias)
        assert torch.equal(leaf.norm.weight, reference.norm.weight)
        assert torch.equal(leaf.head.weight, reference.head.weight)
        assert torch.equal(leaf.head.bias, reference.head.bias)

    def test_round_trip_forward_parity_vs_hand_wired(self):
        reference, leaf = _small_pair(seed=11)
        load_timm_vit_state_dict(leaf, reference.state_dict())
        reference = reference.double()
        leaf = leaf.double()
        torch.manual_seed(0)
        x = torch.randn(4, 3, 8, 8, dtype=torch.float64)
        with torch.no_grad():
            want = reference(x)
            got = leaf(x)
        torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_unknown_key_fails_loud(self):
        reference, _ = _small_pair()
        timm_state = dict(reference.state_dict())
        timm_state["head_dist.weight"] = torch.randn(10, 16)
        with pytest.raises(KeyError, match="head_dist.weight"):
            timm_vit_to_leaf_state_dict(timm_state)

    def test_deit_tiny_shapes_load_strict(self):
        # Full DeiT-Tiny-shaped synthetic state dict lands strict=True (every
        # leaf parameter covered, every timm tensor placed).
        torch.manual_seed(5)
        reference = _TimmStyleViT(
            image_size=224, patch_size=16, in_channels=3, dim=192, num_heads=3,
            depth=12, mlp_ratio=4.0, num_classes=1000,
        )
        timm_state = reference.state_dict()
        assert timm_state["blocks.0.attn.qkv.weight"].shape == (3 * 192, 192)
        leaf = load_timm_vit_state_dict(deit_tiny_leaf(), timm_state)
        assert torch.equal(leaf.pos_embed, reference.pos_embed)
        assert torch.equal(
            leaf.blocks[11].attn.in_proj_weight,
            reference.blocks[11].attn.qkv.weight,
        )


# ── Registration ─────────────────────────────────────────────────────────────

class TestTorchViTLeafBuilderRegistration:
    def test_registered_in_torch_category(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY
        from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry

        assert "torch_vit_leaf" in BUILDERS_REGISTRY
        assert ModelRegistry.get_category("torch_vit_leaf") == "torch"

    def test_default_build_is_deit_tiny_geometry(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        builder = BUILDERS_REGISTRY["torch_vit_leaf"](
            device="cpu", input_shape=(3, 224, 224), num_classes=1000,
            pipeline_config={},
        )
        model = builder.build({})
        assert isinstance(model, LeafVisionTransformer)
        assert model.pos_embed.shape == (1, 197, 192)
        assert len(model.blocks) == 12
        assert model.head.out_features == 1000

    def test_small_config_builds_and_runs(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        builder = BUILDERS_REGISTRY["torch_vit_leaf"](
            device="cpu", input_shape=(3, 8, 8), num_classes=10,
            pipeline_config={},
        )
        model = builder.build(
            {"patch_size": 4, "embed_dim": 16, "num_heads": 2, "depth": 1,
             "mlp_ratio": 2.0}
        ).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 8, 8))
        assert out.shape == (2, 10)

    def test_validate_config(self):
        from mimarsinan.models.builders import BUILDERS_REGISTRY

        builder_cls = BUILDERS_REGISTRY["torch_vit_leaf"]
        good = {"patch_size": 4, "embed_dim": 16, "num_heads": 2, "depth": 1}
        assert builder_cls.validate_config(good, {}, (3, 8, 8))
        assert not builder_cls.validate_config(
            {**good, "patch_size": 3}, {}, (3, 8, 8)
        )  # 8 % 3 != 0
        assert not builder_cls.validate_config(
            {**good, "num_heads": 3}, {}, (3, 8, 8)
        )  # 16 % 3 != 0
        assert not builder_cls.validate_config(good, {}, (3, 8, 12))  # non-square
