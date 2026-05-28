"""Torchvision ViT (small) conversion + forward smoke test on CPU."""

from __future__ import annotations

import pytest
import torch

torchvision = pytest.importorskip("torchvision")
from torchvision.models import VisionTransformer  # noqa: E402

from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


def _build_small_vit():
    return VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=10,
    )


class TestSmallVitConversion:
    def test_convert_and_forward_matches_native(self):
        """Small ViT must convert and produce native-equivalent output."""
        torch.manual_seed(0)
        model = _build_small_vit().eval()
        input_shape = (3, 32, 32)
        flow = convert_torch_model(
            model, input_shape, num_classes=10, device="cpu",
        )
        x = torch.randn(2, *input_shape)
        with torch.no_grad():
            native = model(x)
            converted = flow(x)
        assert converted.shape == (2, 10)
        assert torch.allclose(converted, native, atol=1e-4), (
            f"converted ViT diverged from native; max diff "
            f"{(converted - native).abs().max().item():.2e}"
        )
