"""Tensor statistics that survive deployment-scale inputs."""

from __future__ import annotations

import torch

# torch.quantile rejects inputs above 2^24 elements ("input tensor is too
# large"). A ViT patch-embed input at batch 512 is 77M elements, so any
# calibration that quantiles a live activation must guard the cap.
QUANTILE_ELEMENT_LIMIT = 1 << 24


def subsample_to_limit(
    tensor: torch.Tensor, limit: int = QUANTILE_ELEMENT_LIMIT
) -> torch.Tensor:
    """Flatten and stride down to at most ``limit`` elements (deterministic)."""
    flat = tensor.detach().reshape(-1)
    if flat.numel() <= limit:
        return flat
    return flat[:: max(1, flat.numel() // limit)]


def safe_quantile(
    tensor: torch.Tensor, q: float, limit: int = QUANTILE_ELEMENT_LIMIT
) -> torch.Tensor:
    """``torch.quantile`` that tolerates oversized inputs by subsampling.

    Exact below the cap; above it the value is a strided-sample estimate
    (calibration-grade — the caller wants a distribution shoulder, not an
    order statistic).
    """
    flat = subsample_to_limit(tensor, limit)
    if not flat.is_floating_point():
        flat = flat.float()
    return torch.quantile(flat, q)
