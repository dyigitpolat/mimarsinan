"""Canonical TTFS closed-form math (numpy); torch wrappers delegate here."""

from __future__ import annotations

import numpy as np
import torch


def ttfs_continuous_activation(
    weighted_sum: np.ndarray,
    threshold: float,
    *,
    activation_scale: float = 1.0,
) -> np.ndarray:
    """relu(V)/theta capped to [0, 1] in activation-scale units."""
    v = np.asarray(weighted_sum, dtype=np.float64)
    thr = float(threshold) * float(activation_scale)
    if thr <= 0:
        return np.zeros_like(v, dtype=np.float64)
    return np.clip(v / thr, 0.0, 1.0)


def ttfs_continuous_activation_torch(
    weighted_sum: torch.Tensor,
    threshold: torch.Tensor,
    *,
    activation_scale: float = 1.0,
) -> torch.Tensor:
    v = weighted_sum.detach().cpu().numpy()
    thr = float(threshold.item()) if hasattr(threshold, "item") else float(threshold)
    out = ttfs_continuous_activation(v, thr, activation_scale=activation_scale)
    return torch.tensor(out, dtype=weighted_sum.dtype, device=weighted_sum.device)


def ttfs_quantized_fire_mask(
    weighted_sum: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Binary fire mask: weighted_sum >= threshold."""
    return (np.asarray(weighted_sum, dtype=np.float64) >= float(threshold)).astype(
        np.float64
    )
