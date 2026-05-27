"""Closed-form TTFS activation kernels (shared by unified and hybrid flows)."""

from __future__ import annotations

import numpy as np
import torch


def ttfs_quantized_activation_np(
    v: np.ndarray,
    threshold: np.ndarray,
    simulation_length: int,
) -> np.ndarray:
    """Numpy ``ttfs_quantized_activation`` for chip_simulation reference paths."""
    s = int(simulation_length)
    safe = np.maximum(np.asarray(threshold, dtype=np.float64), 1e-12)
    k_fire_raw = np.ceil(s * (1.0 - v / safe))
    fires = k_fire_raw < s
    k_fire = np.clip(k_fire_raw, 0, s - 1)
    out = np.where(fires, (s - k_fire) / s, 0.0)
    return out.astype(np.float64, copy=False)


def ttfs_quantized_activation(
    V: torch.Tensor,
    threshold: torch.Tensor,
    simulation_length: int,
) -> torch.Tensor:
    """``(S - clamp(ceil(S*(1-V/θ)), 0, S-1)) / S`` with fire mask."""
    S = simulation_length
    safe_thresh = threshold.clamp(min=1e-12)
    k_fire_raw = torch.ceil(S * (1.0 - V / safe_thresh))
    fires = k_fire_raw < S
    k_fire = k_fire_raw.clamp(0, S - 1)
    return torch.where(fires, (S - k_fire) / S, torch.zeros_like(k_fire))
