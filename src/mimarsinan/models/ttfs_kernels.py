"""Closed-form TTFS activation kernels (shared by unified and hybrid flows)."""

from __future__ import annotations

import torch


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
