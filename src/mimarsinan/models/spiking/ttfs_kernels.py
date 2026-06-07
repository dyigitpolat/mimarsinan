"""Closed-form TTFS activation kernels (thin aliases over the wire kernel pair)."""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.models.spiking.wire_semantics import (
    ttfs_quantized_staircase,
    ttfs_quantized_staircase_np,
)


def ttfs_quantized_activation_np(
    v: np.ndarray,
    threshold: np.ndarray,
    simulation_length: int,
) -> np.ndarray:
    """Numpy ``ttfs_quantized_activation`` for chip_simulation reference paths."""
    return ttfs_quantized_staircase_np(v, threshold, simulation_length)


def ttfs_quantized_activation(
    V: torch.Tensor,
    threshold: torch.Tensor,
    simulation_length: int,
) -> torch.Tensor:
    """``(S - clamp(ceil(S*(1-V/θ)), 0, S-1)) / S`` with fire mask."""
    return ttfs_quantized_staircase(V, threshold, simulation_length)
