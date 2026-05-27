"""Shared spiking-simulator configuration constants and validation."""

from __future__ import annotations

import torch

COMPUTE_DTYPE = torch.float64

FIRING_MODES = frozenset({"Default", "Novena", "TTFS"})
SPIKE_MODES = frozenset({"Stochastic", "Deterministic", "FrontLoaded", "Uniform", "TTFS"})
THRESHOLDING_MODES = frozenset({"<", "<="})
TTFS_SPIKING_MODES = frozenset({"ttfs", "ttfs_quantized"})
TTFS_FIRING_MODES = frozenset({"TTFS"})


def validate_spiking_init(
    *,
    firing_mode: str,
    spike_mode: str,
    thresholding_mode: str,
) -> None:
    if firing_mode not in FIRING_MODES:
        raise ValueError(f"Invalid firing_mode: {firing_mode!r}")
    if spike_mode not in SPIKE_MODES:
        raise ValueError(f"Invalid spike_mode: {spike_mode!r}")
    if thresholding_mode not in THRESHOLDING_MODES:
        raise ValueError(f"Invalid thresholding_mode: {thresholding_mode!r}")
