"""Shared pruning compaction helpers for IR and SoftCore paths."""

from __future__ import annotations

import numpy as np


def compact_hardware_bias_columns(
    hardware_bias: np.ndarray | None,
    keep_cols: np.ndarray,
) -> np.ndarray | None:
    """Slice ``hardware_bias`` along neuron columns when pruning compacts a core."""
    if hardware_bias is None:
        return None
    return hardware_bias[keep_cols]
