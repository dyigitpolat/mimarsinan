"""Shared pruning compaction helpers for IR and SoftCore paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: The per-column ``Σ_i |W_ij|`` of the rows compaction dropped from a core's owned matrix (absent: none dropped). The
#: model still reads those rows; beside an offset grid they hold its code of value 0 (``entry_offsets`` reads it, N7).
PRUNED_ROW_MAGNITUDE = "pruned_row_magnitude"


def record_dropped_rows(node, matrix: np.ndarray, rows: Sequence[int], keep_cols=None) -> None:
    """Add the dropped ``rows``' per-column ``Σ|W|`` of ``matrix`` (columns already compacted) to the node's record;
    an earlier record is compacted to ``keep_cols`` first. Nothing is recorded where nothing was ever dropped."""
    prior = getattr(node, PRUNED_ROW_MAGNITUDE, None)
    if not len(rows) and prior is None:
        return
    magnitude = np.abs(np.asarray(matrix, dtype=np.float64)[np.asarray(rows, dtype=np.int64), :]).sum(axis=0)
    if prior is not None:
        prior = np.asarray(prior, dtype=np.float64)
        magnitude = magnitude + (prior if keep_cols is None else prior[np.asarray(keep_cols, dtype=np.int64)])
    setattr(node, PRUNED_ROW_MAGNITUDE, magnitude)


def compact_hardware_bias_columns(
    hardware_bias: np.ndarray | None,
    keep_cols: "np.ndarray | Sequence[int]",
) -> np.ndarray | None:
    """Slice ``hardware_bias`` along neuron columns when pruning compacts a core."""
    if hardware_bias is None:
        return None
    return hardware_bias[keep_cols]
