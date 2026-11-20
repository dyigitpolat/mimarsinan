"""Shared pruning compaction helpers for IR and SoftCore paths."""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: The per-column ``Σ_i |W_ij|`` of the rows compaction dropped from a core's owned matrix (absent: none dropped). The
#: model still reads those rows; beside an offset grid they hold its code of value 0 (``entry_offsets`` reads it, N7).
PRUNED_ROW_MAGNITUDE = "pruned_row_magnitude"


def record_dropped_rows(node, matrix: np.ndarray, rows: Sequence[int]) -> None:
    """Add the dropped ``rows``' per-column ``Σ|W|`` of ``matrix`` (columns already compacted) to the node's record;
    an earlier record already follows the kept columns (a value held per neuron, carried by the column compaction's
    ``take_neuron_columns``). Nothing is recorded where nothing was ever dropped."""
    prior = getattr(node, PRUNED_ROW_MAGNITUDE, None)
    if not len(rows) and prior is None:
        return
    magnitude = np.abs(np.asarray(matrix, dtype=np.float64)[np.asarray(rows, dtype=np.int64), :]).sum(axis=0)
    if prior is not None:
        magnitude = magnitude + np.asarray(prior, dtype=np.float64)
    setattr(node, PRUNED_ROW_MAGNITUDE, magnitude)

