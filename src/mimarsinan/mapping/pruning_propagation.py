"""Centralized propagative pruning for a single weight matrix.

Given a matrix and an initial set of pruned rows/columns (or derived from
zero-threshold), runs the fixpoint: a row that only feeds pruned columns is
pruned; a column that only receives from pruned rows is pruned. Used by
ir_pruning for both owned-weight cores and weight banks so the logic lives
in one place.
"""

from __future__ import annotations

from typing import Set, Tuple

import numpy as np


def compute_propagated_pruned_rows_cols(
    matrix: np.ndarray,
    zero_threshold: float = 1e-8,
    conn_eps: float | None = None,
    initial_zero_rows: Set[int] | None = None,
    initial_zero_cols: Set[int] | None = None,
) -> Tuple[Set[int], Set[int]]:
    """Compute pruned row and column indices with propagative fixpoint.

    Rows (axons) and columns (neurons) are pruned when:
    - Initially: row/column has negligible weight (sum of abs below zero_threshold),
      or they are in initial_zero_rows / initial_zero_cols when provided.
    - Propagative: a row that only feeds pruned columns is pruned; a column
      that only receives from pruned rows is pruned. Iterate until fixpoint.

    Args:
        matrix: Weight matrix (axons x neurons), e.g. core_matrix.
        zero_threshold: Below this sum-of-abs, a row/column is considered zero.
        conn_eps: Epsilon for "has connection"; if None, uses min(1e-12, zero_threshold*1e-4).
        initial_zero_rows: Optional initial set of row indices to treat as pruned.
        initial_zero_cols: Optional initial set of column indices to treat as pruned.

    Returns:
        (pruned_rows_set, pruned_cols_set) both as sets of indices.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    n_axons, n_neurons = mat.shape
    if conn_eps is None:
        conn_eps = min(1e-12, zero_threshold * 1e-4)

    if initial_zero_rows is not None and initial_zero_cols is not None:
        zero_rows = set(initial_zero_rows)
        zero_cols = set(initial_zero_cols)
    else:
        zero_rows = set(
            i for i in range(n_axons)
            if np.abs(mat[i, :]).sum() < zero_threshold
        )
        zero_cols = set(
            j for j in range(n_neurons)
            if np.abs(mat[:, j]).sum() < zero_threshold
        )
        if initial_zero_rows is not None:
            zero_rows |= set(initial_zero_rows)
        if initial_zero_cols is not None:
            zero_cols |= set(initial_zero_cols)

    changed = True
    while changed:
        changed = False
        for i in range(n_axons):
            if i in zero_rows:
                continue
            non_zero_cols = set(
                j for j in range(n_neurons)
                if np.abs(mat[i, j]) >= conn_eps
            )
            if non_zero_cols and non_zero_cols <= zero_cols:
                zero_rows.add(i)
                changed = True
        for j in range(n_neurons):
            if j in zero_cols:
                continue
            non_zero_rows = set(
                i for i in range(n_axons)
                if np.abs(mat[i, j]) >= conn_eps
            )
            if non_zero_rows and non_zero_rows <= zero_rows:
                zero_cols.add(j)
                changed = True

    return zero_rows, zero_cols
