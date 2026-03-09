"""Centralized propagative pruning for a single weight matrix.

Given a matrix and an initial set of pruned rows/columns (or derived from
zero-threshold), runs the fixpoint: a row that only feeds pruned columns is
pruned; a column that only receives from pruned rows is pruned. Exempt indices
are never added to the pruned set (at init or in the fixpoint). Used by
ir_pruning for both owned-weight cores and weight banks so the logic lives
in one place.
"""

from __future__ import annotations

from typing import AbstractSet, Set, Tuple

import numpy as np


def compute_propagated_pruned_rows_cols(
    matrix: np.ndarray,
    zero_threshold: float = 1e-8,
    conn_eps: float | None = None,
    initial_zero_rows: Set[int] | None = None,
    initial_zero_cols: Set[int] | None = None,
    exempt_rows: AbstractSet[int] = frozenset(),
    exempt_cols: AbstractSet[int] = frozenset(),
) -> Tuple[Set[int], Set[int]]:
    """Compute pruned row and column indices with propagative fixpoint.

    Rows (axons) and columns (neurons) are pruned when:
    - Initially: row/column has negligible weight (sum of abs below zero_threshold),
      or they are in initial_zero_rows / initial_zero_cols when provided.
      Exempt indices are never added at init.
    - Propagative: a row that only feeds pruned columns is pruned; a column
      that only receives from pruned rows is pruned. Exempt indices are never
      added during propagation. Iterate until fixpoint.

    Args:
        matrix: Weight matrix (axons x neurons), e.g. core_matrix.
        zero_threshold: Below this sum-of-abs, a row/column is considered zero.
        conn_eps: Epsilon for "has connection"; if None, uses min(1e-12, zero_threshold*1e-4).
        initial_zero_rows: Optional initial set of row indices to treat as pruned.
        initial_zero_cols: Optional initial set of column indices to treat as pruned.
        exempt_rows: Row indices that must never be added to the pruned set.
        exempt_cols: Column indices that must never be added to the pruned set.

    Returns:
        (pruned_rows_set, pruned_cols_set) both as sets of indices.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    n_axons, n_neurons = mat.shape
    if conn_eps is None:
        conn_eps = min(1e-12, zero_threshold * 1e-4)

    exempt_rows = frozenset(exempt_rows) if exempt_rows is not None else frozenset()
    exempt_cols = frozenset(exempt_cols) if exempt_cols is not None else frozenset()

    if initial_zero_rows is not None and initial_zero_cols is not None:
        zero_rows = set(initial_zero_rows) - exempt_rows
        zero_cols = set(initial_zero_cols) - exempt_cols
    else:
        zero_rows = {
            i for i in range(n_axons)
            if i not in exempt_rows and np.abs(mat[i, :]).sum() < zero_threshold
        }
        zero_cols = {
            j for j in range(n_neurons)
            if j not in exempt_cols and np.abs(mat[:, j]).sum() < zero_threshold
        }
        if initial_zero_rows is not None:
            zero_rows |= set(initial_zero_rows) - exempt_rows
        if initial_zero_cols is not None:
            zero_cols |= set(initial_zero_cols) - exempt_cols

    abs_conn = np.abs(mat) >= conn_eps
    has_any_conn_row = abs_conn.any(axis=1)
    has_any_conn_col = abs_conn.any(axis=0)

    changed = True
    while changed:
        changed = False
        for i in range(n_axons):
            if i in zero_rows or i in exempt_rows or not has_any_conn_row[i]:
                continue
            if all(j in zero_cols for j in range(n_neurons) if abs_conn[i, j]):
                zero_rows.add(i)
                changed = True
        for j in range(n_neurons):
            if j in zero_cols or j in exempt_cols or not has_any_conn_col[j]:
                continue
            if all(i in zero_rows for i in range(n_axons) if abs_conn[i, j]):
                zero_cols.add(j)
                changed = True

    return zero_rows, zero_cols
