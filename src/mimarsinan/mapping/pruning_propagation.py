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
    # Work in float32 for the magnitude comparisons. Int8 weight matrices (post
    # quantization) are cheap to upcast once; float32 is plenty of headroom for
    # a sum-of-abs comparison against a small threshold and uses half the memory
    # of float64.
    mat = np.asarray(matrix)
    if mat.dtype != np.float32 and mat.dtype != np.float64:
        mat_f = mat.astype(np.float32, copy=False)
    else:
        mat_f = mat
    n_axons, n_neurons = mat_f.shape
    if conn_eps is None:
        conn_eps = min(1e-12, zero_threshold * 1e-4)

    # Boolean exempt masks
    exempt_row_mask = np.zeros(n_axons, dtype=bool)
    if exempt_rows:
        idx = np.fromiter(
            (i for i in exempt_rows if 0 <= i < n_axons),
            dtype=np.int64,
        )
        exempt_row_mask[idx] = True
    exempt_col_mask = np.zeros(n_neurons, dtype=bool)
    if exempt_cols:
        idx = np.fromiter(
            (j for j in exempt_cols if 0 <= j < n_neurons),
            dtype=np.int64,
        )
        exempt_col_mask[idx] = True

    # Initial zero masks: either from provided sets or from value-based threshold.
    if initial_zero_rows is not None and initial_zero_cols is not None:
        zero_row_mask = _set_to_mask(initial_zero_rows, n_axons)
        zero_col_mask = _set_to_mask(initial_zero_cols, n_neurons)
    else:
        abs_mat = np.abs(mat_f)
        row_sum = abs_mat.sum(axis=1)
        col_sum = abs_mat.sum(axis=0)
        zero_row_mask = row_sum < zero_threshold
        zero_col_mask = col_sum < zero_threshold
        if initial_zero_rows is not None:
            zero_row_mask |= _set_to_mask(initial_zero_rows, n_axons)
        if initial_zero_cols is not None:
            zero_col_mask |= _set_to_mask(initial_zero_cols, n_neurons)

    # Exempt indices are never pruned.
    zero_row_mask &= ~exempt_row_mask
    zero_col_mask &= ~exempt_col_mask

    # Connectivity mask: which (i,j) entries are above the "has connection" threshold.
    # Using float32 abs is fine; result is a bool matrix of shape (n_axons, n_neurons).
    abs_conn = np.abs(mat_f) >= conn_eps
    has_any_conn_row = abs_conn.any(axis=1)
    has_any_conn_col = abs_conn.any(axis=0)

    # Fixpoint: a row dies if all its connections are to already-dead cols;
    # a col dies if all its connections are from already-dead rows. Exempt
    # indices never die. Rows/cols with no connection at all are handled by
    # the initial phase only.
    while True:
        alive_cols = ~zero_col_mask
        # For each row, does it still have any connection to an alive col?
        row_has_alive_target = (abs_conn & alive_cols[None, :]).any(axis=1)
        row_dies = (
            ~row_has_alive_target
            & has_any_conn_row
            & ~zero_row_mask
            & ~exempt_row_mask
        )

        alive_rows = ~zero_row_mask
        col_has_alive_source = (abs_conn & alive_rows[:, None]).any(axis=0)
        col_dies = (
            ~col_has_alive_source
            & has_any_conn_col
            & ~zero_col_mask
            & ~exempt_col_mask
        )

        if not (row_dies.any() or col_dies.any()):
            break

        zero_row_mask |= row_dies
        zero_col_mask |= col_dies

    return _mask_to_set(zero_row_mask), _mask_to_set(zero_col_mask)


def _set_to_mask(indices, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if not indices:
        return mask
    idx = np.fromiter(
        (int(i) for i in indices if 0 <= int(i) < length),
        dtype=np.int64,
    )
    if idx.size:
        mask[idx] = True
    return mask


def _mask_to_set(mask: np.ndarray) -> Set[int]:
    return set(int(i) for i in np.flatnonzero(mask))
