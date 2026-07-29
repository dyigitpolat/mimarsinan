"""Centralized propagative fixpoint pruning for a single weight matrix, shared by ir_pruning for both owned-weight cores and weight banks."""

from __future__ import annotations

from typing import AbstractSet, Set, Tuple

import numpy as np

from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    require_elimination_propagation,
)


def compute_propagated_pruned_rows_cols(
    matrix: np.ndarray,
    zero_threshold: float = 1e-8,
    conn_eps: float | None = None,
    initial_zero_rows: Set[int] | None = None,
    initial_zero_cols: Set[int] | None = None,
    exempt_rows: AbstractSet[int] = frozenset(),
    exempt_cols: AbstractSet[int] = frozenset(),
    cols_with_implicit_source: AbstractSet[int] = frozenset(),
    mode: str = ELIMINATION_PROPAGATION_CASCADE,
) -> Tuple[Set[int], Set[int]]:
    """Compute pruned row and column indices under a propagation mode.

    Rows (axons) and columns (neurons) are pruned when:
    - Initially: row/column has negligible weight (sum of abs below zero_threshold),
      or they are in initial_zero_rows / initial_zero_cols when provided.
      Exempt indices are never added at init.
    - Propagative (cascade only): a row that only feeds pruned columns is
      pruned; a column that only receives from pruned rows is pruned. Exempt
      indices are never added during propagation. Iterate until fixpoint.

    Args:
        matrix: Weight matrix (axons x neurons), e.g. core_matrix.
        zero_threshold: Below this sum-of-abs, a row/column is considered zero.
        conn_eps: Epsilon for "has connection"; if None, uses min(1e-12, zero_threshold*1e-4).
        initial_zero_rows: Optional initial set of row indices to treat as pruned.
        initial_zero_cols: Optional initial set of column indices to treat as pruned.
        exempt_rows: Row indices that must never be added to the pruned set.
        exempt_cols: Column indices that must never be added to the pruned set.
        cols_with_implicit_source: Column indices that have an out-of-matrix
            live source (e.g. ``hardware_bias`` carries a non-zero per-neuron
            offset that produces spikes regardless of axon connectivity). Such
            columns are never killed by within-matrix propagation: even when
            every row feeding them dies, the implicit source keeps them alive.
        mode: Propagation arm (see ``propagation_mode``):
            - ``"masked"``: skip the fixpoint and return exactly the seeded
              (exemption-filtered) sets — the allocation-naive LOWER BOUND,
              with no structural reasoning at all. This is deliberately NOT
              "the single-layer structured-pruning baseline": that baseline
              (``"closure"``) still couples a killed neuron with its consumer
              rows.
            - ``"closure"``: identical to ``"masked"`` at single-matrix
              granularity — one-hop seed-group coupling is a cross-matrix
              phenomenon handled by the global driver; within one matrix
              closure adds nothing and never discovers emergent deadness.
            - ``"cascade"`` (default): the full within-matrix fixpoint.

    Returns:
        (pruned_rows_set, pruned_cols_set) both as sets of indices.
    """
    mode = require_elimination_propagation(mode)
    mat = np.asarray(matrix)
    if mat.dtype != np.float32 and mat.dtype != np.float64:
        mat_f = mat.astype(np.float32, copy=False)
    else:
        mat_f = mat
    n_axons, n_neurons = mat_f.shape
    if conn_eps is None:
        conn_eps = min(1e-12, zero_threshold * 1e-4)

    exempt_row_mask = _set_to_mask(exempt_rows, n_axons)
    exempt_col_mask = _set_to_mask(exempt_cols, n_neurons)

    # An out-of-matrix live source (e.g. non-zero hardware_bias) marks a column
    # as never killed by within-matrix propagation, though it can still be
    # pruned via the initial seed.
    implicit_col_alive_mask = _set_to_mask(cols_with_implicit_source, n_neurons)

    # When explicit seeds are provided for both axes, treat them as the full
    # initial pruned set (the caller is expected to have already incorporated
    # value-based deadness exactly once, before the fixpoint).
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

    zero_row_mask &= ~exempt_row_mask
    zero_col_mask &= ~exempt_col_mask

    if mode != ELIMINATION_PROPAGATION_CASCADE:
        return _mask_to_set(zero_row_mask), _mask_to_set(zero_col_mask)

    abs_conn = np.abs(mat_f) >= conn_eps
    has_any_conn_row = np.asarray(abs_conn.any(axis=1))
    has_any_conn_col = np.asarray(abs_conn.any(axis=0))

    # Fixpoint: a row dies once all its connections target dead cols; a col
    # dies once all its connections come from dead rows. Exempt indices never die.
    while True:
        row_dies, col_dies = _one_step_deaths(
            abs_conn=abs_conn,
            has_any_conn_row=has_any_conn_row,
            has_any_conn_col=has_any_conn_col,
            zero_row_mask=zero_row_mask,
            zero_col_mask=zero_col_mask,
            exempt_row_mask=exempt_row_mask,
            exempt_col_mask=exempt_col_mask,
            implicit_col_alive_mask=implicit_col_alive_mask,
        )
        if not (row_dies.any() or col_dies.any()):
            break

        zero_row_mask |= row_dies
        zero_col_mask |= col_dies

    return _mask_to_set(zero_row_mask), _mask_to_set(zero_col_mask)


def matrix_one_step_deaths(
    matrix: np.ndarray,
    *,
    pruned_rows: AbstractSet[int],
    pruned_cols: AbstractSet[int],
    exempt_rows: AbstractSet[int] = frozenset(),
    exempt_cols: AbstractSet[int] = frozenset(),
    cols_with_implicit_source: AbstractSet[int] = frozenset(),
    zero_threshold: float = 1e-8,
    conn_eps: float | None = None,
) -> Tuple[Set[int], Set[int]]:
    """ONE synchronous wave of the within-matrix starvation operator.

    Returns only the NEW deaths enabled by the given state: rows whose every
    connection targets a dead column, and columns whose every connection
    originates at a dead row (implicit-source columns never starve). Iterating
    this operator to quiescence is exactly the cascade fixpoint — the ledger's
    depth replay leans on that identity.
    """
    mat = np.asarray(matrix)
    if mat.dtype != np.float32 and mat.dtype != np.float64:
        mat = mat.astype(np.float32, copy=False)
    n_axons, n_neurons = mat.shape
    if conn_eps is None:
        conn_eps = min(1e-12, zero_threshold * 1e-4)
    abs_conn = np.abs(mat) >= conn_eps
    row_dies, col_dies = _one_step_deaths(
        abs_conn=abs_conn,
        has_any_conn_row=np.asarray(abs_conn.any(axis=1)),
        has_any_conn_col=np.asarray(abs_conn.any(axis=0)),
        zero_row_mask=_set_to_mask(pruned_rows, n_axons),
        zero_col_mask=_set_to_mask(pruned_cols, n_neurons),
        exempt_row_mask=_set_to_mask(exempt_rows, n_axons),
        exempt_col_mask=_set_to_mask(exempt_cols, n_neurons),
        implicit_col_alive_mask=_set_to_mask(
            cols_with_implicit_source, n_neurons
        ),
    )
    return _mask_to_set(row_dies), _mask_to_set(col_dies)


def _one_step_deaths(
    *,
    abs_conn: np.ndarray,
    has_any_conn_row: np.ndarray,
    has_any_conn_col: np.ndarray,
    zero_row_mask: np.ndarray,
    zero_col_mask: np.ndarray,
    exempt_row_mask: np.ndarray,
    exempt_col_mask: np.ndarray,
    implicit_col_alive_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """The cascade loop body: one wave of row/col starvation as boolean masks."""
    alive_cols = ~zero_col_mask
    row_has_alive_target = (abs_conn & alive_cols[None, :]).any(axis=1)
    row_dies = (
        ~row_has_alive_target
        & has_any_conn_row
        & ~zero_row_mask
        & ~exempt_row_mask
    )

    alive_rows = ~zero_row_mask
    col_has_alive_source = (abs_conn & alive_rows[:, None]).any(axis=0)
    col_has_alive_source = col_has_alive_source | implicit_col_alive_mask
    col_dies = (
        ~col_has_alive_source
        & has_any_conn_col
        & ~zero_col_mask
        & ~exempt_col_mask
    )
    return row_dies, col_dies


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
