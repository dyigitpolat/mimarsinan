"""The cross-core wave operators as array reductions.

Each kernel is the one-for-one twin of a reference kernel and computes the
IDENTICAL predicate over identical inputs -- pure set membership, no floating
point anywhere, so bit-identity is by construction and the differential tests
in ``test_flat_kernels.py`` hold both sides equal on every topology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from mimarsinan.mapping.pruning.graph.flat.state import FlatState
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cols_with_nonzero_bias,
)

__all__ = [
    "BankBatch",
    "build_bank_batches",
    "flat_cross_core_dead_axons",
    "flat_orphan_neurons",
    "flat_within_matrix_fixpoint",
]


def flat_cross_core_dead_axons(state: FlatState, col_dead: np.ndarray) -> np.ndarray:
    """Twin of ``_cross_core_dead_axons`` over the whole graph at once.

    OFF axons are always dead; a direct axon dies with its producer port; a
    relayed axon dies when EVERY transfer-mapped producer port is dead.
    Returns a boolean over flat axon rows.
    """
    dead = np.zeros(state.n_rows, dtype=bool)
    if state.always_dead_rows.size:
        dead[state.always_dead_rows] = True
    if state.direct_axon.size:
        dead[state.direct_axon] = col_dead[state.direct_producer]
    if state.group_axon.size:
        live = ~col_dead[state.group_producers]
        live_per_group = np.add.reduceat(
            live.astype(np.int64), state.group_offsets[:-1]
        )
        dead[state.group_axon] = live_per_group == 0
    return dead


def flat_orphan_neurons(state: FlatState, row_dead: np.ndarray) -> np.ndarray:
    """Twin of ``_orphans_from_plan`` over the whole graph at once.

    A port with no consumer is orphaned outright; otherwise it dies when every
    consuming axon row is dead. Output and transfer-protected ports never
    orphan. Returns a boolean over flat neuron ports.
    """
    dead = np.zeros(state.n_cols, dtype=bool)
    if state.orphan_now.size:
        dead[state.orphan_now] = True
    if state.consumer_port.size:
        live = ~row_dead[state.consumer_rows]
        live_per_port = np.add.reduceat(
            live.astype(np.int64), state.consumer_offsets[:-1]
        )
        dead[state.consumer_port] = live_per_port == 0
    dead[state.never_orphan] = False
    return dead


@dataclass
class BankBatch:
    """All instances sharing one (bank, column-slice) view, batched.

    Bank-backed cores cannot fold (their carriers are read-only), so the view
    matrix -- and with it ``conn``, ``has_conn_*`` and the per-instance
    implicit-bias columns -- is static for the whole analysis.
    """

    core_positions: List[int]        # dense indices into FlatState ordering
    conn_f: np.ndarray               # (A, N) float32 0/1 connection pattern
    has_conn_row: np.ndarray         # (A,)  row has any connection
    has_conn_col: np.ndarray         # (N,)  col has any connection
    exempt_rows_m: np.ndarray        # (I, A)
    exempt_cols_m: np.ndarray        # (I, N)
    implicit_m: np.ndarray           # (I, N) bias-alive columns per instance
    implicit_sets: List[frozenset]   # same, as sets (differential tests)


def build_bank_batches(ctx, state) -> List[BankBatch]:
    """Group bank-backed cores by their exact shared view; owned cores stay
    on the per-core path (they can fold, and there are few of them)."""
    groups: dict = {}
    for k, node in enumerate(ctx.neural_cores):
        if node.core_matrix is not None:
            continue
        bid = getattr(node, "weight_bank_id", None)
        if bid is None or bid not in ctx.banks:
            continue
        if (getattr(ctx.constants, "deltas", None) or {}).get(node.id) is not None:
            continue                     # defensive: a folded view never batches
        groups.setdefault((bid, node.weight_row_slice), []).append(k)

    batches: List[BankBatch] = []
    for (_bid, _slc), positions in groups.items():
        rep = ctx.neural_cores[positions[0]]
        mat = ctx.base_node_matrix(rep)
        if mat is None:
            continue
        # the reference's exact cast-and-threshold, once per shared view
        mat_f = mat if mat.dtype in (np.float32, np.float64) else mat.astype(np.float32, copy=False)
        conn_eps = min(1e-12, ctx.zero_threshold * 1e-4)
        conn = np.abs(mat_f) >= conn_eps
        n_ax, n_ne = conn.shape
        ex_r = np.zeros((len(positions), n_ax), dtype=bool)
        ex_c = np.zeros((len(positions), n_ne), dtype=bool)
        imp = np.zeros((len(positions), n_ne), dtype=bool)
        imp_sets: List[frozenset] = []
        for pos, k in enumerate(positions):
            node = ctx.neural_cores[k]
            assert ctx.base_node_matrix(node).shape == conn.shape
            for i in ctx.exempt_rows.get(node.id, frozenset()):
                if 0 <= i < n_ax:
                    ex_r[pos, i] = True
            for j in ctx.exempt_cols.get(node.id, frozenset()):
                if 0 <= j < n_ne:
                    ex_c[pos, j] = True
            alive = _cols_with_nonzero_bias(
                ctx.node_bias(node), n_ne, ctx.zero_threshold
            )
            imp_sets.append(alive)
            for j in alive:
                imp[pos, j] = True
        batches.append(BankBatch(
            core_positions=list(positions),
            conn_f=conn.astype(np.float32),
            has_conn_row=np.asarray(conn.any(axis=1)),
            has_conn_col=np.asarray(conn.any(axis=0)),
            exempt_rows_m=ex_r, exempt_cols_m=ex_c,
            implicit_m=imp, implicit_sets=imp_sets,
        ))
    return batches


def flat_within_matrix_fixpoint(batch: BankBatch, row_dead, col_dead, state=None):
    """All instances' ``compute_propagated_pruned_rows_cols`` at once.

    The identical predicate in the identical inner order: both directions read
    the SAME pre-step masks, implicit-source columns never starve, exempt
    indices never die (and are filtered from the seeds, as the reference does
    at init). Counts are float32 0/1 sums, exact up to 2**24 -- far above any
    axon count -- so ``count > 0`` is bit-equivalent to the reference's
    boolean ``any``.
    """
    if state is not None:
        zero_r = np.stack([row_dead[state.rows_of(k)] for k in batch.core_positions])
        zero_c = np.stack([col_dead[state.cols_of(k)] for k in batch.core_positions])
    else:
        zero_r, zero_c = row_dead.copy(), col_dead.copy()
    zero_r &= ~batch.exempt_rows_m
    zero_c &= ~batch.exempt_cols_m

    conn_t = batch.conn_f.T
    while True:
        alive_c = (~zero_c).astype(np.float32)
        row_has_target = (alive_c @ conn_t) > 0.0
        row_dies = (~row_has_target
                    & batch.has_conn_row[None, :]
                    & ~zero_r & ~batch.exempt_rows_m)

        alive_r = (~zero_r).astype(np.float32)
        col_has_source = ((alive_r @ batch.conn_f) > 0.0) | batch.implicit_m
        col_dies = (~col_has_source
                    & batch.has_conn_col[None, :]
                    & ~zero_c & ~batch.exempt_cols_m)

        if not (row_dies.any() or col_dies.any()):
            break
        zero_r |= row_dies
        zero_c |= col_dies
    return zero_r, zero_c
