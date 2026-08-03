"""The flat cascade: Jacobi waves over flat state, phase-ordered like the reference.

One wave =
  (1) constant-lattice sweep + commit (the UNCHANGED machinery, exactly where
      the reference loop runs it),
  (2) every core's cross-core deadness computed from the PRE-WAVE state via the
      flat kernels, then within-matrix propagation per core (the already-
      vectorized ``compute_propagated_pruned_rows_cols``, verbatim), commits
      applied after all cores are computed -- Jacobi across cores,
  (3) the bank kernels, reading the just-committed node sets -- the same
      node-then-bank phase order the reference sweep uses.

Monotone operators reach one least fixpoint regardless of visit order, so the
final sets equal the reference's Gauss-Seidel result; that equality is not
assumed but enforced by ``TestJacobiNeverRegressesGaussSeidel``. The returned
wave count is canonical (order-independent); the reference's sweep count is an
artifact of node-id visit order and differs by design.
"""

from __future__ import annotations

from typing import Dict, Set

import numpy as np

from mimarsinan.mapping.pruning.graph.constant_folding import refresh_constant_folds
from mimarsinan.mapping.pruning.graph.flat.kernels import (
    flat_cross_core_dead_axons,
    flat_orphan_neurons,
)
from mimarsinan.mapping.pruning.graph.flat.state import FlatState, build_flat_state
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cols_with_nonzero_bias,
    _refresh_bank_pruning,
)
from mimarsinan.mapping.pruning.graph.pruning_propagation import (
    compute_propagated_pruned_rows_cols,
)

__all__ = ["run_cascade_waves"]


def _sync_masks(ctx, state: FlatState, row_dead: np.ndarray, col_dead: np.ndarray) -> None:
    """Mirror the context's per-node sets into the flat masks, EXACTLY.

    The ONE sync point (start of every wave, after the lattice commit): the
    lattice commit and the node commits both mutate the ctx sets, and the sets
    are REPLACED wholesale by the kernels, so the mirror is rebuilt rather
    than accumulated -- an add-only mirror went stale the first time a fold
    killed a row outside the node loop, which is precisely how the flat
    engine silently lost the bias-only collapse.
    """
    row_dead[:] = False
    col_dead[:] = False
    for k, nid in enumerate(state.node_ids):
        rows = state.rows_of(k)
        cols = state.cols_of(k)
        for i in ctx.pruned_rows.get(nid, ()):
            if 0 <= i < rows.stop - rows.start:
                row_dead[rows.start + i] = True
        for j in ctx.pruned_cols.get(nid, ()):
            if 0 <= j < cols.stop - cols.start:
                col_dead[cols.start + j] = True


def run_cascade_waves(ctx) -> int:
    """Drive ctx to the cascade fixpoint; returns the canonical wave count."""
    state = build_flat_state(ctx)
    row_dead = np.zeros(state.n_rows, dtype=bool)
    col_dead = np.zeros(state.n_cols, dtype=bool)
    _sync_masks(ctx, state, row_dead, col_dead)

    # Static per-core facts (folding may mutate bias/matrix via the carrier;
    # ctx.node_matrix / node_bias always return the EFFECTIVE structures, so
    # they are re-read per wave rather than cached here).
    exempt_rows: Dict[int, frozenset] = {
        nid: frozenset(ctx.exempt_rows.get(nid, frozenset())) for nid in state.node_ids
    }
    exempt_cols: Dict[int, frozenset] = {
        nid: frozenset(ctx.exempt_cols.get(nid, frozenset())) for nid in state.node_ids
    }

    # A core's within-matrix result is a pure function of its seed sets and
    # its (effective) matrix; when neither moved since the last wave, the call
    # is skipped. Folding can move the matrix through the carrier delta, so a
    # committed fold re-arms its core explicitly below.
    last_seed_rows: Dict[int, Set[int]] = {}
    last_seed_cols: Dict[int, Set[int]] = {}

    waves = 0
    while True:
        waves += 1
        sweep = refresh_constant_folds(ctx)
        folded_now = {fold[0] for fold in getattr(sweep, "folds", ()) or ()}
        for nid in folded_now:
            last_seed_rows.pop(nid, None)      # matrix moved; must re-derive
        changed = sweep.commit(ctx)
        _sync_masks(ctx, state, row_dead, col_dead)

        # -- node phase: Jacobi across cores ------------------------------
        dead_axons = flat_cross_core_dead_axons(state, col_dead)
        orphans = flat_orphan_neurons(state, row_dead)

        commits: list[tuple[int, int, Set[int], Set[int]]] = []
        for k, node in enumerate(ctx.neural_cores):
            mat = ctx.node_matrix(node)
            if mat is None:
                continue
            nid = node.id
            rows = state.rows_of(k)
            cols = state.cols_of(k)
            cross_rows = {
                int(i) for i in np.flatnonzero(dead_axons[rows])
            } - exempt_rows[nid]
            cross_cols = {
                int(j) for j in np.flatnonzero(orphans[cols])
            } - exempt_cols[nid]

            seed_rows = ctx.pruned_rows[nid] | cross_rows
            seed_cols = ctx.pruned_cols[nid] | cross_cols
            if (last_seed_rows.get(nid) == seed_rows
                    and last_seed_cols.get(nid) == seed_cols):
                continue
            last_seed_rows[nid] = seed_rows
            last_seed_cols[nid] = seed_cols
            bias = ctx.node_bias(node)
            new_rows, new_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=ctx.zero_threshold,
                initial_zero_rows=seed_rows,
                initial_zero_cols=seed_cols,
                exempt_rows=exempt_rows[nid],
                exempt_cols=exempt_cols[nid],
                cols_with_implicit_source=_cols_with_nonzero_bias(
                    bias, mat.shape[1], ctx.zero_threshold
                ),
                mode="cascade",
            )
            if new_rows != ctx.pruned_rows[nid] or new_cols != ctx.pruned_cols[nid]:
                commits.append((k, nid, new_rows, new_cols))

        for _k, nid, new_rows, new_cols in commits:
            changed = True
            ctx.pruned_rows[nid] = new_rows
            ctx.pruned_cols[nid] = new_cols

        # -- bank phase: same position as the reference sweep --------------
        for bank_id, bank in ctx.banks.items():
            if _refresh_bank_pruning(
                bank=bank,
                bank_id=bank_id,
                zero_threshold=ctx.zero_threshold,
                bank_nodes=ctx.bank_node_lookup[bank_id],
                bank_consumers=ctx.bank_consumers.get(bank_id, set()),
                model_output_neurons=ctx.model_output_neurons,
                pruned_rows=ctx.pruned_rows,
                pruned_cols=ctx.pruned_cols,
                bank_pruned_rows=ctx.bank_pruned_rows,
                bank_pruned_cols=ctx.bank_pruned_cols,
                exempt_rows=ctx.exempt_rows,
                exempt_cols=ctx.exempt_cols,
            ):
                changed = True

        if not changed:
            break
    return waves
