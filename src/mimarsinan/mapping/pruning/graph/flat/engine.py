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

import time
from typing import Dict, Set

import numpy as np

from mimarsinan.mapping.pruning.graph.constant_folding import refresh_constant_folds
from mimarsinan.mapping.pruning.graph.flat.kernels import (
    build_bank_batches,
    flat_cross_core_dead_axons,
    flat_orphan_neurons,
    flat_within_matrix_fixpoint,
)
from mimarsinan.mapping.pruning.graph.flat.state import build_flat_state
from mimarsinan.mapping.pruning.graph.flat.verify import (
    FlatEngineQuiescenceError,
    _sync_masks,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cols_with_nonzero_bias,
    _refresh_bank_pruning,
)
from mimarsinan.mapping.pruning.graph.pruning_propagation import (
    compute_propagated_pruned_rows_cols,
)

__all__ = ["FlatEngineQuiescenceError", "run_cascade_waves"]




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

    # [P4a] bank-backed instances share an immutable view (their carriers are
    # read-only, so no fold can move it): their within-matrix fixpoints run as
    # one batched kernel per shared view. Owned cores (which can fold) keep the
    # per-core reference call.
    batches = build_bank_batches(ctx, state)
    batched = {k for b in batches for k in b.core_positions}

    # [P4b] the lattice gather re-derives a core only when something it reads
    # moved: a port it reads descended, a producer's columns changed, its own
    # sets changed, or it folded. Wave 1 gathers everything. The relation is
    # allowed to OVER-approximate (costs a re-derivation) and is checked
    # against UNDER-approximation by the final full gather below.
    rearm = None
    node_ids = state.node_ids

    waves = 0
    while True:
        waves += 1
        _tw = time.perf_counter()
        # ops write their descents STRAIGHT into the lattice ("wiring, not a
        # hop"), bypassing sweep.descents -- diffing the keys captures them.
        keys_before = set(ctx.constants.lattice.values)
        sweep = refresh_constant_folds(ctx, only_ids=rearm)
        op_descents = set(ctx.constants.lattice.values) - keys_before
        folded_now = {fold[0] for fold in getattr(sweep, "folds", ()) or ()}
        descents_now = [port for port, _v in getattr(sweep, "descents", ()) or ()]
        lattice_killed = set(getattr(sweep, "dead_rows", {}) or {})
        for nid in folded_now:
            last_seed_rows.pop(nid, None)      # matrix moved; must re-derive
        changed = sweep.commit(ctx)
        _sync_masks(ctx, state, row_dead, col_dead)
        _ts = time.perf_counter()

        # -- node phase: Jacobi across cores ------------------------------
        dead_axons = flat_cross_core_dead_axons(state, col_dead)
        orphans = flat_orphan_neurons(state, row_dead)

        commits: list[tuple[int, int, Set[int], Set[int]]] = []

        # batched node phase: seeds = current state ∪ cross deadness, gathered
        # straight from the flat masks; exempt filtering happens at batch init
        # exactly where the reference function applies it.
        seed_rows_flat = row_dead | dead_axons
        seed_cols_flat = col_dead | orphans
        for batch in batches:
            new_r, new_c = flat_within_matrix_fixpoint(
                batch, seed_rows_flat, seed_cols_flat, state
            )
            for pos, k in enumerate(batch.core_positions):
                nid = state.node_ids[k]
                rows = state.rows_of(k)
                cols = state.cols_of(k)
                if (np.array_equal(new_r[pos], row_dead[rows])
                        and np.array_equal(new_c[pos], col_dead[cols])):
                    continue
                commits.append((
                    k, nid,
                    {int(i) for i in np.flatnonzero(new_r[pos])},
                    {int(j) for j in np.flatnonzero(new_c[pos])},
                ))

        for k, node in enumerate(ctx.neural_cores):
            if k in batched:
                continue
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

        _tn = time.perf_counter()
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

        rearm = set()
        for port in op_descents:
            for k in state.port_readers.get(tuple(port), ()):
                rearm.add(node_ids[k])
        for port in descents_now:
            for k in state.port_readers.get(tuple(port), ()):
                rearm.add(node_ids[k])
        rearm |= folded_now | lattice_killed
        for _k, nid, _r, _c in commits:
            rearm.add(nid)
            for rk in state.core_readers.get(nid, ()):
                rearm.add(node_ids[rk])

        if time.perf_counter() - _tw > 5.0:   # slow waves only; tests stay silent
            print(f"[CascadeWave] w={waves} sweep={_ts - _tw:.1f}s "
                  f"nodes={_tn - _ts:.1f}s banks+rearm={time.perf_counter() - _tn:.1f}s "
                  f"commits={len(commits)} rearm={len(rearm)}", flush=True)
        if not changed:
            break

    # The contract lock: one unrestricted gather that MUST be a no-op.
    verify = refresh_constant_folds(ctx)
    if verify.commit(ctx):
        raise FlatEngineQuiescenceError(
            "the final full lattice gather produced new facts after the "
            "change-tracked fixpoint reported quiescence: a re-arm edge is "
            "missing (state.port_readers / core_readers)"
        )
    return waves
