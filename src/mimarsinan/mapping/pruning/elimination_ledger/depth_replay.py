"""Per-kill propagation-depth replay: synchronous BFS waves of the cascade operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import numpy as np

from mimarsinan.mapping.pruning.graph.constant_folding import (
    refresh_constant_folds,
)
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_modes import (
    run_bank_alias_fixpoint,
    run_closure,
    run_masked,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cols_with_nonzero_bias,
    _cross_core_dead_axons,
    _orphan_neurons,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    GlobalPruningContext,
)
from mimarsinan.mapping.pruning.graph.pruning_propagation import (
    matrix_one_step_deaths,
)


@dataclass
class DepthReplay:
    """Kill sets labeled with the wave index (depth) each kill became forced.

    Seeds are depth 0. Wave t applies ONE causal step of every cascade
    operator (consumer coupling, orphan discovery, within-matrix/within-bank
    starvation) to the depth <= t-1 state, then closes coordinate aliasing
    through the shared banks at the same depth. Iterating to quiescence
    reproduces the production fixpoint exactly — the ledger builder asserts
    that.
    """

    row_depths: Dict[int, Dict[int, int]] = field(default_factory=dict)
    col_depths: Dict[int, Dict[int, int]] = field(default_factory=dict)
    bank_row_depths: Dict[int, Dict[int, int]] = field(default_factory=dict)
    bank_col_depths: Dict[int, Dict[int, int]] = field(default_factory=dict)
    waves: int = 0


def replay_kill_depths(ctx: GlobalPruningContext, *, mode: str) -> DepthReplay:
    """Replay ``mode`` on a freshly seeded context, labeling kills with depths.

    Mutates ``ctx`` state to the mode's final kill sets (the caller verifies
    them against the production run).
    """
    run_masked(ctx)
    replay = DepthReplay()
    _label_state(ctx, replay, depth=0)
    if mode == ELIMINATION_PROPAGATION_MASKED:
        return replay
    if mode == ELIMINATION_PROPAGATION_CLOSURE:
        run_closure(ctx)
        _label_state(ctx, replay, depth=1)
        replay.waves = 1
        return replay

    depth = 0
    while True:
        depth += 1
        if not _cascade_wave(ctx, replay, depth):
            replay.waves = depth - 1
            break
    return replay


def _cascade_wave(
    ctx: GlobalPruningContext, replay: DepthReplay, depth: int
) -> bool:
    """One synchronous causal wave + alias closure; True iff anything died.

    The constant lattice sweeps against the same PRE-wave state as every
    other operator and is committed alongside them, so the replay traverses
    folds identically to the production fixpoint and the ledger's
    reconciliation guard keeps its by-construction meaning.
    """
    sweep = refresh_constant_folds(ctx)
    new_rows, new_cols = _node_causal_kills(ctx)
    bank_rows, bank_cols = _bank_causal_kills(ctx)
    lattice_descended = sweep.commit(ctx)

    for nid, rows in new_rows.items():
        ctx.pruned_rows[nid] |= rows
    for nid, cols in new_cols.items():
        ctx.pruned_cols[nid] |= cols
    for bid, rows in bank_rows.items():
        ctx.bank_pruned_rows[bid] |= rows
    for bid, cols in bank_cols.items():
        ctx.bank_pruned_cols[bid] |= cols

    run_bank_alias_fixpoint(ctx, kernel_mode=ELIMINATION_PROPAGATION_MASKED)
    # A wave that only DESCENDED the lattice (a column became constant but its
    # readers sit behind an opaque op) kills nothing yet still enables the next
    # wave, so quiescence means "no kills AND no descents".
    return _label_state(ctx, replay, depth=depth) or lattice_descended


def _node_causal_kills(
    ctx: GlobalPruningContext,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """One step of consumer coupling + orphan discovery + matrix starvation."""
    new_rows: Dict[int, Set[int]] = {}
    new_cols: Dict[int, Set[int]] = {}
    for node in ctx.neural_cores:
        mat = ctx.node_matrix(node)
        if mat is None:
            continue
        nid = node.id
        n_neurons = mat.shape[1]
        exempt_r = ctx.exempt_rows.get(nid, frozenset())
        exempt_c = ctx.exempt_cols.get(nid, frozenset())

        rows = _cross_core_dead_axons(
            node, ctx.pruned_cols, ctx.computeop_transfers
        )
        cols = _orphan_neurons(
            nid,
            n_neurons,
            ctx.pruned_rows,
            ctx.consumer_axons,
            ctx.model_output_neurons,
            ctx.computeop_transfers,
        )
        starved_rows, starved_cols = matrix_one_step_deaths(
            mat,
            pruned_rows=ctx.pruned_rows[nid],
            pruned_cols=ctx.pruned_cols[nid],
            exempt_rows=exempt_r,
            exempt_cols=exempt_c,
            cols_with_implicit_source=_cols_with_nonzero_bias(
                ctx.node_bias(node), n_neurons, ctx.zero_threshold,
            ),
            zero_threshold=ctx.zero_threshold,
        )
        rows = ((rows | starved_rows) - exempt_r) - ctx.pruned_rows[nid]
        cols = ((cols | starved_cols) - exempt_c) - ctx.pruned_cols[nid]
        if rows:
            new_rows[nid] = rows
        if cols:
            new_cols[nid] = cols
    return new_rows, new_cols


def _bank_causal_kills(
    ctx: GlobalPruningContext,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """One step of within-bank starvation over the pre-wave bank state."""
    new_rows: Dict[int, Set[int]] = {}
    new_cols: Dict[int, Set[int]] = {}
    for bank_id, bank in ctx.banks.items():
        n_neurons = bank.core_matrix.shape[1]
        bank_exempt_rows: Set[int] = set()
        bank_exempt_cols: Set[int] = set()
        implicit_cols: Set[int] = set(_cols_with_nonzero_bias(
            getattr(bank, "hardware_bias", None), n_neurons,
            ctx.zero_threshold,
        ))
        for node in ctx.bank_node_lookup[bank_id]:
            start = (node.weight_row_slice or (0, n_neurons))[0]
            bank_exempt_rows |= set(ctx.exempt_rows.get(node.id, frozenset()))
            bank_exempt_cols |= {
                start + j
                for j in ctx.exempt_cols.get(node.id, frozenset())
            }
            node_bias = getattr(node, "hardware_bias", None)
            if node_bias is None:
                continue
            size = int(np.asarray(node_bias).size)
            implicit_cols |= {
                start + j
                for j in _cols_with_nonzero_bias(
                    node_bias, size, ctx.zero_threshold
                )
            }
        rows, cols = matrix_one_step_deaths(
            bank.core_matrix,
            pruned_rows=ctx.bank_pruned_rows[bank_id],
            pruned_cols=ctx.bank_pruned_cols[bank_id],
            exempt_rows=bank_exempt_rows,
            exempt_cols=bank_exempt_cols,
            cols_with_implicit_source=implicit_cols,
            zero_threshold=ctx.zero_threshold,
        )
        if rows:
            new_rows[bank_id] = rows
        if cols:
            new_cols[bank_id] = cols
    return new_rows, new_cols


def _label_state(
    ctx: GlobalPruningContext, replay: DepthReplay, *, depth: int
) -> bool:
    """Record ``depth`` for every kill not labeled yet; True iff any was new."""
    changed = False
    for per_key, labels in (
        (ctx.pruned_rows, replay.row_depths),
        (ctx.pruned_cols, replay.col_depths),
        (ctx.bank_pruned_rows, replay.bank_row_depths),
        (ctx.bank_pruned_cols, replay.bank_col_depths),
    ):
        for key, indices in per_key.items():
            labeled = labels.setdefault(key, {})
            for i in indices:
                if i not in labeled:
                    labeled[i] = depth
                    changed = True
    return changed
