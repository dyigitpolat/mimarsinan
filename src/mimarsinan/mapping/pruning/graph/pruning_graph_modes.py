"""Masked and closure arms of the global elimination-propagation axis."""

from __future__ import annotations

from typing import Dict, Mapping, Set

from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_MASKED,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _cross_core_dead_axons,
    _refresh_bank_pruning,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    GlobalPruningContext,
)


def run_bank_alias_fixpoint(ctx: GlobalPruningContext, *, kernel_mode: str) -> None:
    """Reconcile node-coordinate and bank-coordinate kill sets until stable.

    This is pure coordinate ALIASING of shared physical structure, not
    propagation: a bank row dies only when dead in every sharing node
    (union rule), a bank column aggregates the per-node views, and bank-level
    kills project back into every sharing node. With ``kernel_mode="masked"``
    the within-bank kernel adds nothing beyond exemption filtering, so the
    masked/closure arms alias without discovering any deadness.
    """
    while True:
        changed = False
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
                mode=kernel_mode,
            ):
                changed = True
        if not changed:
            break


def run_masked(ctx: GlobalPruningContext) -> None:
    """Allocation-naive LOWER BOUND: exactly the seeded, exemption-filtered
    sets — no coupling, no emergent deadness, no iteration. Only the bank
    aliasing closure runs, because seeds on shared physical structure must
    stay coordinate-consistent (union rule)."""
    run_bank_alias_fixpoint(ctx, kernel_mode=ELIMINATION_PROPAGATION_MASKED)


def run_closure(ctx: GlobalPruningContext) -> None:
    """ONE-HOP seed-group coupling — the DepGraph/torch-pruning-equivalent
    baseline. From the masked (seed) state, apply exactly one coupling step:

    - forward: a dead producer neuron kills the reading axon row in every
      DIRECT consumer (identity ComputeOp relays included);
    - backward: a producer neuron every reader of which is seed-dead is the
      paired member of the same seed group and dies with it (owned-matrix
      producers only — killing a shared bank column for one instance's group
      is a cascade-level decision).

    NO emergent-deadness discovery (a neuron whose inputs all died stays
    alive) and NO iteration; a final bank-aliasing pass keeps shared
    structure coordinate-consistent.
    """
    run_masked(ctx)
    seed_rows: Dict[int, Set[int]] = {
        nid: set(s) for nid, s in ctx.pruned_rows.items()
    }
    seed_cols: Dict[int, Set[int]] = {
        nid: set(s) for nid, s in ctx.pruned_cols.items()
    }

    coupled_rows = _forward_consumer_coupling(ctx, seed_cols)
    coupled_cols = _backward_producer_coupling(ctx, seed_rows, seed_cols)

    for nid, rows in coupled_rows.items():
        ctx.pruned_rows[nid] |= rows
    for nid, cols in coupled_cols.items():
        ctx.pruned_cols[nid] |= cols

    run_bank_alias_fixpoint(ctx, kernel_mode=ELIMINATION_PROPAGATION_MASKED)


def _forward_consumer_coupling(
    ctx: GlobalPruningContext,
    seed_cols: Mapping[int, Set[int]],
) -> Dict[int, Set[int]]:
    """Axon rows whose source neuron is seed-dead: one hop, no iteration."""
    out: Dict[int, Set[int]] = {}
    for node in ctx.neural_cores:
        dead = _cross_core_dead_axons(
            node, seed_cols, ctx.computeop_producer_map
        )
        new = (dead - ctx.exempt_rows.get(node.id, frozenset())) - \
            ctx.pruned_rows[node.id]
        if new:
            out[node.id] = new
    return out


def _backward_producer_coupling(
    ctx: GlobalPruningContext,
    seed_rows: Mapping[int, Set[int]],
    seed_cols: Mapping[int, Set[int]],
) -> Dict[int, Set[int]]:
    """Producer neurons whose every reader is seed-dead (the paired group member).

    Conservative guards mirror the cascade's orphan protections: model-output
    neurons, ComputeOp-referenced neurons, exempt columns, and zero-consumer
    neurons (never touched by any seed group) all survive.
    """
    owned_ids = {
        n.id for n in ctx.neural_cores if n.core_matrix is not None
    }
    out: Dict[int, Set[int]] = {}
    for (m, k), consumers in ctx.consumer_axons.items():
        if m not in owned_ids or not consumers:
            continue
        if k in seed_cols.get(m, set()):
            continue
        if k in ctx.exempt_cols.get(m, frozenset()):
            continue
        if (m, k) in ctx.model_output_neurons:
            continue
        if (m, k) in ctx.computeop_referenced:
            continue
        if all(i in seed_rows.get(n, set()) for n, i in consumers):
            out.setdefault(m, set()).add(k)
    return out
