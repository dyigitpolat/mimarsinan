from __future__ import annotations
from typing import AbstractSet, Mapping, Tuple
from mimarsinan.mapping.ir import IRGraph
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    ELIMINATION_PROPAGATION_CLOSURE,
    ELIMINATION_PROPAGATION_MASKED,
    require_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_modes import (
    run_closure,
    run_masked,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_refresh import (
    _refresh_bank_pruning,
    _refresh_node_pruning,
    _resolve_node_matrix,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_seeding import (
    GlobalPruningContext,
    build_global_pruning_context,
)
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
from mimarsinan.mapping.pruning.liveness_transfer import (
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
)
def compute_global_pruned_sets(
    graph: IRGraph,
    *,
    zero_threshold: float = 1e-8,
    initial_per_node: Mapping[int, Tuple[AbstractSet[int], AbstractSet[int]]] | None = None,
    initial_per_bank: Mapping[int, Tuple[AbstractSet[int], AbstractSet[int]]] | None = None,
    exempt_rows_per_node: Mapping[int, AbstractSet[int]] | None = None,
    exempt_cols_per_node: Mapping[int, AbstractSet[int]] | None = None,
    mode: str = ELIMINATION_PROPAGATION_CASCADE,
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
) -> GlobalPruningResult:
    """Run global pruning under one propagation arm (default: cascade fixpoint).

    Args:
        graph: Unified IR graph. Pruning runs pre-segmentation: ``IRSource``
            with ``node_id == -2`` denotes model input data, and entries in
            ``graph.output_sources`` denote model output logits.
        zero_threshold: Sum-of-abs threshold for value-based per-matrix init.
        initial_per_node: Optional ``{node_id: (rows, cols)}`` seed sets;
            indices that are also exempt are silently dropped.
        initial_per_bank: Optional ``{bank_id: (rows, cols)}`` seed sets in the
            bank's own coordinate system.
        exempt_rows_per_node: Per-node row indices that must never be pruned.
        exempt_cols_per_node: Per-node column indices that must never be pruned.
        mode: ``"masked"`` (allocation-naive lower bound: seeds only),
            ``"closure"`` (one-hop seed-group coupling, no emergent deadness,
            no iteration), or ``"cascade"`` (the bidirectional, recursive
            cross-core fixpoint — the default deployment path).
        computeop_liveness_transfers: ``"full"`` (default: per-op transfer
            functions relay deadness through elementwise activations, index
            bijections, and pooling regions) or ``"identity_only"`` (the
            pre-W4b relay, for A/B).
    """
    mode = require_elimination_propagation(mode)
    if not graph.nodes and not (getattr(graph, "weight_banks", {}) or {}):
        return GlobalPruningResult()

    ctx = build_global_pruning_context(
        graph,
        zero_threshold=zero_threshold,
        initial_per_node=initial_per_node,
        initial_per_bank=initial_per_bank,
        exempt_rows_per_node=exempt_rows_per_node,
        exempt_cols_per_node=exempt_cols_per_node,
        computeop_liveness_transfers=computeop_liveness_transfers,
    )
    if not ctx.neural_cores and not ctx.banks:
        return GlobalPruningResult()

    if mode == ELIMINATION_PROPAGATION_MASKED:
        run_masked(ctx)
        iterations = 0
    elif mode == ELIMINATION_PROPAGATION_CLOSURE:
        run_closure(ctx)
        iterations = 1
    else:
        iterations = _run_cascade_fixpoint(ctx)

    return ctx.to_result(fixpoint_iterations=iterations)


def _run_cascade_fixpoint(ctx: GlobalPruningContext) -> int:
    """The bidirectional cross-core fixpoint; returns the sweep count."""
    iterations = 0
    while True:
        iterations += 1
        changed = False
        for node in ctx.neural_cores:
            mat = _resolve_node_matrix(node, ctx.banks)
            if mat is None:
                continue
            if _refresh_node_pruning(
                node=node,
                mat=mat,
                zero_threshold=ctx.zero_threshold,
                pruned_rows=ctx.pruned_rows,
                pruned_cols=ctx.pruned_cols,
                consumer_axons=ctx.consumer_axons,
                model_output_neurons=ctx.model_output_neurons,
                computeop_transfers=ctx.computeop_transfers,
                exempt_rows=ctx.exempt_rows,
                exempt_cols=ctx.exempt_cols,
            ):
                changed = True

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
    return iterations
