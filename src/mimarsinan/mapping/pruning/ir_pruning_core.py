from __future__ import annotations
from typing import Dict, Sequence, Tuple
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.boundary_policy import assert_unified_ir_for_pruning
from mimarsinan.mapping.pruning.ir_liveness import NodeLiveness, compute_liveness
from mimarsinan.mapping.pruning.graph.propagation_mode import (
    ELIMINATION_PROPAGATION_CASCADE,
    require_elimination_propagation,
)
from mimarsinan.mapping.pruning.graph.constant_folding import apply_constant_folds
from mimarsinan.mapping.pruning.graph.pruning_graph_core import compute_global_pruned_sets
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
from mimarsinan.mapping.pruning.ir_pruning_helpers import (
    _attach_pre_compaction_metadata,
    _boundary_policy_exemptions,
    _collect_initial_seeds,
    _force_dead_nodes_fully_pruned,
    _log_value_based_summary,
    _rewire_sources,
)
from mimarsinan.mapping.pruning.ir_pruning_compact import (
    _attach_bank_metadata,
    _compact_node,
    _reset_post_compaction_masks,
    _validate_outputs_remain,
)
from mimarsinan.mapping.pruning.liveness_transfer import (
    DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    DEFAULT_ELIMINATION_CONSTANT_FOLDING,
)
def prune_ir_graph(
    ir_graph: IRGraph,
    zero_threshold: float = 1e-8,
    *,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    store_heatmap: bool = False,
    simulation_steps: int = 32,
    spiking_mode: str = "lif",
    elimination_propagation: str = ELIMINATION_PROPAGATION_CASCADE,
    computeop_liveness_transfers: str = DEFAULT_COMPUTEOP_LIVENESS_TRANSFERS,
    elimination_constant_folding: str = DEFAULT_ELIMINATION_CONSTANT_FOLDING,
    precomputed_result: "GlobalPruningResult | None" = None,
) -> IRGraph:
    """Prune and compact ``ir_graph`` in place; return the same instance.

    Under the default ``elimination_propagation="cascade"`` pruning is
    bidirectional/recursive across NeuralCore boundaries; ``"closure"`` stops
    at one-hop seed-group coupling, ``"masked"`` reclaims only the seeds
    (the allocation-naive lower bound). ComputeOps relay deadness through
    their registered liveness transfers (``computeop_liveness_transfers=
    "full"``: elementwise activations, index bijections, pooling regions;
    ``"identity_only"`` reproduces the pre-W4b identity-relay barrier);
    unknown ops stay opaque barriers. With
    ``elimination_constant_folding="full"`` [W4b-2] a CONST axon row is
    folded onto its core's constant carrier and eliminated exactly (the
    ``"off"`` kill-switch reproduces the zero-only cascade byte for byte).
    Model input data axons and output logits are never pruned; DEAD cores are
    deleted, surviving cores compacted.
    """
    elimination_propagation = require_elimination_propagation(
        elimination_propagation
    )
    if not ir_graph.nodes:
        return ir_graph

    graph = ir_graph
    assert_unified_ir_for_pruning(graph)

    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        graph, initial_pruned_per_node, initial_pruned_per_bank
    )

    # The caller may already hold this exact analysis: the ledger's arm run
    # computes it over the SAME graph, seeds, exemptions and policy, and
    # nothing between the two mutates the graph (``compute_elimination_arms``
    # never mutates, and the record emitter only reads). Recomputing it is a
    # second full fixpoint for a result we already have.
    result = precomputed_result
    if result is None:
        result = compute_global_pruned_sets(
            graph,
            zero_threshold=zero_threshold,
            initial_per_node=seed_per_node,
            initial_per_bank=seed_per_bank,
            exempt_rows_per_node=exempt_rows,
            exempt_cols_per_node=exempt_cols,
            mode=elimination_propagation,
            computeop_liveness_transfers=computeop_liveness_transfers,
            elimination_constant_folding=elimination_constant_folding,
            spiking_mode=spiking_mode,
        )

    if not (initial_pruned_per_node or initial_pruned_per_bank):
        _log_value_based_summary(result)

    # Materialize the analysis: every CONST row's contribution moves onto its
    # core's carrier BEFORE liveness / metadata / compaction read the weights,
    # so all of them see the program the chip will actually run.
    folded_cores = apply_constant_folds(graph, result.constant_folds)
    if folded_cores:
        print(
            f"[Pruning] constant folding: {result.constant_folds.total_folded_rows()} "
            f"axon row(s) folded onto the carriers of {folded_cores} core(s)"
        )

    _attach_pre_compaction_metadata(graph, result, store_heatmap=store_heatmap)

    liveness = compute_liveness(
        graph,
        simulation_steps=simulation_steps,
        spiking_mode=spiking_mode,
        pruning_result=result,
        zero_threshold=zero_threshold,
    )
    dead_node_ids = sorted(
        nid for nid, status in liveness.per_node.items()
        if status == NodeLiveness.DEAD
    )
    _force_dead_nodes_fully_pruned(graph, dead_node_ids, result)
    _rewire_sources(graph, result.pruned_cols_per_node)
    _validate_outputs_remain(graph)

    if dead_node_ids:
        graph.remove_nodes(dead_node_ids)
        print(
            f"[Pruning] prune_ir_graph: removed {len(dead_node_ids)} DEAD "
            f"NeuralCore(s) after liveness analysis"
        )

    for node in graph.nodes:
        if isinstance(node, NeuralCore) and node.core_matrix is not None:
            _compact_node(
                node,
                pruned_rows=result.pruned_rows_per_node.get(node.id, set()),
                pruned_cols=result.pruned_cols_per_node.get(node.id, set()),
            )

    _reset_post_compaction_masks(graph)
    _attach_bank_metadata(graph, result, store_heatmap=store_heatmap)

    return graph
