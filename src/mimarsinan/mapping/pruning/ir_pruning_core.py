from __future__ import annotations
from typing import Dict, Sequence, Tuple
from mimarsinan.mapping.ir import IRGraph, NeuralCore
from mimarsinan.mapping.pruning.boundary_policy import assert_unified_ir_for_pruning
from mimarsinan.mapping.pruning.ir_liveness import NodeLiveness, compute_liveness
from mimarsinan.mapping.pruning.graph.pruning_graph_core import compute_global_pruned_sets
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
def prune_ir_graph(
    ir_graph: IRGraph,
    zero_threshold: float = 1e-8,
    *,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    store_heatmap: bool = False,
    simulation_steps: int = 32,
    spiking_mode: str = "lif",
) -> IRGraph:
    """Prune and compact ``ir_graph`` in place; return the same instance.

    Pruning is bidirectional/recursive across NeuralCore boundaries; ComputeOps block
    functional propagation. Model input data axons and output logits are never pruned;
    DEAD cores are deleted, surviving cores compacted.
    """
    if not ir_graph.nodes:
        return ir_graph

    graph = ir_graph
    assert_unified_ir_for_pruning(graph)

    exempt_rows, exempt_cols = _boundary_policy_exemptions(graph)
    seed_per_node, seed_per_bank = _collect_initial_seeds(
        graph, initial_pruned_per_node, initial_pruned_per_bank
    )

    result = compute_global_pruned_sets(
        graph,
        zero_threshold=zero_threshold,
        initial_per_node=seed_per_node,
        initial_per_bank=seed_per_bank,
        exempt_rows_per_node=exempt_rows,
        exempt_cols_per_node=exempt_cols,
    )

    if not (initial_pruned_per_node or initial_pruned_per_bank):
        _log_value_based_summary(result)

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
