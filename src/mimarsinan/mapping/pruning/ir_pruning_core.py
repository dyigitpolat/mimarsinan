from __future__ import annotations
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank
from mimarsinan.mapping.pruning.boundary_policy import assert_unified_ir_for_pruning
from mimarsinan.mapping.pruning.ir_liveness import NodeLiveness, compute_liveness
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

    Pruning is bidirectional and recursive across NeuralCore boundaries.
    ComputeOp nodes block functional cross-core propagation but structural
    deadness (pruned upstream neurons) still flows through ComputeOp wiring.
    Model-level input data axons (``IRSource.node_id == -2``) and model
    output logits (entries in ``ir_graph.output_sources``) are never pruned.

    After propagation, every NeuralCore is classified by
    :func:`mimarsinan.mapping.pruning.ir_liveness.compute_liveness` as
    ``LIVE`` / ``BIAS_ONLY`` / ``DEAD``. ``DEAD`` nodes are deleted via
    :meth:`IRGraph.remove_nodes` so they no longer consume hardware slots,
    simulation cycles, or UI surface area. Surviving nodes are physically
    compacted in place.

    Args:
        simulation_steps: Integration window for bias-only liveness.
        spiking_mode: ``lif``, ``ttfs``, or ``ttfs_quantized``; forwarded to
            :func:`compute_liveness`.
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
