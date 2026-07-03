from __future__ import annotations
from typing import Dict, Sequence, Set, Tuple
import numpy as np
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.boundary_policy import compute_model_io_boundary_policy
from mimarsinan.mapping.pruning.graph.pruning_graph_types import GlobalPruningResult
def _force_dead_nodes_fully_pruned(
    graph: IRGraph,
    dead_node_ids: Sequence[int],
    result: GlobalPruningResult,
) -> None:
    """Mark every neuron / axon of a DEAD node as pruned, so ``_rewire_sources`` rewrites
    each consumer reference to ``IRSource(-1, 0)`` before the node is removed."""
    if not dead_node_ids:
        return
    dead_set = set(dead_node_ids)
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.id not in dead_set:
            continue
        if node.core_matrix is not None:
            n_axons, n_neurons = node.core_matrix.shape
        else:
            try:
                mat = node.get_core_matrix(graph)
                n_axons, n_neurons = mat.shape
            except Exception:
                continue
        result.pruned_rows_per_node[node.id] = set(range(n_axons))
        result.pruned_cols_per_node[node.id] = set(range(n_neurons))


def _masks_to_sets(
    row_mask: Sequence[bool], col_mask: Sequence[bool]
) -> Tuple[Set[int], Set[int]]:
    """Convert boolean ``True = pruned`` masks to index sets."""
    return (
        {i for i, m in enumerate(row_mask) if m},
        {j for j, m in enumerate(col_mask) if m},
    )


def _boundary_policy_exemptions(
    graph: IRGraph,
) -> Tuple[Dict[int, frozenset], Dict[int, frozenset]]:
    """Per-node frozensets of rows/cols that must never be pruned (model I/O)."""
    policy = compute_model_io_boundary_policy(graph)
    return policy.exempt_rows_per_node, policy.exempt_cols_per_node


def _collect_initial_seeds(
    graph: IRGraph,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None,
) -> Tuple[
    Dict[int, Tuple[Set[int], Set[int]]],
    Dict[int, Tuple[Set[int], Set[int]]],
]:
    """Convert model-supplied boolean masks into solver seed sets.

    Masks are clipped or zero-padded to the actual core / bank shape. Bias-row
    handling is implicit because callers pass ``False`` for bias.
    """
    seed_per_node: Dict[int, Tuple[Set[int], Set[int]]] = {}
    if initial_pruned_per_node:
        node_by_id = {n.id: n for n in graph.nodes if isinstance(n, NeuralCore)}
        for nid, (rows, cols) in initial_pruned_per_node.items():
            node = node_by_id.get(nid)
            if node is None or node.core_matrix is None:
                continue
            n_axons, n_neurons = node.core_matrix.shape
            row_mask = list(rows[:n_axons]) + [False] * max(0, n_axons - len(rows))
            col_mask = list(cols[:n_neurons]) + [False] * max(0, n_neurons - len(cols))
            seed_per_node[nid] = _masks_to_sets(row_mask, col_mask)

    seed_per_bank: Dict[int, Tuple[Set[int], Set[int]]] = {}
    banks = getattr(graph, "weight_banks", None) or {}
    if initial_pruned_per_bank:
        for bid, (rows, cols) in initial_pruned_per_bank.items():
            bank = banks.get(bid)
            if bank is None:
                continue
            n_axons, n_neurons = bank.core_matrix.shape
            if len(rows) != n_axons or len(cols) != n_neurons:
                continue
            seed_per_bank[bid] = _masks_to_sets(rows, cols)

    return seed_per_node, seed_per_bank


def _log_value_based_summary(result: GlobalPruningResult) -> None:
    """One-line summary when no model masks were supplied (zero-threshold prune)."""
    total_r = sum(len(s) for s in result.pruned_rows_per_node.values())
    total_c = sum(len(s) for s in result.pruned_cols_per_node.values())
    n_with_any = sum(
        1
        for nid in result.pruned_rows_per_node
        if result.pruned_rows_per_node[nid] or result.pruned_cols_per_node.get(nid)
    )
    print(
        f"[Pruning] prune_ir_graph: no model masks; value-based (zero_threshold): "
        f"nodes_with_pruned_rc={n_with_any} total_pruned_rows={total_r} total_pruned_cols={total_c}"
    )


def _attach_pre_compaction_metadata(
    graph: IRGraph,
    result: GlobalPruningResult,
    *,
    store_heatmap: bool,
) -> None:
    """Stash original matrix and prune masks for owned cores (used by GUI heatmaps)."""
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        mat = node.core_matrix
        n_axons, n_neurons = mat.shape
        zero_rows = result.pruned_rows_per_node.get(node.id, set())
        zero_cols = result.pruned_cols_per_node.get(node.id, set())
        node.pre_pruning_heatmap = (
            np.asarray(mat, dtype=np.float32) if store_heatmap else None
        )
        node.pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        node.pruned_row_mask = [i in zero_rows for i in range(n_axons)]


def _rewire_sources(
    graph: IRGraph,
    pruned_cols_per_node: Dict[int, Set[int]],
) -> None:
    """Replace pruned-neuron sources with off and reindex surviving neurons (owned-matrix cores only).

    Bank-backed cores keep their indexing (compacted later at the soft-core stage). Mutates
    every node's ``input_sources`` and the graph's ``output_sources`` in place.
    """
    owned_node_ids = {
        n.id for n in graph.nodes
        if isinstance(n, NeuralCore) and n.core_matrix is not None
    }
    pruned_cols: Dict[int, Set[int]] = {
        nid: pruned_cols_per_node.get(nid, set()) for nid in owned_node_ids
    }
    reindex_maps: Dict[int, Dict[int, int]] = {}
    for node in graph.nodes:
        if node.id not in owned_node_ids:
            continue
        n_neurons = node.core_matrix.shape[1]
        zero_cols = pruned_cols[node.id]
        new_idx = 0
        remap: Dict[int, int] = {}
        for old_idx in range(n_neurons):
            if old_idx not in zero_cols:
                remap[old_idx] = new_idx
                new_idx += 1
        reindex_maps[node.id] = remap

    def _rewire(sources: np.ndarray) -> np.ndarray:
        flat = sources.flatten()
        for i, src in enumerate(flat):
            if not isinstance(src, IRSource) or src.node_id < 0:
                continue
            if src.node_id not in owned_node_ids:
                continue
            if src.index in pruned_cols[src.node_id]:
                flat[i] = IRSource(node_id=-1, index=0)
            elif src.index in reindex_maps[src.node_id]:
                flat[i] = IRSource(
                    node_id=src.node_id,
                    index=reindex_maps[src.node_id][src.index],
                )
            else:
                raise ValueError(
                    f"prune_ir_graph: source index {src.index} for node_id {src.node_id} "
                    "is neither pruned nor in reindex map; bookkeeping error upstream."
                )
        return flat.reshape(sources.shape)

    for node in graph.nodes:
        if hasattr(node, "input_sources"):
            node.input_sources = _rewire(node.input_sources)

    if graph.output_sources.size:
        graph.output_sources = _rewire(graph.output_sources)

