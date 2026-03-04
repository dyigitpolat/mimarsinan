"""IR graph pruning: eliminate zeroed rows and columns from NeuralCores.

After the pruning adaptation step zeros out insignificant weights,
this post-processing pass physically removes the zeroed structure
from the IR graph:

1. Removes all-zero rows (axons) from core_matrix and input_sources.
2. Removes all-zero columns (neurons) from core_matrix.
3. Rewires downstream sources that referenced pruned neurons to off.
4. Reindexes remaining neuron references.

Uses pruning maps from the model when provided (initial_pruned_per_node /
initial_pruned_per_bank); otherwise infers from matrix values (zero_threshold).
Propagative pruning (row only feeds pruned cols, col only receives from pruned
rows) is centralized in pruning_propagation.compute_propagated_pruned_rows_cols.

Before compacting, stores pre-pruning heatmap and row/col masks on each
NeuralCore for GUI visualization (soft-core pre/post pruning views).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning_propagation import compute_propagated_pruned_rows_cols


def get_initial_pruning_masks_from_model(model, ir_graph: IRGraph):
    """Collect pruning masks from the model for use as initial sets in prune_ir_graph.

    Walks model.get_perceptrons() and IR neural cores in order. When counts match
    (1:1, no tiling), converts each layer's prune_mask / prune_bias_mask to IR
    convention (row_mask, col_mask) with True = pruned.

    Returns:
        (initial_pruned_per_node, initial_pruned_per_bank). Per-node dict maps
        node_id -> (row_mask_list, col_mask_list). Per-bank is left empty unless
        we have a way to map banks to perceptrons (future).
    """
    import torch
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    try:
        perceptrons = model.get_perceptrons()
    except Exception:
        return initial_pruned_per_node, initial_pruned_per_bank
    if len(neural_cores) != len(perceptrons):
        return initial_pruned_per_node, initial_pruned_per_bank
    for i, (node, p) in enumerate(zip(neural_cores, perceptrons)):
        layer = getattr(p, "layer", None)
        if layer is None:
            continue
        prune_mask = getattr(layer, "prune_mask", None)
        if prune_mask is None or not isinstance(prune_mask, torch.Tensor):
            continue
        pm = prune_mask.detach()
        out_f, in_f = pm.shape[0], pm.shape[1]
        prune_bias = getattr(layer, "prune_bias_mask", None)
        if prune_bias is not None:
            row_pruned = prune_bias.detach().cpu().numpy()  # (out_f,) True = pruned
        else:
            row_pruned = pm.any(dim=1).cpu().numpy()
        col_pruned = pm.any(dim=0).cpu().numpy()  # (in_f,) True = pruned
        # IR: (axons, neurons) = (in_f+1, out_f) with bias row last
        n_axons = in_f + 1
        n_neurons = out_f
        try:
            mat = node.get_core_matrix(ir_graph)
            nr, nc = mat.shape
            if nr != n_axons or nc != n_neurons:
                continue
        except Exception:
            continue
        ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)] + [False]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        initial_pruned_per_node[node.id] = (ir_row_mask, ir_col_mask)
    return initial_pruned_per_node, initial_pruned_per_bank


def _masks_to_sets(
    row_mask: Sequence[bool], col_mask: Sequence[bool]
) -> Tuple[Set[int], Set[int]]:
    """Convert boolean masks (True = pruned) to sets of indices."""
    zero_rows = {i for i, m in enumerate(row_mask) if m}
    zero_cols = {j for j, m in enumerate(col_mask) if m}
    return zero_rows, zero_cols


def prune_ir_graph(
    ir_graph: IRGraph,
    zero_threshold: float = 1e-8,
    *,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
) -> IRGraph:
    """Remove zeroed rows and columns from all NeuralCores in the IR graph.

    When initial_pruned_per_node or initial_pruned_per_bank are provided,
    uses those masks (True = pruned) as the initial pruned set, then runs
    propagative fixpoint. Otherwise infers initial set from matrix values
    below zero_threshold.

    Returns a new IRGraph with compacted cores and rewired sources.
    """
    if not ir_graph.nodes:
        return IRGraph(
            nodes=[],
            output_sources=ir_graph.output_sources.copy() if ir_graph.output_sources.size else ir_graph.output_sources,
            weight_banks=copy.deepcopy(ir_graph.weight_banks),
        )

    graph = copy.deepcopy(ir_graph)

    # Pre-Phase: Recursive Topological Pruning
    # Iterate until convergence:
    # 1. Any row connected to an 'off' source (-1) is dead -> zero out the row in core_matrix
    # 2. Any column not read by any downstream node or output is dead -> zero out the column
    changed = True
    while changed:
        changed = False

        # 1. Forward dead connections
        # If an input source is off (-1), its contribution is always 0.
        for node in graph.nodes:
            if not isinstance(node, NeuralCore) or node.core_matrix is None:
                continue
            
            flat_src = node.input_sources.flatten()
            for i, src in enumerate(flat_src):
                if getattr(src, "node_id", getattr(src, "core_", None)) == -1:
                    # If the row is not already all zeros, zero it out and flag change
                    if np.any(np.abs(node.core_matrix[i, :]) > zero_threshold):
                        node.core_matrix[i, :] = 0.0
                        changed = True

        # 2. Backward dead connections
        # Find all neurons that are actually read by something
        read_sources = set()
        
        # Read by output
        for src in graph.output_sources.flatten():
            node_id = getattr(src, "node_id", getattr(src, "core_", None))
            idx = getattr(src, "index", getattr(src, "neuron_", None))
            if node_id >= 0:
                read_sources.add((node_id, idx))
                
        # Read by other nodes
        for node in graph.nodes:
            if not hasattr(node, "input_sources"):
                continue
            for src in node.input_sources.flatten():
                node_id = getattr(src, "node_id", getattr(src, "core_", None))
                idx = getattr(src, "index", getattr(src, "neuron_", None))
                if node_id >= 0:
                    read_sources.add((node_id, idx))
                    
        # Zero out unread columns
        for node in graph.nodes:
            if not isinstance(node, NeuralCore) or node.core_matrix is None:
                continue
            
            n_neurons = node.core_matrix.shape[1]
            for j in range(n_neurons):
                if (node.id, j) not in read_sources:
                    if np.any(np.abs(node.core_matrix[:, j]) > zero_threshold):
                        node.core_matrix[:, j] = 0.0
                        changed = True

    # Phase 1: identify pruned rows/columns with forward/backward propagation per core
    # (centralized in compute_propagated_pruned_rows_cols)
    reindex_maps: Dict[int, Dict[int, int]] = {}
    pruned_neurons: Dict[int, Set[int]] = {}
    pruned_rows: Dict[int, Set[int]] = {}
    init_node = (initial_pruned_per_node or {})

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix  # (axons, neurons)
        n_axons, n_neurons = mat.shape

        init = init_node.get(node.id)
        if init is not None and len(init[0]) == n_axons and len(init[1]) == n_neurons:
            zero_rows, zero_cols = _masks_to_sets(init[0], init[1])
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=zero_threshold,
                initial_zero_rows=zero_rows,
                initial_zero_cols=zero_cols,
            )
        else:
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(mat, zero_threshold)

        pruned_neurons[node.id] = zero_cols
        pruned_rows[node.id] = zero_rows

        new_idx = 0
        remap = {}
        for old_idx in range(n_neurons):
            if old_idx not in zero_cols:
                remap[old_idx] = new_idx
                new_idx += 1
        reindex_maps[node.id] = remap

    # Phase 2: rewire all sources (input_sources of all nodes + output_sources)
    def _rewire_sources(sources: np.ndarray) -> np.ndarray:
        """Rewire source references: pruned neurons -> off, others -> reindexed."""
        flat = sources.flatten()
        for i, src in enumerate(flat):
            if not isinstance(src, IRSource):
                continue
            if src.node_id < 0:
                continue
            if src.node_id in pruned_neurons:
                if src.index in pruned_neurons[src.node_id]:
                    flat[i] = IRSource(node_id=-1, index=0)
                elif src.node_id in reindex_maps and src.index in reindex_maps[src.node_id]:
                    flat[i] = IRSource(node_id=src.node_id, index=reindex_maps[src.node_id][src.index])
        return flat.reshape(sources.shape)

    for node in graph.nodes:
        node.input_sources = _rewire_sources(node.input_sources)

    if graph.output_sources.size:
        graph.output_sources = _rewire_sources(graph.output_sources)

    # Before compacting: store pre-pruning heatmap and masks for GUI (soft-core viz)
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        mat = node.core_matrix
        n_axons, n_neurons = mat.shape
        zero_cols = pruned_neurons.get(node.id, set())
        zero_rows_set = pruned_rows.get(node.id, set())
        node.pre_pruning_heatmap = np.copy(mat).tolist()
        node.pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        node.pruned_row_mask = [r in zero_rows_set for r in range(n_axons)]

    # Phase 3: physically remove zeroed columns from core matrices
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        zero_cols = pruned_neurons.get(node.id, set())
        if zero_cols:
            keep_cols = [c for c in range(node.core_matrix.shape[1]) if c not in zero_cols]
            if keep_cols:
                node.core_matrix = node.core_matrix[:, keep_cols]
            else:
                # All columns pruned — keep a 1-column zero matrix to avoid empty
                node.core_matrix = node.core_matrix[:, :1] * 0.0

    # Phase 4: remove zeroed rows from core matrices and their input sources
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix
        n_axons = mat.shape[0]
        zero_rows_set = pruned_rows.get(node.id, set())
        keep_rows = [int(r) for r in range(n_axons) if r not in zero_rows_set]

        if len(keep_rows) < n_axons:
            flat_src = node.input_sources.flatten()
            if len(keep_rows) > 0:
                node.core_matrix = mat[keep_rows, :]
                node.input_sources = np.array(
                    [flat_src[r] for r in keep_rows], dtype=object
                )
            else:
                # All rows pruned — keep a 1-row zero matrix
                node.core_matrix = mat[:1, :] * 0.0
                node.input_sources = np.array([flat_src[0]], dtype=object)

    # Phase 5: set pruning maps on bank-backed cores from each weight bank
    # (banks are not compacted; masks let soft-core compaction drop rows/cols when materializing)
    # Uses same propagative helper as Phase 1.
    weight_banks = getattr(graph, "weight_banks", None) or {}
    init_bank = (initial_pruned_per_bank or {})
    for bank_id, bank in weight_banks.items():
        mat = bank.core_matrix
        n_axons, n_neurons = mat.shape
        init = init_bank.get(bank_id)
        if init is not None and len(init[0]) == n_axons and len(init[1]) == n_neurons:
            zero_rows, zero_cols = _masks_to_sets(init[0], init[1])
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=zero_threshold,
                initial_zero_rows=zero_rows,
                initial_zero_cols=zero_cols,
            )
        else:
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(mat, zero_threshold)
        pruned_row_mask = [i in zero_rows for i in range(n_axons)]
        pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        for node in graph.nodes:
            if isinstance(node, NeuralCore) and getattr(node, "weight_bank_id", None) == bank_id:
                node.pruned_row_mask = pruned_row_mask
                node.pruned_col_mask = pruned_col_mask

    return graph
