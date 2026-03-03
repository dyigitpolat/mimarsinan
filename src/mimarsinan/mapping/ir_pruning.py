"""IR graph pruning: eliminate zeroed rows and columns from NeuralCores.

After the pruning adaptation step zeros out insignificant weights,
this post-processing pass physically removes the zeroed structure
from the IR graph:

1. Removes all-zero rows (axons) from core_matrix and input_sources.
2. Removes all-zero columns (neurons) from core_matrix.
3. Rewires downstream sources that referenced pruned neurons to off.
4. Reindexes remaining neuron references.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Set, Tuple

import numpy as np

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore


def prune_ir_graph(ir_graph: IRGraph, zero_threshold: float = 1e-8) -> IRGraph:
    """Remove zeroed rows and columns from all NeuralCores in the IR graph.

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

    # Phase 1: identify pruned columns per core and build reindex maps
    # Maps: node_id -> {old_neuron_idx -> new_neuron_idx}  (for kept neurons)
    # Maps: node_id -> set of pruned neuron indices
    reindex_maps: Dict[int, Dict[int, int]] = {}
    pruned_neurons: Dict[int, Set[int]] = {}

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix  # (axons, neurons)
        n_neurons = mat.shape[1]

        # Find all-zero columns
        col_abs = np.abs(mat).sum(axis=0)
        zero_cols = set(int(i) for i in range(n_neurons) if col_abs[i] < zero_threshold)

        pruned_neurons[node.id] = zero_cols

        # Build reindex map: old index -> new index for kept columns
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
        row_abs = np.abs(mat).sum(axis=1)
        keep_rows = [int(r) for r in range(n_axons) if row_abs[r] >= zero_threshold]

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

    return graph
