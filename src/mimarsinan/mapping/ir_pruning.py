"""IR graph pruning: eliminate zeroed rows and columns from NeuralCores.

After the pruning adaptation step zeros out insignificant weights,
this post-processing pass physically removes the zeroed structure
from the IR graph:

1. Removes all-zero rows (axons) from core_matrix and input_sources.
2. Removes all-zero columns (neurons) from core_matrix.
3. Rewires downstream sources that referenced pruned neurons to off.
4. Reindexes remaining neuron references.

Before compacting, stores pre-pruning heatmap and row/col masks on each
NeuralCore for GUI visualization (soft-core pre/post pruning views).
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

    # Phase 1: identify pruned rows/columns with forward/backward propagation per core
    # (receiving axonal targets and producing neuronal sources have propagative effects)
    reindex_maps: Dict[int, Dict[int, int]] = {}
    pruned_neurons: Dict[int, Set[int]] = {}
    pruned_rows: Dict[int, Set[int]] = {}

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix  # (axons, neurons)
        n_axons, n_neurons = mat.shape

        zero_rows = set(
            i for i in range(n_axons)
            if np.abs(mat[i, :]).sum() < zero_threshold
        )
        zero_cols = set(
            j for j in range(n_neurons)
            if np.abs(mat[:, j]).sum() < zero_threshold
        )
        # Use a small epsilon for "has connection" so propagation can prune rows that only
        # feed below-threshold columns (which can still have tiny entries).
        conn_eps = min(1e-12, zero_threshold * 1e-4)
        changed = True
        while changed:
            changed = False
            for i in range(n_axons):
                if i in zero_rows:
                    continue
                non_zero_cols = set(
                    j for j in range(n_neurons)
                    if np.abs(mat[i, j]) >= conn_eps
                )
                if non_zero_cols and non_zero_cols <= zero_cols:
                    zero_rows.add(i)
                    changed = True
            for j in range(n_neurons):
                if j in zero_cols:
                    continue
                non_zero_rows = set(
                    i for i in range(n_axons)
                    if np.abs(mat[i, j]) >= conn_eps
                )
                if non_zero_rows and non_zero_rows <= zero_rows:
                    zero_cols.add(j)
                    changed = True

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
    # Use forward/backward propagation: pruned rows (axons) and pruned cols (neurons) have
    # propagative effects — a row that only feeds pruned cols is dead; a col that only
    # receives from pruned rows is dead. Iterate until fixpoint.
    weight_banks = getattr(graph, "weight_banks", None) or {}
    for bank_id, bank in weight_banks.items():
        mat = bank.core_matrix
        n_axons, n_neurons = mat.shape
        zero_rows: Set[int] = set(
            i for i in range(n_axons)
            if np.abs(mat[i, :]).sum() < zero_threshold
        )
        zero_cols: Set[int] = set(
            j for j in range(n_neurons)
            if np.abs(mat[:, j]).sum() < zero_threshold
        )
        conn_eps = min(1e-12, zero_threshold * 1e-4)
        changed = True
        while changed:
            changed = False
            for i in range(n_axons):
                if i in zero_rows:
                    continue
                non_zero_cols = set(
                    j for j in range(n_neurons)
                    if np.abs(mat[i, j]) >= conn_eps
                )
                if non_zero_cols and non_zero_cols <= zero_cols:
                    zero_rows.add(i)
                    changed = True
            for j in range(n_neurons):
                if j in zero_cols:
                    continue
                non_zero_rows = set(
                    i for i in range(n_axons)
                    if np.abs(mat[i, j]) >= conn_eps
                )
                if non_zero_rows and non_zero_rows <= zero_rows:
                    zero_cols.add(j)
                    changed = True
        pruned_row_mask = [i in zero_rows for i in range(n_axons)]
        pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        for node in graph.nodes:
            if isinstance(node, NeuralCore) and getattr(node, "weight_bank_id", None) == bank_id:
                node.pruned_row_mask = pruned_row_mask
                node.pruned_col_mask = pruned_col_mask

    return graph
