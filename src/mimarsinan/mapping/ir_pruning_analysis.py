"""Pure graph queries for IR pruning: segment split and I/O exemption.

No side-effects; used by ir_pruning for propagative pruning and segment-aware masks.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore


def get_neural_segments(ir_graph: IRGraph) -> List[List[NeuralCore]]:
    """Split IR graph into neural segments (same boundaries as hybrid hard core mapping).

    Segments are contiguous NeuralCore lists between ComputeOp nodes.
    Returns list of segments, each segment a list of NeuralCore nodes.
    """
    segments: List[List[NeuralCore]] = []
    current: List[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current.append(node)
            continue
        if isinstance(node, ComputeOp):
            if current:
                segments.append(current)
                current = []
            continue
        raise TypeError(f"Unknown IR node type in segment split: {type(node)}")
    if current:
        segments.append(current)
    return segments


def compute_segment_io_exemption(
    ir_graph: IRGraph,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Compute per-node segment input-buffer rows and output-buffer columns by IR sources.

    Input-buffer row: axon row i is exempt if input_sources[i] is segment input
    (IRSource node_id == -2 or node_id not in this segment).
    Output-buffer column: neuron column j is exempt if (node_id, j) feeds segment output
    (in graph.output_sources or consumed by a node outside this segment).

    Returns:
        (input_buffer_rows, output_buffer_cols): each dict maps node_id -> set of indices
        that must not be pruned for that node.
    """
    segments = get_neural_segments(ir_graph)
    consumed_by: Dict[int, Set[int]] = defaultdict(set)
    for node in ir_graph.nodes:
        if not hasattr(node, "input_sources"):
            continue
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                consumed_by[src.node_id].add(node.id)
    for src in ir_graph.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            consumed_by[src.node_id].add(-999)

    input_buffer_rows: Dict[int, Set[int]] = {}
    output_buffer_cols: Dict[int, Set[int]] = {}

    for segment in segments:
        segment_ids = {n.id for n in segment}
        for node in segment:
            if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
                continue
            flat_src = node.input_sources.flatten()
            in_buf = set()
            for i, src in enumerate(flat_src):
                if isinstance(src, IRSource):
                    if src.node_id == -2:
                        in_buf.add(i)
                    elif src.node_id >= 0 and src.node_id not in segment_ids:
                        in_buf.add(i)
            if node.id not in input_buffer_rows:
                input_buffer_rows[node.id] = set()
            input_buffer_rows[node.id] |= in_buf

        for node in segment:
            if not isinstance(node, NeuralCore):
                continue
            n_neurons = node.core_matrix.shape[1] if node.core_matrix is not None else None
            if n_neurons is None and getattr(node, "weight_row_slice", None) is not None:
                start, end = node.weight_row_slice
                n_neurons = end - start
            if n_neurons is None:
                continue
            out_buf = set()
            consumers = consumed_by.get(node.id, set())
            has_external_consumer = any(c not in segment_ids for c in consumers if c != -999)
            for j in range(n_neurons):
                is_output = False
                if ir_graph.output_sources.size:
                    for src in ir_graph.output_sources.flatten():
                        if isinstance(src, IRSource) and src.node_id == node.id and src.index == j:
                            is_output = True
                            break
                if is_output or has_external_consumer:
                    out_buf.add(j)
            if node.id not in output_buffer_cols:
                output_buffer_cols[node.id] = set()
            output_buffer_cols[node.id] |= out_buf

    return input_buffer_rows, output_buffer_cols
