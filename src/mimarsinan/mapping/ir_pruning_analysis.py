"""Pure graph queries for IR pruning: segment split and I/O exemption.

No side-effects; used by ir_pruning for propagative pruning and segment-aware masks.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_segmentation import get_neural_segments


def compute_segment_io_exemption(
    ir_graph: IRGraph,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Compute per-node segment input-buffer rows and output-buffer columns by IR sources.

    Input-buffer row: axon row i is exempt only if input_sources[i] is graph input
    (IRSource node_id == -2). Rows fed by the previous segment or ComputeOp are not exempt.
    Output-buffer column: neuron column j is exempt only if (node_id, j) is in
    graph.output_sources (final graph output). Columns consumed by the next segment
    or ComputeOp are not exempt.

    Returns:
        (input_buffer_rows, output_buffer_cols): each dict maps node_id -> set of indices
        that must not be pruned for that node.
    """
    segments = get_neural_segments(ir_graph)
    input_buffer_rows: Dict[int, Set[int]] = {}
    output_buffer_cols: Dict[int, Set[int]] = {}

    for segment in segments:
        for node in segment:
            if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
                continue
            flat_src = node.input_sources.flatten()
            in_buf = set()
            for i, src in enumerate(flat_src):
                if isinstance(src, IRSource) and src.node_id == -2:
                    in_buf.add(i)
            if node.id not in input_buffer_rows:
                input_buffer_rows[node.id] = set()
            input_buffer_rows[node.id] |= in_buf

        for node in segment:
            if not isinstance(node, NeuralCore):
                continue
            out_buf = set()
            if ir_graph.output_sources.size:
                for src in ir_graph.output_sources.flatten():
                    if isinstance(src, IRSource) and src.node_id == node.id:
                        out_buf.add(src.index)
            if node.id not in output_buffer_cols:
                output_buffer_cols[node.id] = set()
            output_buffer_cols[node.id] |= out_buf

    return input_buffer_rows, output_buffer_cols
