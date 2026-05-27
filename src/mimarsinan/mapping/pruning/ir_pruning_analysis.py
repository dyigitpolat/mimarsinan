"""Pure graph queries used by IR pruning.

Currently exposes :func:`compute_graph_io_exemption`, which returns the
per-node row/column indices that must never be pruned because they carry
**model-level** inputs or outputs (model input data axons and model output
logits). The helper for splitting the graph into neural segments
(``get_neural_segments``) is re-exported from
:mod:`mimarsinan.mapping.pruning.ir_segmentation` for convenience.
"""

from __future__ import annotations

from typing import Dict, Set, Tuple

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning.ir_segmentation import get_neural_segments

__all__ = ["compute_graph_io_exemption", "get_neural_segments"]


def compute_graph_io_exemption(
    ir_graph: IRGraph,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Per-node row/column indices that carry model-level I/O.

    Pruning runs on the unified pre-segmentation IR, so "graph I/O" is
    exactly the model-level boundary:

    - **Input-data rows**: axon row ``i`` is exempt iff
      ``node.input_sources[i]`` is ``IRSource(node_id=-2, ...)`` --- the
      sentinel for top-level model input data. Bias axons (``node_id=-3``),
      always-on (``-1``) and intra-graph sources are *not* exempt here.
    - **Output-logit columns**: neuron column ``j`` is exempt iff
      ``(node.id, j)`` appears in ``ir_graph.output_sources`` --- the
      sentinel for top-level model output logits.

    Returns ``(input_data_rows, output_logit_cols)``: each maps
    ``node_id -> set(indices)`` to keep alive during pruning propagation.
    """
    input_data_rows: Dict[int, Set[int]] = {}
    output_logit_cols: Dict[int, Set[int]] = {}

    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
            continue
        in_buf: Set[int] = set()
        for i, src in enumerate(node.input_sources.flatten()):
            if isinstance(src, IRSource) and src.node_id == -2:
                in_buf.add(i)
        input_data_rows[node.id] = in_buf

        out_buf: Set[int] = set()
        if ir_graph.output_sources.size:
            for src in ir_graph.output_sources.flatten():
                if isinstance(src, IRSource) and src.node_id == node.id:
                    out_buf.add(src.index)
        output_logit_cols[node.id] = out_buf

    return input_data_rows, output_logit_cols
