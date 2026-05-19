"""IR graph neural-segment boundaries (shared by pruning and hybrid mapping)."""

from __future__ import annotations

from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore


def get_neural_segments(ir_graph: IRGraph) -> list[list[NeuralCore]]:
    """Split IR graph into neural segments between ComputeOp barriers."""
    segments: list[list[NeuralCore]] = []
    current: list[NeuralCore] = []
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
