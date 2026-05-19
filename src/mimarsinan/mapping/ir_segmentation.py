"""IR graph neural-segment boundaries (shared by pruning and hybrid mapping)."""

from __future__ import annotations

from collections import defaultdict

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore

# Must match hybrid_hardcore_mapping._FINAL_OUTPUT_SENTINEL
_FINAL_OUTPUT_SENTINEL = -999


def build_ir_consumed_by(ir_graph: IRGraph) -> dict[int, set[int]]:
    """Map each IR node id to the set of consumer node ids (or final-output sentinel)."""
    consumed_by: dict[int, set[int]] = defaultdict(set)
    for node in ir_graph.nodes:
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                consumed_by[src.node_id].add(node.id)
    for src in ir_graph.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            consumed_by[src.node_id].add(_FINAL_OUTPUT_SENTINEL)
    return consumed_by


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
