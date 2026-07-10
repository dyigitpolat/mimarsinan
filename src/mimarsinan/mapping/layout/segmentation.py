"""Single source of truth for neural/host segmentation of a mapper graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Union

from mimarsinan.mapping.ir import ComputeOp, NeuralCore


@dataclass
class NeuralSegment:
    """A maximal run of consecutive ``NeuralCore`` nodes between host barriers."""

    nodes: List[NeuralCore]
    label: str


@dataclass
class HostSegment:
    """A single host ``ComputeOp`` barrier between neural segments."""

    compute_op: ComputeOp


Segment = Union[NeuralSegment, HostSegment]


def partition_ir_graph(ir_graph, *, per_hop: bool = False) -> List[Segment]:
    """Partition ``ir_graph.nodes`` into ordered neural / host segments: consecutive
    ``NeuralCore`` runs become ``NeuralSegment``s, each ``ComputeOp`` a ``HostSegment``
    barrier, with a trailing ``neural_segment_final`` run closing the graph.

    ``per_hop=True`` (C3 re-timing) further splits every ``NeuralSegment`` into
    one segment per intra-segment depth level: each hop boundary becomes a
    count-preserving decode/re-encode that resets arrival timing."""
    segments: List[Segment] = []
    current: List[NeuralCore] = []

    def _flush(label: str) -> None:
        seg = NeuralSegment(nodes=current, label=label)
        if per_hop:
            segments.extend(_split_segment_per_hop(seg))
        else:
            segments.append(seg)

    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current.append(node)
            continue
        if isinstance(node, ComputeOp):
            if current:
                _flush(f"neural_segment_until:{node.name}")
                current = []
            segments.append(HostSegment(compute_op=node))
            continue
        raise TypeError(
            f"Unknown IR node type in hybrid compilation: {type(node)}"
        )

    if current:
        _flush("neural_segment_final")
    return segments


def _split_segment_per_hop(segment: NeuralSegment) -> List[NeuralSegment]:
    """One ``NeuralSegment`` per intra-segment depth level, topological order.

    Coalescing / psum groups transfer membrane partial sums (NOT decodable
    counts), so a segment carrying any group stays whole."""
    if len(segment.nodes) <= 1:
        return [segment]
    if any(
        getattr(n, "coalescing_group_id", None) is not None
        or getattr(n, "psum_group_id", None) is not None
        for n in segment.nodes
    ):
        return [segment]

    ids = {node.id for node in segment.nodes}
    depth: Dict[int, int] = {}
    for node in segment.nodes:
        upstream = [
            depth[src.node_id]
            for src in node.input_sources.flatten()
            if getattr(src, "node_id", -1) in ids and src.node_id in depth
        ]
        depth[node.id] = (max(upstream) + 1) if upstream else 0

    groups: Dict[int, List[NeuralCore]] = {}
    for node in segment.nodes:
        groups.setdefault(depth[node.id], []).append(node)
    return [
        NeuralSegment(nodes=groups[d], label=f"{segment.label}_hop{d}")
        for d in sorted(groups)
    ]


def compute_node_latencies(
    node_input_node_ids: Dict[int, set],
    node_is_neural: Dict[int, bool],
) -> Dict[int, int]:
    """Latency = longest path of neural nodes feeding each node."""
    memo: Dict[int, int] = {}

    def _get(node_id: int) -> int:
        if node_id in memo:
            return memo[node_id]
        deps = node_input_node_ids.get(node_id)
        if not deps:
            memo[node_id] = 0
            return 0
        max_upstream = max(_get(d) for d in deps)
        result = max_upstream + (1 if node_is_neural.get(node_id, False) else 0)
        memo[node_id] = result
        return result

    for node_id in node_input_node_ids:
        _get(node_id)
    return memo


def compute_segment_ids(
    node_input_node_ids: Dict[int, set],
    node_is_neural: Dict[int, bool],
) -> Dict[int, int]:
    """Neural-segment id per node: increments across a host ComputeOp dependency."""
    memo: Dict[int, int] = {}

    def _get(node_id: int) -> int:
        if node_id in memo:
            return memo[node_id]
        deps = node_input_node_ids.get(node_id)
        is_neural = node_is_neural.get(node_id, False)
        if not deps:
            memo[node_id] = 0 if is_neural else -1
            return memo[node_id]
        upstream = [_get(d) for d in deps]
        if is_neural:
            has_compute_dep = any(
                not node_is_neural.get(d, False) for d in deps
            )
            memo[node_id] = max(upstream) + 1 if has_compute_dep else max(upstream)
        else:
            memo[node_id] = max(upstream)
        return memo[node_id]

    for node_id in node_input_node_ids:
        _get(node_id)
    return memo


def compute_host_side_segment_count(
    segment_ids: Dict[int, int],
    node_is_neural: Dict[int, bool],
) -> int:
    """Number of distinct host-side slots (sync barriers) in the layout."""
    host_segments = {
        int(segment_ids[node_id] + 1)
        for node_id, is_neural in node_is_neural.items()
        if not is_neural and node_id in segment_ids
    }
    return len(host_segments)
