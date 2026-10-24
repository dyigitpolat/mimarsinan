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


def segment_depth_groups(nodes: List[NeuralCore]) -> List[List[NeuralCore]] | None:
    """Intra-segment depth levels in topological order, or ``None`` when the
    segment must stay whole (a single node, or psum groups — those transfer
    membrane partial sums, NOT decodable counts).

    [H0] A coalescing group levels as one unit: its cores are whole-fan-in
    neurons (the fan-in is fused on the chip, one dot product per neuron), so
    each emits a decodable count, and every member takes the group's depth (the
    deepest member's). A windowed segment holding one is re-timed per hop like
    any other, as its identity program (no coalescing) is."""
    if len(nodes) <= 1:
        return None
    if any(getattr(n, "psum_group_id", None) is not None for n in nodes):
        return None

    depth = _unit_depths(nodes)
    if depth is None:
        return None

    groups: Dict[int, List[NeuralCore]] = {}
    for node in nodes:
        groups.setdefault(depth[node.id], []).append(node)
    # Depths are contiguous from 0 by construction (each depth d>0 requires an
    # upstream at d-1), so list position == depth level.
    return [groups[d] for d in sorted(groups)]


def _unit_depths(nodes: List[NeuralCore]) -> Dict[int, int] | None:
    """Per node, its unit's longest-path depth over the segment's edges between units (a coalescing group is one
    unit, every other node its own); None when the units' graph has a cycle (the segment then stays whole)."""
    ids = {node.id for node in nodes}
    unit_of: Dict[int, object] = {
        node.id: ("coalescing", group) if (group := getattr(node, "coalescing_group_id", None)) is not None
        else ("node", node.id)
        for node in nodes
    }
    reads: Dict[object, set] = {unit: set() for unit in unit_of.values()}
    for node in nodes:
        for src in node.input_sources.flatten():
            producer = getattr(src, "node_id", -1)
            if producer in ids and unit_of[producer] != unit_of[node.id]:
                reads[unit_of[node.id]].add(unit_of[producer])
    unit_depth: Dict[object, int] = {}
    pending = set(reads)
    while pending:
        ready = [u for u in pending if reads[u] <= unit_depth.keys()]
        if not ready:
            return None
        for unit in ready:
            unit_depth[unit] = max((unit_depth[p] + 1 for p in reads[unit]), default=0)
        pending.difference_update(ready)
    return {node.id: unit_depth[unit_of[node.id]] for node in nodes}


def _split_segment_per_hop(segment: NeuralSegment) -> List[NeuralSegment]:
    """One ``NeuralSegment`` per intra-segment depth level, topological order."""
    groups = segment_depth_groups(segment.nodes)
    if groups is None:
        return [segment]
    return [
        NeuralSegment(nodes=group, label=f"{segment.label}_hop{d}")
        for d, group in enumerate(groups)
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
