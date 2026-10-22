"""Per-level requirement census of an IR: host ops, segments, boundary bytes, MAC split, lane work, dynamic writes."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.mapping.capability.boundary import BoundaryElements, boundary_elements
from mimarsinan.mapping.capability.ladder import (
    CHIP,
    HOST,
    INPUT_NODE,
    LADDER,
    ON_CHIP_WRITE_BACK,
    PLACEMENT_LABEL,
    TRANSPARENT,
    CapabilityLevel,
    CensusNode,
    live_source_columns,
    payload_writes,
    place_level,
    source_columns,
)
from mimarsinan.mapping.capability.op_kinds import ir_op_kinds, stationary_operand
from mimarsinan.mapping.capability.requirements import CROSSBAR, INPUT_ELEMENTS, OUTPUT_ELEMENTS, requirement_of
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.layout.segmentation import (
    compute_host_side_segment_count,
    compute_node_latencies,
    compute_segment_ids,
)
from mimarsinan.mapping.support.schedule.pass_carry import boundary_transfer_bytes
from mimarsinan.mapping.verification.onchip_majority import count_host_ops


@dataclass(frozen=True)
class SegmentCensus:
    """One in-order neural segment (one resident pass of the unbounded-core identity program)."""

    cores: int
    cells_committed: int
    written_cells_producer: int
    latency_steps: int


@dataclass(frozen=True)
class LevelCensus:
    """The census of one member's IR at one capability level (and payload-write variant where D is declared)."""

    level: str
    payload_write: Optional[str]
    label: Optional[str]
    value_bits: int
    host_ops: int
    host_kinds: Dict[str, int]
    neural_segments: int
    host_segments: int
    dependency_neural_segments: int
    dependency_host_segments: int
    boundary: BoundaryElements
    boundary_bytes: int
    host_macs: int
    host_macs_by_kind: Dict[str, int]
    stated_boundaries: Dict[str, int]
    onchip_op_macs: int
    neural_cores: int
    cells_committed: int
    #: Core cells fed by a value (no bias row, no padding axon): a floor under the logical on-chip MACs.
    core_mac_floor: int
    #: Per kind, the op types of MAC-carrying ops the MAC SSOT counts 0 (no rule yet), on each side.
    uncounted_host_products: Dict[str, Tuple[str, ...]]
    uncounted_chip_products: Dict[str, Tuple[str, ...]]
    dynamic_payload_elements: int
    dynamic_payload_bytes: int
    lane_ops: int
    element_ops_by_kind: Dict[str, int]
    latency_steps: int
    segments: Tuple[SegmentCensus, ...]


def value_bytes(elements: int, value_bits: int) -> int:
    """Bytes of ``elements`` values crossing at ``value_bits`` (``pass_carry.boundary_transfer_bytes``, per element)."""
    return boundary_transfer_bytes(width=int(elements), timesteps=int(value_bits))


def _ops_graph(nodes: Sequence[CensusNode]) -> IRGraph:
    ops = [n.ir_node for n in nodes if isinstance(n.ir_node, ComputeOp)]
    return IRGraph(nodes=list(ops), output_sources=np.zeros(0, dtype=object))


def _host_macs_by_kind(host: Sequence[CensusNode]) -> Dict[str, int]:
    kinds = sorted({n.kind for n in host})
    return {kind: count_host_ops(_ops_graph([n for n in host if n.kind == kind])) for kind in kinds}


def _stated_boundaries(by_kind: Mapping[str, int]) -> Dict[str, int]:
    """Host MACs behind each stated boundary of a host-placed kind (the requirement table names the boundary)."""
    stated: Dict[str, int] = {}
    for kind, macs in by_kind.items():
        boundary = requirement_of(kind).host_boundary
        if boundary:
            stated[boundary] = stated.get(boundary, 0) + macs
    return stated


def _work(node: CensusNode) -> int:
    rule = requirement_of(node.kind).work
    if rule == OUTPUT_ELEMENTS:
        return node.output_count
    if rule == INPUT_ELEMENTS:
        return int(node.source_ids.size)
    return 0


def _cells(node: CensusNode) -> int:
    core = node.ir_node
    return int(core.get_input_count()) * int(core.get_output_count()) if isinstance(core, NeuralCore) else 0


def _valued_cells(node: CensusNode) -> int:
    """A core's cells whose axon carries a value (placement already dropped off and always-on sources)."""
    return int(node.source_ids.size) * node.output_count if isinstance(node.ir_node, NeuralCore) else 0


def _uncounted_products(nodes: Sequence[CensusNode]) -> Dict[str, Tuple[str, ...]]:
    """Op types per kind of the MAC-carrying ComputeOps that the MAC SSOT (``count_host_ops``) counts 0."""
    found: Dict[str, set] = {}
    for node in nodes:
        op = node.ir_node
        if (isinstance(op, ComputeOp) and requirement_of(node.kind).carries_macs
                and count_host_ops(_ops_graph([node])) == 0):
            found.setdefault(node.kind, set()).add(str(op.op_type))
    return {kind: tuple(sorted(types)) for kind, types in sorted(found.items())}


def _payload_elements(node: CensusNode) -> int:
    """Cells an on-chip dynamic product writes per inference: its stationary operand."""
    op = node.ir_node
    if node.side != CHIP or not isinstance(op, ComputeOp) or "D" not in requirement_of(node.kind).classes:
        return 0
    _, size, _ = stationary_operand(op)
    return size


def _resolved_dependencies(nodes: Sequence[CensusNode]) -> Dict[int, set]:
    """Producer node ids per non-transparent node, wiring passed through, the network input left out."""
    through: Dict[int, set] = {}
    deps: Dict[int, set] = {}
    for node in nodes:
        found = set()
        for producer in np.unique(node.source_ids).tolist():
            if producer == INPUT_NODE:
                continue
            found |= through.get(producer, {producer})
        if node.side == TRANSPARENT:
            through[node.id] = found
        else:
            deps[node.id] = found
    return deps


def _in_order_segments(nodes: Sequence[CensusNode]) -> List[List[CensusNode]]:
    """``partition_ir_graph``'s rule over placed nodes: chip runs between host ops, each host op its own segment."""
    runs: List[List[CensusNode]] = []
    current: List[CensusNode] = []
    for node in nodes:
        if node.side == CHIP:
            current.append(node)
        elif node.side == HOST:
            if current:
                runs.append(current)
            current = []
            runs.append([node])
    if current:
        runs.append(current)
    return runs


def _segment_census(run: Sequence[CensusNode], deps: Mapping[int, set]) -> SegmentCensus:
    members = {node.id for node in run}
    crossbar = {node.id: requirement_of(node.kind).placement == CROSSBAR for node in run}
    # Execution order keeps the SSOT's memoized walk one frame deep per node, whatever the chain length.
    latencies = compute_node_latencies({node.id: deps[node.id] & members for node in run}, crossbar)
    steps = 1 + max((latencies[i] for i, is_crossbar in crossbar.items() if is_crossbar), default=0)
    return SegmentCensus(
        cores=sum(isinstance(node.ir_node, NeuralCore) for node in run),
        cells_committed=sum(_cells(node) for node in run),
        written_cells_producer=sum(_payload_elements(node) for node in run),
        latency_steps=steps,
    )


def level_census(
    ir_graph: IRGraph, level: CapabilityLevel, *, value_bits: int,
    payload_write: Optional[str] = ON_CHIP_WRITE_BACK, kinds: Optional[Mapping[int, str]] = None,
    columns: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> LevelCensus:
    """Place ``ir_graph`` at ``level`` and count; MACs come from the MAC SSOT (``count_host_ops``) only."""
    if "D" not in level.classes:
        payload_write = None
    nodes = place_level(ir_graph, level, kinds if kinds is not None else ir_op_kinds(ir_graph),
                        payload_write=payload_write, columns=columns)
    host = [n for n in nodes if n.side == HOST]
    chip = [n for n in nodes if n.side == CHIP]
    runs = _in_order_segments(nodes)
    deps = _resolved_dependencies(nodes)
    is_chip = {n.id: n.side == CHIP for n in nodes if n.side != TRANSPARENT}
    segment_ids = compute_segment_ids(deps, is_chip)
    neural_runs = [run for run in runs if run[0].side == CHIP]
    segments = tuple(_segment_census(run, deps) for run in neural_runs)
    boundary = boundary_elements(nodes, live_source_columns(ir_graph.output_sources.flatten()))
    lane = [n for n in chip if requirement_of(n.kind).is_lane]
    element_ops = Counter()
    for node in chip:
        if requirement_of(node.kind).placement != CROSSBAR and _work(node):
            element_ops[node.kind] += _work(node)
    payload = sum(s.written_cells_producer for s in segments)
    by_kind = _host_macs_by_kind(host)
    return LevelCensus(
        level=level.name, payload_write=payload_write, label=None if not level.classes else PLACEMENT_LABEL,
        value_bits=int(value_bits), host_ops=len(host), host_kinds=dict(Counter(n.kind for n in host)),
        neural_segments=len(neural_runs), host_segments=len(runs) - len(neural_runs),
        dependency_neural_segments=len({segment_ids[n.id] for n in chip}),
        dependency_host_segments=compute_host_side_segment_count(segment_ids, is_chip),
        boundary=boundary, boundary_bytes=value_bytes(boundary.total, value_bits),
        host_macs=count_host_ops(_ops_graph(host)), host_macs_by_kind=by_kind,
        stated_boundaries=_stated_boundaries(by_kind), onchip_op_macs=count_host_ops(_ops_graph(chip)),
        neural_cores=sum(s.cores for s in segments), cells_committed=sum(s.cells_committed for s in segments),
        core_mac_floor=sum(_valued_cells(n) for n in chip),
        uncounted_host_products=_uncounted_products(host), uncounted_chip_products=_uncounted_products(chip),
        dynamic_payload_elements=payload, dynamic_payload_bytes=value_bytes(payload, value_bits),
        lane_ops=sum(_work(n) for n in lane), element_ops_by_kind=dict(element_ops),
        latency_steps=sum(s.latency_steps for s in segments), segments=segments,
    )


def ladder_census(
    ir_graph: IRGraph, *, value_bits: int, levels: Sequence[CapabilityLevel] = LADDER,
    kinds: Optional[Mapping[int, str]] = None,
) -> Tuple[LevelCensus, ...]:
    """One row per level, two at a level that declares D (on-chip write-back first)."""
    kinds = kinds if kinds is not None else ir_op_kinds(ir_graph)
    columns = source_columns(ir_graph)
    return tuple(
        level_census(ir_graph, level, value_bits=value_bits, payload_write=write, kinds=kinds, columns=columns)
        for level in levels for write in payload_writes(level)
    )
