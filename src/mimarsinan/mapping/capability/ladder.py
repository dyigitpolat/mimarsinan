"""The capability ladder C0-C6 and its placement rule on the toolchain IR (a requirement placement, no lowering)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.op_kinds import stationary_operand
from mimarsinan.mapping.capability.requirements import CROSSBAR, LANE, WIRING, requirement_of
from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRNode, NeuralCore
from mimarsinan.mapping.ir.source import encode_ir_sources

#: Printed on every C1-C6 row (lens_ae.md Q1): above C0 the census places requirements, nothing is lowered yet.
PLACEMENT_LABEL = "requirement placement on the toolchain IR (no lowering yet)"

CHIP, HOST, TRANSPARENT = "chip", "host", "transparent"
INPUT_NODE, OFF_NODE, ALWAYS_ON_NODE = -2, -1, -3

#: A dynamic product's payload is written from an on-chip output stage, or by a host op before it (the design
#: note's ``bank_program``); scout_network.md §3.3 reports D both ways.
ON_CHIP_WRITE_BACK, HOST_PAYLOAD_WRITE = "on_chip_write_back", "host_payload_write"
PAYLOAD_WRITE_KIND = "payload_write"


class UnknownCapabilityLevelError(ToolchainRefusal, KeyError):
    """A capability level the ladder does not declare."""


@dataclass(frozen=True)
class CapabilityLevel:
    name: str
    classes: frozenset
    title: str


def _level(index: int, classes: str, title: str) -> CapabilityLevel:
    return CapabilityLevel(f"C{index}", frozenset(classes), title)


#: scout_network.md:197-199, titles as the oracle prints them.
LADDER: Tuple[CapabilityLevel, ...] = (
    _level(0, "", "C0 today (MVM contract)"),
    _level(1, "LAEPCW", "C1 = C0 + L A E P C W (arith toggle)"),
    _level(2, "LAEPCWGS", "C2 = C1 + G S (grouped conv, static-linear lowerings)"),
    _level(3, "LAEPCWGSD", "C3 = C2 + D (dynamic-operand MM)"),
    _level(4, "LAEPCWGSDX", "C4 = C3 + X (operand product)"),
    _level(5, "LAEPCWGSDXT", "C5 = C4 + T (table law)"),
    _level(6, "LAEPCWGSDXTR", "C6 = C5 + R (row reductions: softmax, norms)"),
)


def capability_level(name: str) -> CapabilityLevel:
    for level in LADDER:
        if level.name == name:
            return level
    raise UnknownCapabilityLevelError(f"capability level {name!r} is not on the ladder {[known.name for known in LADDER]}")


def payload_writes(level: CapabilityLevel) -> Tuple[Optional[str], ...]:
    """The payload-write variants a level reports: both where D is declared, none elsewhere."""
    return (ON_CHIP_WRITE_BACK, HOST_PAYLOAD_WRITE) if "D" in level.classes else (None,)


@dataclass(frozen=True)
class CensusNode:
    """One placed node in execution order; ``ir_node`` is None for a payload write the placement inserts."""

    id: int
    kind: str
    side: str
    source_ids: np.ndarray
    source_indices: np.ndarray
    output_count: int
    ir_node: Optional[IRNode]


def _columns(sources) -> Tuple[np.ndarray, np.ndarray]:
    """The columnar int32 wire format (``encode_ir_sources``): a quarter of the object array's memory."""
    encoded = encode_ir_sources(np.asarray(sources, dtype=object))
    if not isinstance(encoded, tuple):
        raise TypeError("IR input sources must be IRSource arrays")
    return encoded[2], encoded[3]


def _live(ids: np.ndarray, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    keep = (ids != OFF_NODE) & (ids != ALWAYS_ON_NODE)
    return (ids, indices) if bool(keep.all()) else (ids[keep], indices[keep])


def live_source_columns(sources) -> Tuple[np.ndarray, np.ndarray]:
    """``(producer ids, element indices)`` of the sources that carry a value (off and always-on dropped)."""
    return _live(*_columns(sources))


def _output_count(node: IRNode) -> int:
    if isinstance(node, NeuralCore):
        return int(node.get_output_count())
    if isinstance(node, ComputeOp):
        count = 1
        for dim in node.output_shape or ():
            count *= int(dim)
        return count
    raise TypeError(f"IR node {node.name!r} is neither a NeuralCore nor a ComputeOp")


def _producers_on_chip(ids: np.ndarray, side: Mapping[int, str]) -> bool:
    producers = np.unique(ids)
    if producers.size == 0:
        return False
    for producer in producers.tolist():
        if producer != INPUT_NODE and producer not in side:
            raise ValueError(f"IR node order is not topological: producer {producer} is placed after its consumer")
        if side.get(producer) != CHIP:
            return False
    return True


def _side(kind: str, ids: np.ndarray, side: Mapping[int, str], level: CapabilityLevel) -> str:
    requirement = requirement_of(kind)
    if not requirement.met_by(level.classes):
        return HOST
    if requirement.placement == CROSSBAR:
        return CHIP
    on_chip = _producers_on_chip(ids, side)
    if requirement.placement == LANE:
        return CHIP if on_chip else HOST
    if requirement.placement == WIRING:
        return CHIP if on_chip else TRANSPARENT
    return HOST


def source_columns(ir_graph: IRGraph) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """``{node id: (producer ids, element indices)}``, encoded once and shared by every level's placement."""
    return {node.id: _columns(node.input_sources.flatten()) for node in ir_graph.nodes}


def place_level(
    ir_graph: IRGraph, level: CapabilityLevel, kinds: Mapping[int, str], *,
    payload_write: Optional[str] = ON_CHIP_WRITE_BACK,
    columns: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[CensusNode, ...]:
    """Place every IR node at ``level`` in IR order: a requirement subset of the declared set, every producer on chip.

    Under ``HOST_PAYLOAD_WRITE`` each on-chip dynamic product reads its stationary operand from a host
    payload-write node inserted just before it.
    """
    columns = columns if columns is not None else source_columns(ir_graph)
    side: Dict[int, str] = {}
    placed = []
    next_id = 1 + max((node.id for node in ir_graph.nodes), default=0)
    for node in ir_graph.nodes:
        ids, indices = columns[node.id]
        kind = kinds[node.id]
        side[node.id] = _side(kind, _live(ids, indices)[0], side, level)
        if (payload_write == HOST_PAYLOAD_WRITE and side[node.id] == CHIP and isinstance(node, ComputeOp)
                and "D" in requirement_of(kind).classes):
            offset, size, _ = stationary_operand(node)
            write = CensusNode(next_id, PAYLOAD_WRITE_KIND, HOST, *_live(ids[offset:offset + size],
                                                                         indices[offset:offset + size]), size, None)
            placed.append(write)
            ids, indices = ids.copy(), indices.copy()
            ids[offset:offset + size] = next_id
            indices[offset:offset + size] = np.arange(size)
            next_id += 1
        placed.append(CensusNode(node.id, kind, side[node.id], *_live(ids, indices), _output_count(node), node))
    return tuple(placed)
