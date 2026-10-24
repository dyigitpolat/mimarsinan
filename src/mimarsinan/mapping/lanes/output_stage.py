"""The output stage of a priced point (W08): lane work, host boundary bytes and host round trips, labelled.

Read from G7's census SSOT (``mapping/capability/census.py``): a deployed program is counted at C0 (its cores on
chip, its remaining host ops on the host), so a lane program's absorbed ops are on chip by construction and its
lane work is the fusion's (``executed``). A program on a lane chip whose lanes do not run (a float value program)
is counted as deployed and, beside it, at the declared classes as a requirement placement (``estimated``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from mimarsinan.mapping.capability.census import LevelCensus, level_census
from mimarsinan.mapping.capability.ladder import LADDER, CapabilityLevel
from mimarsinan.mapping.ir.lane_law import operand_axons
from mimarsinan.mapping.lanes.declaration import LaneDeclaration, lane_declaration, lanes_executed
from mimarsinan.mapping.lanes.forms import pool_form
from mimarsinan.mapping.lanes.ir.fusion import LANE_REPORT
from mimarsinan.mapping.lanes.plan import plan_lanes

EXECUTED, ESTIMATED = "executed", "estimated"
#: [W08] The fused adds' operand rows: priced as the lane's operand port, not as crossbar cells (the census subtracts
#: them); the packed program still places them on crossbar axons, which the core count and footprint include.
OPERAND_PORT_NOTE = ("operand rows: {rows:,} zero-weight crossbar rows of fused adds are priced as the lanes' operand "
                     "port (no cell, MAC or programming byte); the packed program still places them on crossbar axons, "
                     "so its core count and resident footprint include them (array overhead, not priced apart)")
#: Printed beside a placement (G7's label, the lanes not executed by the deployed program).
PLACED_LABEL = "estimated: requirement placement on the toolchain IR, lanes not executed"


@dataclass(frozen=True)
class StageCensus:
    """One placement's output stage per inference."""

    lane_ops: Dict[str, int]
    host_ops: int
    host_round_trips: int
    host_barriers: int
    neural_segments: int
    boundary_elements: Dict[str, int]
    host_bytes: int

    @property
    def lane_ops_total(self) -> int:
        return int(sum(self.lane_ops.values()))


@dataclass(frozen=True)
class OutputStage:
    """A point's output stage: its label, the declared lane block, the deployed census and (lanes not executed) the
    placement at the declared classes."""

    label: str
    classes: Tuple[str, ...]
    lanes_per_core: Optional[int]
    lane_block: Optional[str]
    deployed: StageCensus
    placed: Optional[StageCensus] = None
    notes: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "OutputStage":
        def census(value):
            return None if value is None else StageCensus(**dict(value))

        data = dict(data)
        return cls(label=str(data["label"]), classes=tuple(data["classes"]), lanes_per_core=data["lanes_per_core"],
                   lane_block=data["lane_block"], deployed=census(data["deployed"]),  # type: ignore[arg-type]
                   placed=census(data.get("placed")), notes=tuple(data.get("notes", ())))


def stage_census(row: LevelCensus, lane_ops: Optional[Mapping[str, int]] = None) -> StageCensus:
    """A census row's output stage; ``lane_ops`` replaces the row's placed lane work by an executed count."""
    work = dict(lane_ops) if lane_ops is not None else {
        kind: int(n) for kind, n in sorted(row.element_ops_by_kind.items())}
    boundary = {"entry": row.boundary.entry, "carry_in": row.boundary.carry_in, "carry_out": row.boundary.carry_out,
                "readout": row.boundary.readout}
    return StageCensus(lane_ops=work, host_ops=row.host_ops, host_round_trips=row.host_segments,
                       host_barriers=row.dependency_host_segments, neural_segments=row.neural_segments,
                       boundary_elements=boundary, host_bytes=row.boundary_bytes)


def declared_level(declaration: LaneDeclaration) -> CapabilityLevel:
    return CapabilityLevel("declared", frozenset(declaration.classes),
                           f"declared lanes {''.join(sorted(declaration.classes))}")


def output_stage_of_program(ir_graph, platform: Mapping[str, Any]) -> Optional[OutputStage]:
    """The deployed program's output stage, or ``None`` when the chip declares no ``onchip_arithmetic``."""
    declaration = lane_declaration(platform)
    if declaration is None:
        return None
    bits = int(platform.get("activation_bits") or 8)
    deployed_row = level_census(ir_graph, LADDER[0], value_bits=bits)
    report = getattr(ir_graph, LANE_REPORT, None)
    executed = lanes_executed(platform) is not None and report is not None
    deployed = stage_census(deployed_row, dict(report["lane_ops"]) if executed and report else {})
    placed, notes = None, []
    rows = sum(operand_axons(n) for n in ir_graph.nodes) if executed else 0
    if rows:  # every IR core's operand axons (a bank position core holds its bank's whole operand block)
        notes.append(OPERAND_PORT_NOTE.format(rows=rows))
    if not executed and declaration.any:
        placed = stage_census(level_census(ir_graph, declared_level(declaration), value_bits=bits))
        notes.append("lanes declared, not executed: the program is on the float value path (no readout_law)")
    if platform.get("activation_bits") is None:
        notes.append("host bytes at 8 bits per element: the program declares no activation_bits")
    return OutputStage(label=EXECUTED if executed else (ESTIMATED if declaration.any else "no lanes"),
                       classes=tuple(sorted(declaration.classes)), lanes_per_core=declaration.lanes_per_core,
                       lane_block=declaration.lane_block, deployed=deployed, placed=placed, notes=tuple(notes))


def declared_lanes_per_tile(platform: Mapping[str, Any]) -> Optional[int]:
    """Lanes per floorplan tile (the G3 lane rows' factor): lanes per core times the resolved cores per tile;
    ``None`` without a declared lane count, ``0`` on a chip declaring no lane class."""
    declaration = lane_declaration(platform)
    if declaration is None:
        return None
    if not declaration.any:
        return 0
    if declaration.lanes_per_core is None:
        return None
    return int(declaration.lanes_per_core) * int(platform.get("cores_per_tile_resolved") or 1)


def _elements(shape) -> int:
    return int(math.prod(int(d) for d in shape)) if shape else 0


def _pool_outputs(node) -> int:
    """The output elements of a pool lane's op: its traced output, else its window over its traced input."""
    if node.output_shape is not None:
        return _elements(node.output_shape)
    shapes = node.input_shapes
    form = pool_form(node.module, tuple(shapes[0]) if shapes and shapes[0] is not None else None)
    if form is None:
        return 0
    channels, height, width = (int(d) for d in shapes[0])
    rows = (height + 2 * form.padding[0] - form.kernel[0]) // form.stride[0] + 1
    cols = (width + 2 * form.padding[1] - form.kernel[1]) // form.stride[1] + 1
    return channels * rows * cols


def planned_lane_ops(model_repr, platform: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    """The lane work per inference the declared lanes run on ``model_repr``, by kind and counted as the IR fusion
    counts it (``fusion.LaneFusion.lane_ops``: a clamp's and an add's output elements, a pool's requantized outputs)
    and read off the lane plan on the mapper graph and its traced shapes; zeros where the lanes do not run (no
    ``readout_law``), ``None`` on a chip declaring no ``onchip_arithmetic``. The candidate plane's estimate of the
    work a record measures."""
    declaration = lane_declaration(platform)
    if declaration is None:
        return None
    work = {"clamp": 0, "add": 0, "pool": 0}
    executed = lanes_executed(platform)
    if executed is None:
        return work
    plan = plan_lanes(model_repr, executed)

    def first_input(node) -> int:
        shapes = node.input_shapes
        return _elements(shapes[0]) if shapes and shapes[0] is not None else 0

    for lane in plan.perceptron_lanes:
        if lane.clamp is not None:
            work["clamp"] += first_input(lane.clamp)
        if lane.add is not None:
            work["add"] += first_input(lane.add)
        if lane.reduce is not None:
            work["pool"] += _pool_outputs(lane.reduce)
    for pool in plan.pool_lanes:
        work["pool"] += _pool_outputs(pool.pool)
    return work


def output_stage_of_level(row: LevelCensus, platform: Mapping[str, Any]) -> Optional[OutputStage]:
    """A capability-level census row's output stage (the exploration's candidate points): a requirement placement,
    labelled estimated; ``None`` when the chip declares no ``onchip_arithmetic``."""
    declaration = lane_declaration(platform)
    if declaration is None:
        return None
    return OutputStage(label=ESTIMATED, classes=tuple(sorted(declaration.classes)),
                       lanes_per_core=declaration.lanes_per_core, lane_block=declaration.lane_block,
                       deployed=stage_census(row), notes=(PLACED_LABEL,))
