"""A capability level's requirement placement scheduled on a core budget (CL4's chip space): its passes and cuts.

The level's on-chip crossbar work in program order is cut into tiles of a declared core geometry and packed greedily
into passes of at most ``budget`` tiles within each neural segment (a pass never crosses a host op). The schedule
states, per pass, the static cells of every tile it loads from the backing store (with each tile's used extent), the
tiles and cells its producers write per inference (a dynamic product's stationary operand) and its depth span (one
execution stage); per cut, the elements produced before it and read after it, counted at every cut they span (G5's
plan census convention, the same width both ways).

It prices nothing and states no period rule: the deployment record turns it into S0's period census through G3's
one rule, the executed rotation (``deployment_record/quantities/from_level.py``), with the residency floor pinning
every tile where the program fits its budget (``pinned``: programmed once, 0 B per inference).

A requirement placement: above C0 nothing is lowered, so the tiles of an operator the level moves on chip are an
estimate, stated in ``TILE_RULE``; every core occupies its own tiles (no packing of two cores into one tile).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.capability.census import in_order_segments, resolved_dependencies
from mimarsinan.mapping.capability.ladder import CHIP, ON_CHIP_WRITE_BACK, CapabilityLevel, CensusNode, place_level
from mimarsinan.mapping.capability.op_kinds import ir_op_kinds, stationary_operand
from mimarsinan.mapping.capability.requirements import CROSSBAR, requirement_of
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.layout.segmentation import compute_node_latencies
from mimarsinan.mapping.support.period_census import natural
from mimarsinan.mapping.verification.onchip_majority import count_host_ops

TILE_RULE = ("each core spans ceil(axons / A) x ceil(neurons / N) tiles of the declared core (axons with the bias "
             "row), each tile holding its block of the core; an on-chip dynamic product holds its stationary operand "
             "per head in producer-written tiles; any other crossbar operator the level moves on chip holds its MACs "
             "in cells at full tiles (extent unstated); no two cores share a tile")
PASS_RULE = "greedy passes of at most C tiles in program order within each neural segment"
#: (after_pass, elements read out after it, elements re-injected before the next): G3's ``BoundaryWidths``.
BoundaryWidths = Tuple[int, int, int]


class LevelScheduleError(ToolchainRefusal, ValueError):
    """A level that cannot be scheduled on the declared cores: a core wider than the core without the target's split
    permission, a core of more tiles than the budget, or a budget that is no positive count."""


@dataclass(frozen=True)
class CoreGeometry:
    """The declared core a level is scheduled on, with the target's permissions to split a wider core."""

    max_axons: int
    max_neurons: int
    allow_coalescing: bool
    allow_splitting: bool


@dataclass(frozen=True)
class ScheduledPass:
    """One pass of the scheduled level (one execution stage of one neural segment)."""

    segment_index: int
    #: Static cells of each tile the pass loads from the backing store, in load order.
    tile_cells: Tuple[int, ...]
    #: Used (axons, neurons) of each of those tiles; None where a tile states none (an operator placed by MACs).
    tile_extents: Optional[Tuple[Tuple[int, int], ...]]
    #: Tiles holding a producer-written stationary operand, and the cells producers write into them per inference.
    producer_tiles: int
    producer_cells: int
    #: The pass's depth span: the greatest less the least latency of its crossbar work.
    stage_latency: int

    @property
    def tiles(self) -> int:
        return len(self.tile_cells) + self.producer_tiles


@dataclass(frozen=True)
class LevelSchedule:
    """The level on ``budget`` tiles: its passes in program order and the carried width at every pass cut."""

    level: str
    payload_write: Optional[str]
    budget: int
    passes: Tuple[ScheduledPass, ...]
    boundary_widths: Tuple[BoundaryWidths, ...]

    @property
    def tiles(self) -> int:
        """F: the tiles of every crossbar unit of the level's program."""
        return sum(p.tiles for p in self.passes)

    @property
    def pinned(self) -> bool:
        """The program fits its budget, so the residency floor pins every tile (W03)."""
        return self.tiles <= self.budget

    @property
    def label(self) -> str:
        write = "" if self.payload_write is None else f" {self.payload_write}"
        return f"{self.level}{write} on {self.budget} tiles; {PASS_RULE}; {TILE_RULE}"


@dataclass(frozen=True)
class _Unit:
    tiles: int
    static_cells: Tuple[int, ...]
    extents: Optional[Tuple[Tuple[int, int], ...]]
    producer_tiles: int
    producer_cells: int


@dataclass(frozen=True)
class _Run:
    """One neural segment of the placed level, prepared once for every budget."""

    ids: np.ndarray
    units: Tuple[Optional[_Unit], ...]
    #: Per node, its distinct producers inside the run (where a lane or wiring node runs).
    producers: Tuple[Tuple[int, ...], ...]
    latencies: Tuple[Optional[int], ...]
    #: Every source read inside the run whose producer is in the run: (producer, element, consumer) node ids.
    reads: Tuple[np.ndarray, np.ndarray, np.ndarray]


def _chunks(extent: int, core: int, permitted: bool, what: str, name: str) -> List[int]:
    if extent > core and not permitted:
        raise LevelScheduleError(f"{name}: {extent} {what} exceed the declared core's {core} and the target permits no "
                                 f"{'coalescing' if what == 'axons' else 'neuron splitting'}")
    full, rest = divmod(int(extent), int(core))
    return [int(core)] * full + ([rest] if rest else [])


def _blocks(axons: int, neurons: int, geometry: CoreGeometry, name: str) -> List[Tuple[int, int]]:
    return [(a, n) for a in _chunks(axons, geometry.max_axons, geometry.allow_coalescing, "axons", name)
            for n in _chunks(neurons, geometry.max_neurons, geometry.allow_splitting, "neurons", name)]


def _unit(node: CensusNode, geometry: CoreGeometry) -> Optional[_Unit]:
    """The node's tiles and cells when it is crossbar work on chip, else None (lanes and wiring take no tile)."""
    ir_node = node.ir_node
    if node.side != CHIP or requirement_of(node.kind).placement != CROSSBAR:
        return None
    if isinstance(ir_node, NeuralCore):
        blocks = _blocks(int(ir_node.get_input_count()), int(ir_node.get_output_count()), geometry, ir_node.name)
        return _Unit(len(blocks), tuple(a * n for a, n in blocks), tuple(blocks), 0, 0)
    assert isinstance(ir_node, ComputeOp)
    if "D" in requirement_of(node.kind).classes:
        _, size, shape = stationary_operand(ir_node)
        rows, cols = (int(shape[-2]), int(shape[-1])) if len(shape) >= 2 else (1, int(size))
        heads = int(size) // max(1, rows * cols)
        producer_tiles = heads * len(_blocks(rows, cols, geometry, ir_node.name))
        return _Unit(producer_tiles, (), (), producer_tiles, int(size))
    cells = int(count_host_ops(IRGraph(nodes=[ir_node], output_sources=np.zeros(0, dtype=object))))
    per_tile = geometry.max_axons * geometry.max_neurons
    full, rest = divmod(cells, per_tile)
    static = (per_tile,) * full + ((rest,) if rest or not full else ())
    return _Unit(len(static), static, None, 0, 0)


def _prepare(run: Sequence[CensusNode], geometry: CoreGeometry, deps: Mapping[int, set]) -> _Run:
    ids = np.array([node.id for node in run], dtype=np.int64)
    members = set(ids.tolist())
    crossbar = {node.id: requirement_of(node.kind).placement == CROSSBAR for node in run}
    latencies = compute_node_latencies({node.id: deps[node.id] & members for node in run}, crossbar)
    producers, reads = [], ([], [], [])
    for node in run:
        inside = np.isin(node.source_ids, ids)
        producers.append(tuple(np.unique(node.source_ids[inside]).tolist()))
        reads[0].append(node.source_ids[inside])
        reads[1].append(node.source_indices[inside])
        reads[2].append(np.full(int(inside.sum()), node.id, dtype=np.int64))
    joined = tuple(np.concatenate(part).astype(np.int64) if part else np.zeros(0, dtype=np.int64) for part in reads)
    return _Run(ids=ids, units=tuple(_unit(node, geometry) for node in run), producers=tuple(producers),
                latencies=tuple(latencies[node.id] if crossbar[node.id] else None for node in run),
                reads=(joined[0], joined[1], joined[2]))


def _assign(run: _Run, budget: int) -> Tuple[List[List[_Unit]], np.ndarray, List[int]]:
    """Greedy passes of at most ``budget`` tiles over the run's crossbar units in program order, the pass of every
    node (a lane or wiring node runs in the pass of its latest producer in the run, else the run's first), and each
    pass's depth span."""
    passes: List[List[_Unit]] = [[]]
    spans: List[List[int]] = [[]]
    used = 0
    of: Dict[int, int] = {}
    for node_id, unit, producers, latency in zip(run.ids.tolist(), run.units, run.producers, run.latencies):
        if unit is None:
            of[node_id] = max((of[p] for p in producers), default=0)
            continue
        if unit.tiles > budget:
            raise LevelScheduleError(f"IR node {node_id}: a core of {unit.tiles} tiles exceeds the core budget of "
                                     f"{budget} tiles")
        if used + unit.tiles > budget and passes[-1]:
            passes.append([])
            spans.append([])
            used = 0
        passes[-1].append(unit)
        spans[-1].append(int(latency or 0))
        used += unit.tiles
        of[node_id] = len(passes) - 1
    pass_of = np.array([of[i] for i in run.ids.tolist()], dtype=np.int64)
    return passes, pass_of, [max(s) - min(s) if s else 0 for s in spans]


def _cut_widths(run: _Run, pass_of: np.ndarray, cuts: int) -> List[int]:
    """Elements produced in a pass and read in a later pass of the run, counted at every cut they span (each
    (producer, element) once per cut)."""
    if cuts == 0 or run.reads[0].size == 0:
        return [0] * cuts
    order = np.argsort(run.ids)

    def passes_of(node_ids: np.ndarray) -> np.ndarray:
        return pass_of[order[np.searchsorted(run.ids, node_ids, sorter=order)]]

    produced, consumed = passes_of(run.reads[0]), passes_of(run.reads[2])
    later = produced < consumed
    keys = run.reads[0][later] * (int(run.reads[1].max(initial=0)) + 1) + run.reads[1][later]
    if keys.size == 0:
        return [0] * cuts
    sort = np.argsort(keys, kind="stable")
    keys, first, last = keys[sort], produced[later][sort], consumed[later][sort]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    widths = np.zeros(cuts + 1, dtype=np.int64)
    np.add.at(widths, first[starts], 1)
    np.add.at(widths, np.maximum.reduceat(last, starts), -1)
    return np.cumsum(widths[:cuts]).astype(int).tolist()


def _schedule(level: CapabilityLevel, payload_write: Optional[str], runs: Sequence[_Run], budget: int) -> LevelSchedule:
    passes: List[ScheduledPass] = []
    widths: List[BoundaryWidths] = []
    for segment, run in enumerate(runs):
        assigned, pass_of, spans = _assign(run, budget)
        start = len(passes)
        for units, span in zip(assigned, spans):
            extents = [u.extents for u in units]
            passes.append(ScheduledPass(
                segment_index=segment, tile_cells=tuple(c for u in units for c in u.static_cells),
                tile_extents=None if any(e is None for e in extents) else tuple(x for e in extents for x in e or ()),
                producer_tiles=sum(u.producer_tiles for u in units),
                producer_cells=sum(u.producer_cells for u in units), stage_latency=span))
        widths.extend((start + cut, width, width)
                      for cut, width in enumerate(_cut_widths(run, pass_of, len(assigned) - 1)))
    return LevelSchedule(level=level.name, payload_write=payload_write, budget=budget, passes=tuple(passes),
                         boundary_widths=tuple(widths))


@dataclass(frozen=True)
class PreparedLevel:
    """A level placed and its neural segments cut into tiles once: ``schedule`` packs it on any budget."""

    level: CapabilityLevel
    payload_write: Optional[str]
    runs: Tuple[_Run, ...]

    @property
    def widest(self) -> int:
        """The tiles of the widest crossbar unit: no budget below it holds the level."""
        return max((u.tiles for run in self.runs for u in run.units if u is not None), default=0)

    def schedule(self, budget: int) -> LevelSchedule:
        if natural(budget, "core budget", error=LevelScheduleError) == 0:
            raise LevelScheduleError("a core budget of 0 tiles holds no pass")
        return _schedule(self.level, self.payload_write, self.runs, int(budget))


def prepare_level(
    ir_graph: IRGraph, level: CapabilityLevel, *, geometry: CoreGeometry,
    payload_write: Optional[str] = ON_CHIP_WRITE_BACK, kinds: Optional[Mapping[int, str]] = None,
    columns: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> PreparedLevel:
    """``ir_graph`` placed at ``level`` and its on-chip crossbar work cut into tiles of ``geometry`` (a core the
    target's permissions cannot split refuses here, whatever the budget)."""
    if "D" not in level.classes:
        payload_write = None
    nodes = place_level(ir_graph, level, kinds if kinds is not None else ir_op_kinds(ir_graph),
                        payload_write=payload_write, columns=columns)
    deps = resolved_dependencies(nodes)
    runs = tuple(_prepare(run, geometry, deps) for run in in_order_segments(nodes) if run[0].side == CHIP)
    return PreparedLevel(level, payload_write, runs)


def level_schedule(
    ir_graph: IRGraph, level: CapabilityLevel, *, budget: int, geometry: CoreGeometry,
    payload_write: Optional[str] = ON_CHIP_WRITE_BACK, kinds: Optional[Mapping[int, str]] = None,
    columns: Optional[Mapping[int, Tuple[np.ndarray, np.ndarray]]] = None,
) -> LevelSchedule:
    """The level placed on ``budget`` tiles of ``geometry`` (``prepare_level`` packs one placement on many budgets)."""
    return prepare_level(ir_graph, level, geometry=geometry, payload_write=payload_write, kinds=kinds,
                         columns=columns).schedule(budget)
