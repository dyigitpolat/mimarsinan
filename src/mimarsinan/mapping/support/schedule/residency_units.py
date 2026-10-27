"""[E85/W03] Toolchain residency units: tile groups of each layer, live sets at unit cuts, the step function, its census."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
from mimarsinan.mapping.packing.pass_timing import delayed_reads
from mimarsinan.mapping.packing.residency_tiles import PackedCore, layer_of_each_spec, program_order, tile_groups
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_pins import pin_passes, pinned_tiles
from mimarsinan.mapping.support.residency_pins import pinned_cores as pinned_cores  # re-exported (F3 moved it)
from mimarsinan.mapping.support.period_census import (
    PassBoundaryCensus, PassCensus, PassPeriodCensus, phase_crossings, rotation_rewrites,
)
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.pass_cut import cut_nodes_from_census as cut_nodes_from_census
from mimarsinan.mapping.support.schedule.residency_front import (
    Q_SPAN, ResidencyStepFunction, phase_partition, plan_steps, q_fit,
)

Step = Tuple[int, int, int]
@dataclass(frozen=True)
class ResidencyUnits:
    """The planner's input on one program: per layer its tile groups (``tile_groups``), in a program order over every
    neural segment (``program_order`` [F2]), so a pass may hold part of a layer.

    ``live[u]`` is the element count live across the cut after unit ``u`` (0 at a forced cut, where the traffic is
    the host boundary priced as today, and after the last unit); ``forced`` holds the cut positions that are
    segment boundaries; ``members[u]`` the spec indices of unit ``u``; ``layers[u]`` its layer; ``cores[u]`` the
    cores it occupies in its layer packed alone (``tiles[u]`` their tiles), empty on units stated by tile counts;
    ``spans`` each wire carried between units of a segment, (width, producing unit, last consuming unit) [W03];
    ``delayed_reads`` the reads a pass can replay, (width, producing unit, reading unit, producer, delay) [F2];
    ``producer[u]`` per core of ``cores[u]`` the cells of the operand-bank regions it holds, () on a program without
    dynamic operand banks [F3].
    """

    tiles: Tuple[int, ...]
    live: Tuple[int, ...]
    forced: Tuple[int, ...]
    members: Tuple[Tuple[int, ...], ...]
    layers: Tuple[Optional[int], ...]
    segments: Tuple[int, ...]
    cores: Tuple[Tuple[PackedCore, ...], ...] = ()
    spans: Optional[Tuple[Tuple[int, int, int], ...]] = None
    delayed_reads: Tuple[Tuple[int, int, int, int, int], ...] = ()
    producer: Tuple[Tuple[int, ...], ...] = ()

    @property
    def operand_tiles(self) -> Tuple[int, ...]:
        """[F3] Per unit, the tiles of its operand cores (never pinned: they rotate); () without operand banks."""
        return tuple(sum(int(t) for (_, t), p in zip(cores, cells) if p) for cores, cells in
                     zip(self.cores, self.producer)) if self.producer else ()


def residency_units(specs: Sequence[LayoutSoftCoreSpec], nodes: Sequence[CutNode],
                    core_types: Sequence[LayoutHardCoreType], *, segments: Sequence[int],
                    allow_coalescing: bool = False, allow_splitting: bool = False,
                    bonds: Sequence[Tuple[int, int]] = ()) -> ResidencyUnits:
    """Units and live sets of a program from the plane's specs and cut nodes (``core_id`` = spec index): each layer's
    tile groups, a spec kept with every spec of its layer it shares a packed core with, reads or is read by, or that a
    declared ``bond`` (spec index pairs: a coalescing group on the builder) ties to it."""
    layer = layer_of_each_spec(specs, nodes)
    order: Dict[Tuple[int, Any], List[int]] = {}
    for index in range(len(specs)):
        order.setdefault((int(segments[index]), layer[index]), []).append(index)
    tied = [(int(n.core_id), int(s)) for n in nodes for s in n.sources] + [(int(a), int(b)) for a, b in bonds]
    operand = any(getattr(spec, "core_role", None) == OPERAND_ROLE for spec in specs)
    groups: List[Tuple[Any, ...]] = []
    for key in sorted(order, key=lambda k: (k[0], order[k][0])):
        local = {spec: i for i, spec in enumerate(order[key])}
        pairs = [(local[a], local[b]) for a, b in tied if a in local and b in local]
        for members, *cores in tile_groups([specs[i] for i in order[key]], core_types,
                                           allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                                           bonds=pairs, producer=operand):
            groups.append((key[0], key[1], tuple(order[key][m] for m in members), *cores))
    groups = program_order(groups, nodes)
    by_id = {int(n.core_id): n for n in nodes}
    live, spans = [0] * len(groups), []
    for segment in sorted({g[0] for g in groups}):
        units = [u for u, g in enumerate(groups) if g[0] == segment]
        try:
            cut = PassCut.over([by_id[i] for u in units for i in groups[u][2]], [groups[u][2] for u in units])
        except ValueError as exc:
            raise ResidencyPlanError(f"residency_plan: the unit order is not a program order in segment "
                                     f"{segment}: {exc}") from exc
        for unit, width in zip(units[:-1], cut.cut_widths()):
            live[unit] = width
        spans += [(int(w.width), units[w.produced_in], units[w.last_consumed_in]) for w in cut.carried]
    return ResidencyUnits(
        tiles=tuple(sum(t for _, t in g[3]) for g in groups), live=tuple(live),
        forced=tuple(u for u in range(1, len(groups)) if groups[u][0] != groups[u - 1][0]),
        members=tuple(g[2] for g in groups), layers=tuple(g[1] for g in groups),
        segments=tuple(g[0] for g in groups), cores=tuple(g[3] for g in groups), spans=tuple(spans),
        delayed_reads=delayed_reads(specs, nodes, [g[2] for g in groups], segments),
        producer=tuple(g[4] for g in groups) if operand else ())


def residency_steps(units: Sequence[int], live: Sequence[int], C: int, *, forced: Sequence[int] = (),
                    max_passes: Optional[int] = None, operand: Sequence[int] = ()) -> ResidencyStepFunction:
    """``plan_steps`` on toolchain units (at most ``max_passes`` phases per segment [W03]; [F3] ``operand``: per unit
    its operand tiles, which rotate): the step function, or a refusal by name when no plan exists (a unit, one tile
    group, exceeds C, or a segment needs more passes than the cap). Prices nothing: the choice of step is the
    pricer's."""
    plan = plan_steps(units, live, int(C), forced=forced, segment_cap=max_passes, operand=operand)
    if plan is not None:
        return plan
    big = [i for i, t in enumerate(units) if int(t) > int(C)]
    if big:
        raise ResidencyPlanError(
            f"residency_plan: no plan exists at C = {C}: units {big} (each one tile group: cores its specs share, "
            f"or one spec's tiles) exceed the core budget")
    bounds = [0, *sorted(set(int(p) for p in forced)), len(units)]
    need = {k: q_fit(units[a:b], int(C)) for k, (a, b) in enumerate(zip(bounds, bounds[1:]))}
    raise ResidencyPlanError(
        f"residency_plan: no plan exists at C = {C} within max_schedule_passes = {max_passes} per segment: the "
        f"fewest passes per segment are {need}")


def stateless_step(plan: ResidencyStepFunction) -> Step:
    """The stateless plan as a step: r = C (no pinned tile), every tile rewritten each period."""
    return (plan.C, plan.stateless[0], plan.stateless[1])


@dataclass(frozen=True)
class PlanPeriodFacts:
    """The period census of one plan step in tiles, cells and elements (the pricer turns them into bytes): the
    phases, per pass its planned tiles and pins, its packed cores and the cores the pinned rule keeps, and per
    boundary the elements live across it in each direction."""

    r: int
    budget: int
    phases: Tuple[Tuple[int, int], ...]
    pass_tiles: Tuple[int, ...]
    pass_pinned: Tuple[int, ...]
    boundary_elements: Tuple[int, ...]
    #: Per pass, every core it packs into (empty: facts stated by tile counts only), and the pinned rule's cores.
    pass_cores: Tuple[Tuple[PackedCore, ...], ...] = ()
    pass_resident_cores: Tuple[FrozenSet[int], ...] = ()
    #: [W03] Per boundary, the (out, in) elements of the crossing rule (``phase_crossings``); None: stated by live sets.
    boundary_crossings: Optional[Tuple[Tuple[int, int], ...]] = None
    #: [F3] Per pass, each packed core's operand-bank cells (written every inference, resident or not); (): none.
    pass_producer: Tuple[Tuple[int, ...], ...] = ()

    @property
    def Q(self) -> int:
        return len(self.phases)

    @property
    def written_tiles(self) -> int:
        """W = F - C + r, the step's planned tile writes (0 on a resident program); [F3] at least the operand tiles,
        which are never pinned."""
        return sum(self.pass_tiles) - sum(self.pass_pinned)

    @property
    def carried_elements(self) -> int:
        """Both directions at every cut: out of the chip and back in (2 x live)."""
        return 2 * sum(self.boundary_elements)

    def _written(self) -> List[List[PackedCore]]:
        """Per pass, every core it writes each period: its static cells (occupied cells less its operand-bank cells
        [F3], which the producer writes every inference) and tiles."""
        if len(self.pass_cores) != len(self.phases):
            raise ResidencyPlanError("residency_plan: these period facts were stated by tile counts and carry no "
                                     "packed cores, so they hold no programmed cells")
        cells = self.pass_producer or ((),) * len(self.pass_cores)
        return [[(c - (prod[i] if prod else 0), t) for i, (c, t) in enumerate(cores) if i not in pins]
                for cores, pins, prod in zip(self.pass_cores, self.pass_resident_cores, cells)]

    @property
    def written_cells(self) -> Tuple[Tuple[int, ...], ...]:
        """Per pass, the static cells of every core it writes each period: its packed cores less the pinned."""
        return tuple(tuple(c for c, _ in cores) for cores in self._written())

    @property
    def producer_cells(self) -> Tuple[int, ...]:
        """[F3] Per pass, the operand-bank cells of its cores (written every inference, resident or not)."""
        return tuple(sum(p) for p in self.pass_producer) if self.pass_producer else (0,) * self.Q

    @property
    def written_core_tiles(self) -> int:
        """W_exec: the tiles of the written cores. It differs from W in either direction: packing a phase jointly can
        save tiles, and packing it on the declared counts can cost tiles (a unit is its layer packed alone on its best
        type, a phase contends for that type)."""
        return sum(t for cores in self._written() for _, t in cores)

    def programmed_bytes(self, bytes_of_cells: Callable[[int], int]) -> int:
        """Programmed bytes per period, defined once: every written core's occupied cells, priced core by core by
        ``bytes_of_cells`` (the record's ``params_bytes`` at the platform's weight width)."""
        return sum(int(bytes_of_cells(c)) for cells in self.written_cells for c in cells)


def facts_with_cores(facts: PlanPeriodFacts, pass_cores: Sequence[Sequence[PackedCore]],
                     producer: Sequence[Sequence[int]] = ()) -> PlanPeriodFacts:
    """The facts with every pass's packed cores and the pinned rule's choice on them; [F3] with ``producer`` (per pass
    each core's operand-bank cells), whose operand cores the rule never pins."""
    cores = tuple(tuple((int(c), int(t)) for c, t in p) for p in pass_cores)
    cells = tuple(tuple(int(c) for c in p) for p in producer) if any(any(p) for p in producer) else ()
    return replace(facts, pass_cores=cores, pass_producer=cells,
                   pass_resident_cores=pin_passes(cores, facts.r, facts.budget, cells))


def plan_period_facts(units: Sequence[int], live: Sequence[int], C: int, step: Step, *, forced: Sequence[int] = (),
                      unit_cores: Optional[Sequence[Sequence[PackedCore]]] = None, max_passes: Optional[int] = None,
                      unit_spans: Optional[Sequence[Tuple[int, int, int]]] = None,
                      unit_delayed_reads: Sequence[Tuple[int, int, int, int, int]] = (),
                      unit_operand: Sequence[int] = (), unit_producer: Sequence[Sequence[int]] = ()) -> PlanPeriodFacts:
    """The period census of one plan step (an entry of ``steps``, or ``stateless_step``, under ``max_passes``); with
    ``unit_cores`` and ``unit_spans`` (``ResidencyUnits.cores``, ``.spans``) each pass's cores and crossings, the
    crossings with the replays of ``unit_delayed_reads`` (``.delayed_reads`` [F2]); [F3] ``unit_operand`` per unit its
    operand tiles (``.operand_tiles``: never pinned) and ``unit_producer`` its cores' operand-bank cells."""
    units = tuple(int(t) for t in units)
    r = int(step[0])
    qf = q_fit(units, int(C), forced)
    if qf is None:
        raise ResidencyPlanError(f"residency_plan: a unit exceeds C = {C}")
    phases = phase_partition(units, live, int(C), r, qf + Q_SPAN, forced, segment_cap=max_passes, operand=unit_operand)
    if phases is None:
        raise ResidencyPlanError(f"residency_plan: no plan at r = {r}, C = {C}")
    tiles = tuple(sum(units[a:b]) for a, b in phases)
    facts = PlanPeriodFacts(r=r, budget=int(C), phases=phases, pass_tiles=tiles,
                            pass_pinned=pinned_tiles(tiles, r, int(C), tuple(sum(unit_operand[a:b]) for a, b in phases)
                                                     if unit_operand else ()),
                            boundary_elements=tuple(int(live[b - 1]) for _, b in phases[:-1]),
                            boundary_crossings=None if unit_spans is None else phase_crossings(
                                unit_spans, phases, unit_delayed_reads))
    if unit_cores is None:
        return facts
    return facts_with_cores(facts, [[c for u in range(a, b) for c in unit_cores[u]] for a, b in phases],
                            [[c for u in range(a, b) for c in unit_producer[u]] for a, b in phases] if unit_producer
                            else ())


def plan_period_census(units: Sequence[int], live: Sequence[int], C: int, step: Step, *,
                       bytes_of_cells: Callable[[int], int], tile_cells: int, bytes_per_element: int,
                       forced: Sequence[int] = (), unit_cores: Optional[Sequence[Sequence[PackedCore]]] = None,
                       label: str = "", max_passes: Optional[int] = None,
                       unit_spans: Optional[Sequence[Tuple[int, int, int]]] = None,
                       unit_delayed_reads: Sequence[Tuple[int, int, int, int, int]] = (),
                       unit_operand: Sequence[int] = (), unit_producer: Sequence[Sequence[int]] = ()
                       ) -> PassPeriodCensus:
    """S0's period census of one plan step (``plan_period_facts``, priced). With ``unit_cores`` a pass's cores are
    its packed cores and its payload their occupied-cell bytes (A1); stated by tiles, each written tile is a full
    tile of ``tile_cells`` cells (W x TB in all). A boundary inside a segment carries (``bytes_per_element``) the
    crossings of ``unit_spans`` and ``unit_delayed_reads`` (the record's rule [W03, F2]), else its live set each way;
    a forced cut starts a segment. No compute steps (0). Writes by the executed rotation: a lone writing pass (a fit's
    one pass) writes nothing; [F3] a pass's operand-bank cells are written every inference (``unit_producer``)."""
    facts = plan_period_facts(units, live, C, step, forced=forced, unit_cores=unit_cores, max_passes=max_passes,
                              unit_spans=unit_spans, unit_delayed_reads=unit_delayed_reads, unit_operand=unit_operand,
                              unit_producer=unit_producer)
    return facts_period_census(facts, forced=forced, bytes_of_cells=bytes_of_cells, tile_cells=tile_cells,
                               bytes_per_element=bytes_per_element, label=label)


def facts_period_census(facts: PlanPeriodFacts, *, forced: Sequence[int] = (), bytes_of_cells: Callable[[int], int],
                        tile_cells: int, bytes_per_element: int, label: str = "") -> PassPeriodCensus:
    """``PassPeriodCensus`` of any plan step's facts: the unit facts of ``plan_period_facts`` or a plane's
    composition, whose passes carry the cores that plane packs (``facts_with_cores``)."""
    cuts = sorted(int(p) for p in forced)
    segment = [sum(1 for p in cuts if p <= a) for a, _ in facts.phases]
    if not facts.pass_cores:
        cores, kept = list(facts.pass_tiles), list(facts.pass_pinned)
        cells = [(t - k) * int(tile_cells) for t, k in zip(cores, kept)]
        payload = [(t - k) * int(bytes_of_cells(int(tile_cells))) for t, k in zip(cores, kept)]
    else:  # physical tiles, as the record and candidate planes count them (a fused core counts its members)
        cores = [sum(int(t) for _, t in c) for c in facts.pass_cores]
        kept = [sum(int(c[i][1]) for i in k) for c, k in zip(facts.pass_cores, facts.pass_resident_cores)]
        cells = [sum(w) for w in facts.written_cells]
        payload = [sum(int(bytes_of_cells(c)) for c in w) for w in facts.written_cells]
    if not rotation_rewrites(sum(1 for c, k in zip(cores, kept) if c > k)):
        kept, cells, payload = list(cores), [0] * len(cores), [0] * len(cores)
    passes = tuple(PassCensus(position=q, segment_index=segment[q], resident=kept[q] == cores[q], cores=cores[q],
                              resident_cores=kept[q], written_cells_backing_store=cells[q],
                              written_cells_producer=facts.producer_cells[q],
                              payload_bytes=payload[q], compute_steps=0) for q in range(facts.Q))
    crossing = (facts.boundary_crossings if facts.boundary_crossings is not None   # [W03] stated, else live each way
                else tuple((v, v) for v in facts.boundary_elements))
    boundaries = tuple(PassBoundaryCensus(after_pass=q, carry_out_bytes=int(o) * int(bytes_per_element),
                                          carry_in_bytes=int(i) * int(bytes_per_element))
                       for q, (o, i) in enumerate(crossing) if segment[q] == segment[q + 1])
    return PassPeriodCensus(passes=passes, boundaries=boundaries, label=label)


def units_period_census(units: ResidencyUnits, C: int, step: Step, *, bytes_of_cells: Callable[[int], int],
                        tile_cells: int, bytes_per_element: int, label: str = "",
                        max_passes: Optional[int] = None) -> PassPeriodCensus:
    """``plan_period_census`` of a step on a program's own units (their tiles, live sets, forced cuts and packed
    cores): the unit census a planned step carries."""
    return plan_period_census(units.tiles, units.live, C, step, bytes_of_cells=bytes_of_cells, tile_cells=tile_cells,
                              bytes_per_element=bytes_per_element, forced=units.forced, unit_cores=units.cores,
                              label=label, max_passes=max_passes, unit_spans=units.spans,
                              unit_delayed_reads=units.delayed_reads, unit_operand=units.operand_tiles,
                              unit_producer=units.producer)
