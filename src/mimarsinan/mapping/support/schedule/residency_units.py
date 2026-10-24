"""[E85] Toolchain residency units: layer-granular tiles, live sets at layer cuts, the step function and its census."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.residency_tiles import PackedCore, packed_cores
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.period_census import PassBoundaryCensus, PassCensus, PassPeriodCensus
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.residency_front import (
    Q_SPAN,
    ResidencyStepFunction,
    phase_partition,
    plan_steps,
    q_fit,
)

Step = Tuple[int, int, int]
@dataclass(frozen=True)
class ResidencyUnits:
    """The planner's input on one program: one unit per layer, in program order over every neural segment.

    ``live[u]`` is the element count live across the cut after unit ``u`` (0 at a forced cut, where the traffic is
    the host boundary priced as today, and after the last unit); ``forced`` holds the cut positions that are
    segment boundaries; ``members[u]`` the spec indices of unit ``u``; ``cores[u]`` the cores unit ``u`` packs into
    alone (``tiles[u]`` their tiles), empty on units stated by tile counts only.
    """

    tiles: Tuple[int, ...]
    live: Tuple[int, ...]
    forced: Tuple[int, ...]
    members: Tuple[Tuple[int, ...], ...]
    layers: Tuple[Optional[int], ...]
    segments: Tuple[int, ...]
    cores: Tuple[Tuple[PackedCore, ...], ...] = ()


def _layer_of_each_spec(specs, nodes) -> List[Any]:
    """Layer identity per spec: its ``perceptron_index``, else the layer it feeds (the first consumer, program order)."""
    consumers: Dict[int, List[int]] = {}
    for node in nodes:
        for source in node.sources:
            consumers.setdefault(int(source), []).append(int(node.core_id))
    layer: List[Any] = [spec.perceptron_index for spec in specs]
    for index in range(len(specs) - 1, -1, -1):
        if layer[index] is None:
            fed = [layer[c] for c in sorted(consumers.get(index, ())) if layer[c] is not None]
            layer[index] = fed[0] if fed else None
    missing = [specs[i].name or i for i, value in enumerate(layer) if value is None]
    if missing:
        raise ResidencyPlanError(
            f"residency_plan needs a layer identity for every core: {missing[:5]} carry no perceptron_index and "
            f"feed no core that does")
    return layer


def residency_units(specs: Sequence[LayoutSoftCoreSpec], nodes: Sequence[CutNode],
                    core_types: Sequence[LayoutHardCoreType], *, segments: Sequence[int],
                    allow_coalescing: bool = False, allow_splitting: bool = False) -> ResidencyUnits:
    """Units and live sets of a program from the plane's specs and cut nodes (``core_id`` = spec index)."""
    layer = _layer_of_each_spec(specs, nodes)
    order: Dict[Tuple[int, Any], List[int]] = {}
    for index in range(len(specs)):
        order.setdefault((int(segments[index]), layer[index]), []).append(index)
    keys = sorted(order, key=lambda k: (k[0], order[k][0]))
    by_id = {int(n.core_id): n for n in nodes}
    live = [0] * len(keys)
    for segment in sorted({k[0] for k in keys}):
        units = [u for u, k in enumerate(keys) if k[0] == segment]
        try:
            cut = PassCut.over([by_id[i] for u in units for i in order[keys[u]]],
                               [order[keys[u]] for u in units])
        except ValueError as exc:
            raise ResidencyPlanError(f"residency_plan: layer order is not a program order in segment "
                                     f"{segment}: {exc}") from exc
        for position, unit in enumerate(units[:-1]):
            live[unit] = cut.cut_width(position)
    forced = tuple(u for u in range(1, len(keys)) if keys[u][0] != keys[u - 1][0])
    cores = tuple(packed_cores([specs[i] for i in order[k]], core_types, allow_coalescing=allow_coalescing,
                               allow_splitting=allow_splitting) for k in keys)
    return ResidencyUnits(
        tiles=tuple(sum(t for _, t in c) for c in cores),
        live=tuple(live), forced=forced, members=tuple(tuple(order[k]) for k in keys),
        layers=tuple(k[1] for k in keys), segments=tuple(k[0] for k in keys), cores=cores,
    )


def cut_nodes_from_census(specs: Sequence[LayoutSoftCoreSpec], census) -> List[CutNode]:
    """The shape-only plane's cut nodes: sources from the wire census, intra-segment only."""
    sources: Dict[int, set] = {i: set() for i in range(len(specs))}
    for producer, consumer in getattr(census, "pair_wires", {}):
        if (specs[int(producer)].segment_id or 0) == (specs[int(consumer)].segment_id or 0):
            sources[int(consumer)].add(int(producer))
    return [CutNode(core_id=i, latency=int(spec.latency_tag or 0), sources=tuple(sorted(sources[i])),
                    out_width=int(spec.output_count)) for i, spec in enumerate(specs)]


def residency_steps(units: Sequence[int], live: Sequence[int], C: int, *,
                    forced: Sequence[int] = ()) -> ResidencyStepFunction:
    """``plan_steps`` on toolchain units: the whole step function, or a refusal by name when no layer-granular
    plan exists (a unit exceeds C). Prices nothing: the choice of step is the pricer's."""
    plan = plan_steps(units, live, int(C), forced=forced)
    if plan is None:
        big = [i for i, t in enumerate(units) if int(t) > int(C)]
        raise ResidencyPlanError(
            f"residency_plan: no layer-granular plan exists at C = {C}: units {big} exceed the core budget")
    return plan


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

    @property
    def Q(self) -> int:
        return len(self.phases)

    @property
    def written_tiles(self) -> int:
        """W = F - C + r, the step's layer-granular writes (0 on a resident program)."""
        return sum(self.pass_tiles) - sum(self.pass_pinned)

    @property
    def carried_elements(self) -> int:
        """Both directions at every cut: out of the chip and back in (2 x live)."""
        return 2 * sum(self.boundary_elements)

    def _written(self) -> List[List[PackedCore]]:
        if len(self.pass_cores) != len(self.phases):
            raise ResidencyPlanError("residency_plan: these period facts were stated by tile counts and carry no "
                                     "packed cores, so they hold no programmed cells")
        return [[core for i, core in enumerate(cores) if i not in pins]
                for cores, pins in zip(self.pass_cores, self.pass_resident_cores)]

    @property
    def written_cells(self) -> Tuple[Tuple[int, ...], ...]:
        """Per pass, the occupied cells of every core it writes each period: its packed cores less the pinned."""
        return tuple(tuple(c for c, _ in cores) for cores in self._written())

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


def pinned_tiles(pass_tiles: Sequence[int], r: int, C: int) -> Tuple[int, ...]:
    """Pins per pass: none without a pinned budget (r = C, the stateless plan: every tile rewritten each period, also
    a phase packed beyond C); else the forced (R_q - r)^+, then the rest of the budget C - r in program order."""
    forced = [max(0, int(t) - int(r)) if int(r) < int(C) else 0 for t in pass_tiles]
    spare = max(0, int(C) - int(r) - sum(forced))
    out = []
    for t, k in zip(pass_tiles, forced):
        extra = min(spare, int(t) - k)
        spare -= extra
        out.append(k + extra)
    return tuple(out)


def pinned_cores(core_payloads: Sequence[int], k: int, core_tiles: Optional[Sequence[int]] = None, *,
                 operand_cores: FrozenSet[int] = frozenset()) -> FrozenSet[int]:
    """Cores of largest payload first (ties to the lower index) while their tiles stay within ``k``; an operand
    core (N2: its payload is rewritten every inference) is never pinned."""
    tiles = core_tiles if core_tiles is not None else [1] * len(core_payloads)
    chosen, used = [], 0
    for i in sorted((i for i in range(len(core_payloads)) if i not in operand_cores),
                    key=lambda i: (-int(core_payloads[i]), i)):
        if used + int(tiles[i]) <= int(k):
            chosen.append(i)
            used += int(tiles[i])
    return frozenset(chosen)


def pin_passes(pass_cores: Sequence[Sequence[PackedCore]], r: int, C: int) -> Tuple[FrozenSet[int], ...]:
    """The one pinned rule, on any plane: ``pinned_tiles`` of each pass's packed tiles, taken by ``pinned_cores``."""
    counts = pinned_tiles([sum(int(t) for _, t in cores) for cores in pass_cores], int(r), int(C))
    return tuple(pinned_cores([c for c, _ in cores], k, [t for _, t in cores]) for cores, k in zip(pass_cores, counts))


def facts_with_cores(facts: PlanPeriodFacts, pass_cores: Sequence[Sequence[PackedCore]]) -> PlanPeriodFacts:
    """The facts with every pass's packed cores and the pinned rule's choice on them."""
    cores = tuple(tuple((int(c), int(t)) for c, t in p) for p in pass_cores)
    return replace(facts, pass_cores=cores, pass_resident_cores=pin_passes(cores, facts.r, facts.budget))


def plan_period_facts(units: Sequence[int], live: Sequence[int], C: int, step: Step, *,
                      forced: Sequence[int] = (),
                      unit_cores: Optional[Sequence[Sequence[PackedCore]]] = None) -> PlanPeriodFacts:
    """The period census of one plan step (an entry of ``steps``, or ``stateless_step``); with ``unit_cores`` (each
    unit's ``ResidencyUnits.cores``) each pass's cores are its units' cores, layer-granular."""
    units = tuple(int(t) for t in units)
    r = int(step[0])
    qf = q_fit(units, int(C), forced)
    if qf is None:
        raise ResidencyPlanError(f"residency_plan: a unit exceeds C = {C}")
    phases = phase_partition(units, live, int(C), r, qf + Q_SPAN, forced)
    if phases is None:
        raise ResidencyPlanError(f"residency_plan: no plan at r = {r}, C = {C}")
    tiles = tuple(sum(units[a:b]) for a, b in phases)
    facts = PlanPeriodFacts(r=r, budget=int(C), phases=phases, pass_tiles=tiles,
                            pass_pinned=pinned_tiles(tiles, r, int(C)),
                            boundary_elements=tuple(int(live[b - 1]) for _, b in phases[:-1]))
    if unit_cores is None:
        return facts
    return facts_with_cores(facts, [[c for u in range(a, b) for c in unit_cores[u]] for a, b in phases])


def plan_period_census(units: Sequence[int], live: Sequence[int], C: int, step: Step, *,
                       bytes_of_cells: Callable[[int], int], tile_cells: int, bytes_per_element: int,
                       forced: Sequence[int] = (), unit_cores: Optional[Sequence[Sequence[PackedCore]]] = None,
                       label: str = "") -> PassPeriodCensus:
    """S0's period census of one plan step (``plan_period_facts``, priced). With ``unit_cores`` a pass's cores are
    its packed cores and its payload their occupied-cell bytes (A1); stated by tiles, each written tile is a full
    tile of ``tile_cells`` cells (W x TB in all). A boundary inside a segment carries its live set each way at
    ``bytes_per_element``; a forced cut starts a segment. A plan step states no compute steps (0)."""
    facts = plan_period_facts(units, live, C, step, forced=forced, unit_cores=unit_cores)
    cuts = sorted(int(p) for p in forced)
    segment = [sum(1 for p in cuts if p <= a) for a, _ in facts.phases]
    if unit_cores is None:
        cores, kept = list(facts.pass_tiles), list(facts.pass_pinned)
        cells = [(t - k) * int(tile_cells) for t, k in zip(cores, kept)]
        payload = [(t - k) * int(bytes_of_cells(int(tile_cells))) for t, k in zip(cores, kept)]
    else:
        cores = [len(c) for c in facts.pass_cores]
        kept = [len(k) for k in facts.pass_resident_cores]
        cells = [sum(w) for w in facts.written_cells]
        payload = [sum(int(bytes_of_cells(c)) for c in w) for w in facts.written_cells]
    passes = tuple(PassCensus(position=q, segment_index=segment[q], resident=kept[q] == cores[q], cores=cores[q],
                              resident_cores=kept[q], written_cells_backing_store=cells[q], written_cells_producer=0,
                              payload_bytes=payload[q], compute_steps=0) for q in range(facts.Q))
    boundaries = tuple(PassBoundaryCensus(after_pass=q, carry_out_bytes=int(live_q) * int(bytes_per_element),
                                          carry_in_bytes=int(live_q) * int(bytes_per_element))
                       for q, live_q in enumerate(facts.boundary_elements) if segment[q] == segment[q + 1])
    return PassPeriodCensus(passes=passes, boundaries=boundaries, label=label)


def units_period_census(units: ResidencyUnits, C: int, step: Step, *, bytes_of_cells: Callable[[int], int],
                        tile_cells: int, bytes_per_element: int, label: str = "") -> PassPeriodCensus:
    """``plan_period_census`` of a step on a program's own units (their tiles, live sets, forced cuts and packed
    cores): the layer-granular census a planned step carries."""
    return plan_period_census(units.tiles, units.live, C, step, bytes_of_cells=bytes_of_cells, tile_cells=tile_cells,
                              bytes_per_element=bytes_per_element, forced=units.forced, unit_cores=units.cores,
                              label=label)


def with_widths(nodes: Sequence[CutNode], specs: Sequence[LayoutSoftCoreSpec]) -> List[CutNode]:
    """Cut nodes carrying each spec's output width and latency tag, the facts both planes state alike (a bank-backed
    IR core holds no matrix to read a width from, and an IR core's latency is unset until the latency pass)."""
    return [replace(n, out_width=int(specs[int(n.core_id)].output_count),
                    latency=int(specs[int(n.core_id)].latency_tag or 0)) for n in nodes]
