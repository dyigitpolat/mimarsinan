"""[E85] One residency composition for both planes: the declared step's phases as each segment's passes, pinned,
under one set of refusal rules (``mapping/support/residency_rules.py``) that both planes apply by the same calls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.platform.mapping_structure import (
    PoolAdmissionError, ResidencyPlanError, residency_declaration,
)
from mimarsinan.mapping.support.period_census import STEP_FLOOR, STEP_STATELESS
from mimarsinan.mapping.support.residency_rules import (
    ProgramGraph,
    check_pinned_fit,
    declared_pool,
    require_segmentation_parity,
)
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, cut_nodes_from_census
from mimarsinan.mapping.support.schedule.residency_front import ResidencyStepFunction
from mimarsinan.mapping.packing.residency_tiles import pack_specs
from mimarsinan.mapping.support.schedule.residency_units import (
    PlanPeriodFacts,
    ResidencyUnits,
    Step,
    facts_with_cores,
    pin_passes,
    plan_period_facts,
    residency_steps,
    residency_units,
    stateless_step,
)
from mimarsinan.mapping.support.schedule.schedule_budget import effective_core_budget


@dataclass(frozen=True)
class ResidencySelection:
    """The declared ``residency_plan``: ``floor`` the minimum-r step, ``stateless`` the reload plan, ``index`` a step."""

    rule: str
    index: Optional[int] = None

    @classmethod
    def from_declaration(cls, value: Mapping[str, Any]) -> "ResidencySelection":
        rule, index = residency_declaration(value)
        return cls(rule=rule, index=index)

    @property
    def step_index(self) -> Optional[int]:
        """The executed step's index in the step function (``floor`` is ``steps[0]``); None for the stateless plan."""
        return 0 if self.rule == STEP_FLOOR else self.index

    def step_of(self, plan: ResidencyStepFunction) -> Step:
        """The declared step, the function walked only as far as it (the floor is its first step)."""
        if self.rule == STEP_STATELESS:
            return stateless_step(plan)
        index = 0 if self.rule == STEP_FLOOR else self.index
        assert index is not None
        step = plan.step(index)
        if step is None:
            raise ResidencyPlanError(
                f"residency_plan index {self.index} is outside the step function ({len(plan.steps)} steps, "
                f"indices 0..{len(plan.steps) - 1})")
        return step


@dataclass(frozen=True)
class ResidencyComposition:
    """The planned program: units, the step function, the executed step and its phases as passes."""

    selection: ResidencySelection
    units: ResidencyUnits
    plan: ResidencyStepFunction
    step: Step
    facts: PlanPeriodFacts

    @property
    def r(self) -> int:
        return int(self.step[0])

    @property
    def budget(self) -> int:
        return int(self.plan.C)

    def pass_members(self) -> Tuple[Tuple[int, ...], ...]:
        """Spec indices of every pass, program order."""
        return tuple(
            tuple(i for u in range(a, b) for i in self.units.members[u]) for a, b in self.facts.phases
        )

    def pass_segments(self) -> Tuple[int, ...]:
        return tuple(self.units.segments[a] for a, _ in self.facts.phases)

    def segment_passes(self) -> Dict[int, List[Tuple[int, ...]]]:
        out: Dict[int, List[Tuple[int, ...]]] = {}
        for segment, members in zip(self.pass_segments(), self.pass_members()):
            out.setdefault(segment, []).append(members)
        return out


#: Spec indices to packed cores and their tiles, and [F3] optionally each core's operand-bank cells.
PhasePacker = Callable[[Sequence[int]], Tuple[Any, ...]]


def compose_units(units: ResidencyUnits, selection: ResidencySelection, budget: int, *, max_schedule_passes: int,
                  pack_phase: Optional[PhasePacker] = None,
                  pool: Optional[Mapping[Tuple[int, int], int]] = None) -> ResidencyComposition:
    """The step function on ``units``, the declared step and its phases; refuses by name what it cannot plan.

    The facts' cores are each unit's own (its layer packed alone), or with ``pack_phase`` (spec indices to packed cores and
    their tiles) every phase packed as one pass with the layout packer, as ``collect_noc_fragments`` packs it, pinned
    by the one rule and checked against the declared ``pool`` (``check_pinned_fit``, the check the builder runs on
    its own packing). The pass partition is the executed program's on both planes; the per-core
    cells are the layout packer's, which equal the builder's only where both packers pack a phase alike.

    [W03] Packed, the ``floor`` is the smallest r from r* whose composition the pool admits: the type-blind budget C
    can place pins on a core type the pool holds too few of, and admission is not monotone in r (a larger r can pick
    wider phases), so every r is tried in order (r* itself wherever its pins fit, every single-type chip so far). A
    declared index or the stateless plan is composed as declared or refused by name (``PoolAdmissionError``)."""
    plan = residency_steps(units.tiles, units.live, int(budget), forced=units.forced, max_passes=max_schedule_passes,
                           operand=units.operand_tiles)
    step = selection.step_of(plan)
    if pack_phase is None or selection.rule != STEP_FLOOR:
        return compose_step(units, selection, plan, step, max_schedule_passes=max_schedule_passes,
                            pack_phase=pack_phase, pool=pool)
    packs: Dict[Tuple[int, ...], Any] = {}

    def packed_once(members: Sequence[int]):
        key = tuple(members)
        if key not in packs:
            try:
                packs[key] = pack_phase(members)
            except PoolAdmissionError as exc:
                packs[key] = exc
        if isinstance(packs[key], PoolAdmissionError):
            raise packs[key]
        return packs[key]

    last: Optional[PoolAdmissionError] = None
    for r in range(int(step[0]), plan.C + 1):
        try:
            return compose_step(units, selection, plan, plan.at(r) or step, max_schedule_passes=max_schedule_passes,
                                pack_phase=packed_once, pool=pool)
        except PoolAdmissionError as exc:
            last = exc
    raise PoolAdmissionError(
        f"residency_plan floor: at every r from {step[0]} to {plan.C} the declared pool refuses the composition (a "
        f"phase packs on no declared core type, or a pass cannot hold the other passes' pins); at r = {plan.C}: {last}")


def compose_step(units: ResidencyUnits, selection: ResidencySelection, plan: ResidencyStepFunction, step: Step, *,
                 max_schedule_passes: int, pack_phase: Optional[PhasePacker] = None,
                 pool: Optional[Mapping[Tuple[int, int], int]] = None) -> ResidencyComposition:
    """One step's composition (``compose_units`` without the floor's walk): its phases, packed and checked."""
    facts = plan_period_facts(units.tiles, units.live, plan.C, step, forced=units.forced, unit_cores=units.cores or None,
                              max_passes=max_schedule_passes, unit_spans=units.spans,
                              unit_delayed_reads=units.delayed_reads, unit_operand=units.operand_tiles,
                              unit_producer=units.producer)
    composition = ResidencyComposition(selection=selection, units=units, plan=plan, step=step, facts=facts)
    over = {s: len(p) for s, p in composition.segment_passes().items() if len(p) > int(max_schedule_passes)}
    if over:
        raise ResidencyPlanError(
            f"residency_plan step {step} needs more passes than max_schedule_passes = {max_schedule_passes} in "
            f"segment(s) {over}")
    if pack_phase is None:
        return composition
    packed = [pack_phase(m) for m in composition.pass_members()]
    phased = facts_with_cores(facts, [p[0] for p in packed], [p[2] for p in packed if len(p) > 2])
    check_pinned_fit([p[1] for p in packed], phased.pass_resident_cores, dict(pool or {}))
    return replace(composition, facts=phased)


@dataclass(frozen=True)
class ResidencyProgram:
    """One plane's program, planned once and composed at any step under the shared refusal rules: its specs and
    units, its cut nodes (``core_id`` = spec index) and program graph, the declared core types and the packing
    permissions. Every cut it makes runs exact by pass projection (``mapping/packing/pass_timing.py``)."""

    specs: Tuple[LayoutSoftCoreSpec, ...]
    units: ResidencyUnits
    nodes: Tuple[CutNode, ...]
    graph: ProgramGraph
    core_types: Tuple[LayoutHardCoreType, ...]
    allow_coalescing: bool
    allow_splitting: bool
    max_schedule_passes: int

    @property
    def budget(self) -> int:
        """C, the program's core budget (``effective_core_budget``)."""
        return effective_core_budget(
            [{"max_axons": t.max_axons, "max_neurons": t.max_neurons, "count": t.count} for t in self.core_types])

    def steps(self) -> ResidencyStepFunction:
        return residency_steps(self.units.tiles, self.units.live, self.budget, forced=self.units.forced,
                               max_passes=self.max_schedule_passes, operand=self.units.operand_tiles)

    def pack_phase(self, members: Sequence[int]) -> Tuple[Any, ...]:
        return pack_specs([self.specs[i] for i in members], self.core_types, allow_coalescing=self.allow_coalescing,
                          allow_splitting=self.allow_splitting, bounded=True)

    def compose(self, selection: ResidencySelection) -> ResidencyComposition:
        """The step's composition, each phase packed and checked against the declared pool; refuses by name what the
        builder refuses."""
        return compose_units(self.units, selection, self.budget, max_schedule_passes=self.max_schedule_passes,
                             pack_phase=self.pack_phase, pool=declared_pool(self.core_types))

    def compose_at(self, selection: ResidencySelection, step: Step,
                   plan: ResidencyStepFunction) -> ResidencyComposition:
        """One ``step`` of ``plan`` composed, packed and checked against the declared pool, with no walk."""
        return compose_step(self.units, selection, plan, step, max_schedule_passes=self.max_schedule_passes,
                            pack_phase=self.pack_phase, pool=declared_pool(self.core_types))


def residency_program(specs: Sequence[LayoutSoftCoreSpec], nodes: Sequence[CutNode],
                      core_types: Sequence[LayoutHardCoreType], *, segments: Sequence[int], program: ProgramGraph,
                      allow_coalescing: bool, allow_splitting: bool, max_schedule_passes: int,
                      bonds: Sequence[Tuple[int, int]] = ()) -> ResidencyProgram:
    """A plane's program under the key: the program graph is the plane's (each spec once, the plane's segments its
    depth segments), its segments keep every edge class of the reference segmentation
    (``require_segmentation_parity``), then the units (``bonds`` kept in one unit) and live sets."""
    program.require_specs(len(specs))
    program.require_depth_segments(segments)
    require_segmentation_parity(program)
    units = residency_units(specs, nodes, core_types, segments=segments, allow_coalescing=allow_coalescing,
                            allow_splitting=allow_splitting, bonds=bonds)
    return ResidencyProgram(specs=tuple(specs), units=units, nodes=tuple(nodes), graph=program,
                            core_types=tuple(core_types), allow_coalescing=bool(allow_coalescing), allow_splitting=bool(allow_splitting),
                            max_schedule_passes=int(max_schedule_passes))


def compose_residency_program(specs: Sequence[LayoutSoftCoreSpec], nodes: Sequence[CutNode],
                              core_types: Sequence[LayoutHardCoreType], *, segments: Sequence[int],
                              residency_plan: Mapping[str, Any], program: ProgramGraph,
                              allow_coalescing: bool, allow_splitting: bool, max_schedule_passes: int,
                              bonds: Sequence[Tuple[int, int]] = ()) -> ResidencyComposition:
    """One composition over every neural segment of a program, on either plane's specs, cut nodes and graph."""
    selection = ResidencySelection.from_declaration(residency_plan)
    return residency_program(specs, nodes, core_types, segments=segments, program=program,
                             allow_coalescing=allow_coalescing, allow_splitting=allow_splitting,
                             max_schedule_passes=max_schedule_passes, bonds=bonds).compose(selection)


@dataclass(frozen=True)
class ShapeOnlyResidency:
    """A declared ``residency_plan`` on the shape-only plane and what its composition reads from the layout walk: the
    wire census (the cut nodes) and the program graph (``ProgramGraph.of_walk``, which the refusal rules read)."""

    residency_plan: Mapping[str, Any]
    census: Any
    program: Optional[ProgramGraph]


def shape_only_program(softcores: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType], census: Any,
                       program: Optional[ProgramGraph], *, allow_coalescing: bool,
                       allow_splitting: bool, max_schedule_passes: int) -> ResidencyProgram:
    """The shape-only plane's program: cut nodes from the layout walk's wire census, segments from the specs, the
    refusal rules over the walk's program graph; an entry point holding neither refuses by name."""
    if census is None:
        raise ResidencyPlanError(
            "residency_plan needs the layout walk's cut census; this shape-only entry point holds none "
            "(collect the wire census, collect_wire_census=True)")
    if program is None:
        raise ResidencyPlanError(
            "residency_plan needs the layout walk's program graph (ProgramGraph.of_walk) on the shape-only plane: "
            "the builder's segmentation and carry rules read it, so without it this plane would plan programs the "
            "builder refuses")
    specs = list(softcores)
    return residency_program(specs, cut_nodes_from_census(specs, census), core_types,
                             segments=[int(s.segment_id or 0) for s in specs], program=program,
                             allow_coalescing=allow_coalescing,
                             allow_splitting=allow_splitting, max_schedule_passes=max_schedule_passes)


def compose_shape_only_program(softcores: Sequence[LayoutSoftCoreSpec], core_types: Sequence[LayoutHardCoreType],
                               residency: ShapeOnlyResidency, *, allow_coalescing: bool, allow_splitting: bool,
                               max_schedule_passes: int) -> ResidencyComposition:
    """The shape-only plane's composition of the declared step (``shape_only_program``, then ``compose``)."""
    selection = ResidencySelection.from_declaration(residency.residency_plan)
    return shape_only_program(softcores, core_types, residency.census, residency.program,
                              allow_coalescing=allow_coalescing,
                              allow_splitting=allow_splitting, max_schedule_passes=max_schedule_passes).compose(selection)


def refuse_without_cut_census(entry_point: str, residency_plan: Optional[Mapping[str, Any]]) -> None:
    """An entry point that holds no cut census refuses the key by name instead of planning U1 silently."""
    if residency_plan is not None:
        raise ResidencyPlanError(
            f"{entry_point} holds no cut census, so it cannot plan the declared residency_plan "
            f"{dict(residency_plan)!r}; the residency composition needs the layout walk's wire census and program "
            f"graph (collect_noc_fragments) or the IR cores (the hard-core builder)")


def pin_program(pass_payloads: Sequence[Sequence[int]], pass_core_types: Sequence[Sequence[Tuple[Any, ...]]],
                r: int, budget: int, producer: Sequence[Sequence[int]] = ()) -> Tuple[FrozenSet[int], ...]:
    """Pinned cores per executed pass: the one pinned rule (``pin_passes``) on the packed passes' cores ([F3]
    ``producer``: per pass each core's operand-bank cells; its operand cores are never pinned)."""
    return pin_passes([[(int(p), len(t)) for p, t in zip(payloads, types)]
                       for payloads, types in zip(pass_payloads, pass_core_types)], int(r), int(budget), producer)
