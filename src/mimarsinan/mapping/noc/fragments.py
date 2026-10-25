"""Per-pass softcore placements + wire census — a candidate's NoC fragments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutSoftCoreSpec,
)
from mimarsinan.mapping.noc.pass_program import PassProgram, packed_pass_program
from mimarsinan.mapping.noc.wire_census import LayoutWireCensus
from mimarsinan.mapping.packing.infeasibility_proofs import CoreCapacityRefusal
from mimarsinan.mapping.packing.pass_timing import project_planned_passes
from mimarsinan.mapping.noc.stages import spec_segments as spec_segments  # the stage rules, re-exported
from mimarsinan.mapping.noc.stages import execution_stage_latencies as execution_stage_latencies
from mimarsinan.mapping.noc.stages import execution_stage_placements as execution_stage_placements
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import (
    ProgramGraph, check_pinned_fit, declared_pool, fit_pins, fit_rule_applies, unscheduled_pack_holds)
from mimarsinan.mapping.support.schedule.pass_planner import (
    plan_program_passes,
)
from mimarsinan.mapping.support.schedule.residency_composition import (
    ResidencyComposition,
    ShapeOnlyResidency,
    pin_program,
)
from mimarsinan.mapping.packing.residency_tiles import CoreTiles


@dataclass(frozen=True)
class LayoutNocFragments:
    """Shape-only pass structure of one candidate.

    ``pass_placements``: per pass, ``(global softcore index, hardcore index)``
    for every placed unit (fragments repeat their origin). ``census``: the
    distinct-wire counts of the walked graph. ``pass_programs``: per pass, the
    occupied-core payload census and its residency. Together with the
    candidate's resolved floorplan and declared activity these are everything
    the NoC, latency and programming models need.
    """

    pass_placements: Tuple[Tuple[Tuple[int, int], ...], ...]
    #: None when no traffic axis asked for it — the placements (cheap) are
    #: always produced because the LATENCY model needs the pass structure.
    census: Optional[LayoutWireCensus]
    pass_programs: Tuple[PassProgram, ...] = ()
    #: [R3] Per-EXECUTION-STAGE placements (stage-local hardcore indices) —
    #: what the runner actually maps; None until the caller that knows the
    #: firing semantics derives them (``execution_stage_placements``).
    stage_placements: Optional[Tuple[Tuple[Tuple[int, int], ...], ...]] = None
    #: [U1 W9] Per spec, the builder segment (IR run, ``spec_runs``) the passes were planned in; None: planned by
    #: the specs' depth segments (no runs stated, or [fix_b] a declared plan's composition, which plans on them).
    #: Every consumer of the fragments reads this one segmentation (``spec_segments``).
    runs: Optional[Tuple[int, ...]] = None


def collect_noc_fragments(
    *,
    softcores: Sequence[LayoutSoftCoreSpec],
    core_types: Sequence[LayoutHardCoreType],
    census: Optional[LayoutWireCensus],
    allow_scheduling: bool,
    allow_neuron_splitting: bool,
    allow_coalescing: bool,
    max_schedule_passes: int,
    residency_plan: Optional[dict] = None,
    program: Optional[ProgramGraph] = None,
    runs: Optional[Sequence[int]] = None,
    declared_cores: Sequence[Mapping[str, Any]] = (),
    position_multiplexing: Optional[bool] = None,
) -> LayoutNocFragments:
    """Place every pass of the candidate program and seal the placements.

    Consumes the ONE planner ``compute_mapping_stats`` and the builder consume [U1] but packs EVERY pass — the NoC
    estimate needs all placements, not just the worst pass's census. Cross-pass producer/consumer pairs are carry
    traffic, not mesh traffic; the estimator excludes them by pass membership. [E85] A declared ``residency_plan`` is
    planned with the walk's ``program`` graph, under the builder's refusal rules. [U1 W9] With the walk's ``runs``
    (``spec_runs``; read off ``program``, else [fix_b] the census, when not given) the specs are grouped by the
    builder's segments and each is planned as the builder plans it; without them by their depth segments, which
    equals the builder wherever the two segmentations group the specs alike (every tier-0 template); the fragments
    state the segments planned (``runs``). [fix_b] Where U1's passes are not pinned (they do not fit the pool side
    by side) while the program's unscheduled pack holds on the declared pool (``unscheduled_pack_holds``), that pack
    is the program, pinned, as the builder builds its single pool (``hybrid_build_fit``); [fix_c] likewise under a
    declared plan whose passes are not pinned whole (``fit_rule_applies``: every plan but the stateless baseline).
    [CL-SG] ``declared_cores`` (the platform's ``cores``) signs each packed core of a per-axon type
    (``carries_sign_register``): its used rows ride the pass program for the payload rule.
    """
    softcores = list(softcores)
    runs = runs if runs is not None else walk_runs(program, len(softcores))
    runs = runs if runs is not None or census is None else census.spec_runs
    runs = None if runs is None else tuple(int(r) for r in runs)
    global_index = {id(sc): i for i, sc in enumerate(softcores)}
    if residency_plan is not None and not allow_scheduling:
        raise ResidencyPlanError("residency_plan is declared but allow_scheduling is not")
    residency: Optional[ResidencyComposition] = None

    if allow_scheduling:
        plan = plan_program_passes(
            softcores, core_types,
            allow_coalescing=allow_coalescing,
            allow_splitting=allow_neuron_splitting,
            max_schedule_passes=max_schedule_passes,
            residency=None if residency_plan is None else ShapeOnlyResidency(
                residency_plan, census, program),
            runs=runs,
            position_multiplexing=position_multiplexing,
        )
        residency = plan.residency
        if not plan.feasible:  # [W04] the capacity refusal both planes name
            raise CoreCapacityRefusal(
                f"NoC fragments need a schedulable candidate; segment(s) "
                f"{list(plan.infeasible_segments())} hold a softcore no "
                f"pass can pack"
            )
        pass_lists = [list(chunk) for chunk in plan.pass_lists]
        # Residency is a per-SEGMENT question: only a segment whose passes
        # the residency composition produced keeps weights installed.
        pass_resident = list(plan.resident_flags)
    else:
        pass_lists = [softcores]
        pass_resident = [False]
    per_pass, programs = _placed_passes(pass_lists, pass_resident, global_index, core_types, residency,
                                        allow_neuron_splitting, allow_coalescing, declared_cores)
    fit = allow_scheduling and fit_rule_applies(residency_plan) and not all(
        p.schedule_resident_cores == frozenset(range(p.cores)) for p in programs) and unscheduled_pack_holds(
        pack_layout(softcores=softcores, core_types=core_types, allow_neuron_splitting=allow_neuron_splitting,
                    allow_coalescing=allow_coalescing), core_types)
    if fit:  # [fix_b, fix_c] the fit rule, under U1 and every declared plan but the stateless baseline
        per_pass, programs = _placed_passes([softcores], [False], global_index, core_types, None,
                                            allow_neuron_splitting, allow_coalescing, declared_cores)
    return LayoutNocFragments(
        pass_placements=tuple(per_pass), census=census, pass_programs=tuple(programs),
        runs=None if residency_plan is not None and not fit else runs,
    )


def _placed_passes(pass_lists, pass_resident, global_index, core_types, residency, allow_neuron_splitting,
                   allow_coalescing, declared_cores) -> Tuple[List[Tuple[Tuple[int, int], ...]], List[PassProgram]]:
    """Every planned pass packed with its placements, and its programs (``packed_pass_program``: per physical tile
    its cells and signed rows, per core its geometry) with the residency marks."""
    per_pass: List[Tuple[Tuple[int, int], ...]] = []
    programs: List[PassProgram] = []
    pass_types: List[Tuple[CoreTiles, ...]] = []
    for pass_softcores, resident in zip(pass_lists, pass_resident):
        pack = pack_layout(
            softcores=pass_softcores, core_types=core_types,
            allow_neuron_splitting=allow_neuron_splitting,
            allow_coalescing=allow_coalescing,
            collect_placements=True,
        )
        if not pack.feasible or pack.placements is None:
            raise CoreCapacityRefusal(
                f"NoC fragments: a planned pass does not pack ({pack.error})"
            )
        per_pass.append(tuple(
            (global_index[id(pass_softcores[local])], hardcore)
            for local, hardcore in pack.placements
        ))
        pass_program, tiles = packed_pass_program(pack.used_core_snapshots or (), resident=bool(resident),
                                                  core_types=core_types, declared_cores=declared_cores)
        programs.append(pass_program)
        pass_types.append(tiles)
    return per_pass, _resident_programs(programs, pass_types, residency, core_types)


def delayed_carry_wires(softcores: Sequence[LayoutSoftCoreSpec], frag: LayoutNocFragments,
                        runs: Optional[Sequence[int]] = None) -> Tuple[Tuple[int, int, int], ...]:
    """[U1 W10] ``(producer, width, consuming pass)`` per delayed carried slice the builder adds beyond one per
    producer and pass: each crosses the chip boundary once more (the boundary-in term). ``runs``: the fragments'
    own when not given."""
    if frag.census is None:
        return ()
    projection = project_planned_passes(
        [int(s.output_count) for s in softcores], [[i for i, _ in p] for p in frag.pass_placements],
        frag.census.pair_wires, spec_segments(softcores, getattr(frag, "runs", None) if runs is None else runs))
    return tuple((p, w, q) for q, extra in enumerate(projection.extra_slices) for p, w in extra)


def walk_runs(program: Optional[ProgramGraph], count: int) -> Optional[Tuple[int, ...]]:
    """Per spec, its IR-run segment read off the walk's program graph (``ProgramGraph.spec_runs``); None without one."""
    return None if program is None else program.spec_runs(count)


def _resident_programs(programs: Sequence[PassProgram], pass_types: Sequence[Tuple[CoreTiles, ...]],
                       residency: Optional[ResidencyComposition],
                       core_types: Sequence[LayoutHardCoreType]) -> List[PassProgram]:
    """The declared plan's pins (``_pinned_programs``), else [W03] every core of a program whose passes fit the
    declared cores side by side (``fit_pins``, the floor at r = 0); otherwise none (U1's executed rotation)."""
    if residency is not None:
        return _pinned_programs(programs, pass_types, residency, core_types)
    pins = fit_pins(list(pass_types), [p.resident for p in programs], declared_pool(core_types))
    return list(programs) if pins is None else [replace(p, schedule_resident_cores=pin)
                                                for p, pin in zip(programs, pins)]


def _pinned_programs(programs: Sequence[PassProgram], pass_types: Sequence[Tuple[CoreTiles, ...]],
                     residency: ResidencyComposition, core_types: Sequence[LayoutHardCoreType]) -> List[PassProgram]:
    """[E85] The pinned rule on each planned pass's packed cells, checked against the declared pool."""
    pins = pin_program([p.core_cells for p in programs], pass_types, residency.r, residency.budget)
    check_pinned_fit(pass_types, pins, declared_pool(core_types))
    return [replace(p, schedule_resident_cores=pin) for p, pin in zip(programs, pins)]
