"""[W03] The floor on chips of two core types: the smallest r whose composition the declared pool admits.

The planner budgets its pins type-blind (C, the 0.8 x pool of ``effective_core_budget``). On a chip of two core types
the pins of the type-blind r* can concentrate on a type the pool holds too few of (t0_01 at x0.125: 13 tiles of
(256, 512) beside a pass, 7 declared), or a phase can need more cores of one type than the pool declares. The floor
is therefore the smallest r from r* whose composition the declared pool admits; admission is not monotone in r (a
larger r can choose a partition with wider phases), so every r is checked in order. Both planes compose the same
step, the builder executes it, and U1 builds at every one of these points; a declared index the pool does not admit
refuses by name.
"""

from __future__ import annotations

import contextlib
import io

import pytest

from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, PoolAdmissionError
from mimarsinan.mapping.support.schedule.residency_composition import ResidencySelection, shape_only_program
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from . import templates as TV

#: (template, core-count scale): two declared core types each, where the type-blind r* is not admitted under the
#: template's own max_schedule_passes.
CASES = [("t0_01", 0.125), ("t0_11", 0.125), ("t0_11", 0.25), ("t0_11", 0.5)]


def _build(vehicle, platform):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)


@pytest.mark.parametrize("name, scale", CASES)
def test_the_floor_builds_where_u1_builds_at_the_smallest_admitted_r(name, scale):
    vehicle = TV.template_vehicle(name)
    platform = TV.residency_platform(vehicle, scale, None)
    assert len(platform["cores"]) == 2
    assert [s for s in _build(vehicle, platform).stages if s.kind == "neural"]                    # U1 builds
    caps = ChipCapabilities.from_platform_constraints(platform)
    types = core_types_from_declaration(platform["cores"])
    program = shape_only_program(vehicle.specs, types, vehicle.census, vehicle.program,
                                 allow_coalescing=caps.allow_coalescing, allow_splitting=caps.allow_neuron_splitting,
                                 max_schedule_passes=caps.max_schedule_passes)
    comp = program.compose(ResidencySelection("floor"))
    r_star = comp.plan.step(0)[0]
    assert comp.r > r_star                                                    # the type-blind r* is not admitted
    for r in range(r_star, comp.r):
        with pytest.raises(PoolAdmissionError):
            program.compose_at(ResidencySelection("floor"), comp.plan.at(r), comp.plan)
    floor = {**platform, "residency_plan": {"step": "floor"}}
    mapping = _build(vehicle, floor)
    passes = [s for s in mapping.stages if s.kind == "neural"]
    frag = collect_noc_fragments(softcores=vehicle.specs, core_types=types, census=vehicle.census,
                                 program=vehicle.program,
                                 **ChipCapabilities.from_platform_constraints(floor).layout_kwargs())
    assert len(passes) == len(frag.pass_programs) == len(comp.facts.phases)
    assert {p.rotating_tiles for p in mapping.schedule_residency_plan} == {comp.r}
    assert [p.schedule_resident_cores for p in frag.pass_programs] == list(comp.facts.pass_resident_cores)
    print(f"[W03] {name} x{scale}: C {comp.budget}, type-blind r* {r_star}, admitted floor r {comp.r}, "
          f"Q {comp.step[2]}, W {comp.facts.written_tiles}")


def test_a_declared_index_the_pool_does_not_admit_refuses_by_name():
    vehicle = TV.template_vehicle("t0_01")
    platform = {**TV.residency_platform(vehicle, 0.125, None), "residency_plan": {"step": "index", "index": 0}}
    with pytest.raises(PoolAdmissionError, match="declared pool"):
        _build(vehicle, platform)
