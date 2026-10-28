"""[fix_b] The fit rule under scheduling: a chip that holds the program's unscheduled pack programs it once.

U1 splits a segment into capacity passes of the planning budget (and residency streams), so on a larger program its
passes spread over more cores than the pool while the toolchain's unscheduled pack of the same program fits the pool
side by side: t0_43 at 104 + 104 cores ran 15 scheduled passes over 527 cores, never pinned, 27,478,784 B per period,
where the single pool packs 205 cores, pinned, 0 B (reviewer probe, ``X/review_fidelity/fit_probe_out.json``). The
fit was judged only on the scheduler's own passes, so the blueprint rule (C >= F programs once, K27) held only for F
the sum of residency-unit tiles, up to 3.9 x the footprint.

The rule, on both planes, each on its own packer: where U1's passes are not pinned (they do not fit the declared pool
side by side) while the program's unscheduled pack holds on the declared pool (the builder's single shared pool, one
stage per segment, every segment side by side; the shape-only plane's one pack of every spec,
``residency_rules.unscheduled_pack_holds``), that pack is the program, every core pinned. Where U1 pins, its program
stands. F, the footprint, is the smallest pool on which the unscheduled pack holds.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List, Optional

import pytest

from mimarsinan.deployment_record.build.program_census import candidate_program_passes, mapping_program_passes
from mimarsinan.deployment_record.build.write_census import core_capacity_of, period_census
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.layout.segmentation import NeuralSegment, partition_ir_graph
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.support.period_census import SCHEDULE_RESIDENCY_PLAN
from mimarsinan.mapping.support.residency_rules import unscheduled_pack_holds
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics

from . import templates as TV


def _platform(vehicle: TV.TemplateVehicle, counts: List[int], scheduling: bool,
              plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cores = [{**ct, "count": int(c)} for ct, c in zip(vehicle.config["cores"], counts)]
    return {**TV.residency_platform(vehicle, 1.0, plan), "cores": cores, "allow_scheduling": scheduling}


def _build(vehicle: TV.TemplateVehicle, platform: Dict[str, Any]):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)


def _period_bytes(passes, platform, bits: int) -> int:
    return sum(p.payload_bytes for p in period_census(passes, weight_bits=bits,
                                                      capacity=core_capacity_of(platform)).census.passes)


def _pinned(stages) -> bool:
    return all(getattr(s, "schedule_resident_cores", None) == frozenset(range(len(s.hard_core_mapping.cores)))
               for s in stages)


def _segments(vehicle: TV.TemplateVehicle) -> int:
    return sum(1 for s in partition_ir_graph(vehicle.ir) if isinstance(s, NeuralSegment))


def _builder_fit(name: str, counts: List[int], *, single_pool: bool = True, plan: Optional[Dict[str, Any]] = None):
    """The scheduled build at ``counts`` (U1, or under a declared residency ``plan``) equals the single-pool build:
    one pass per segment, every core pinned, 0 B per period."""
    vehicle = TV.template_vehicle(name)
    bits = int(vehicle.config.get("weight_bits", 8))
    single = _build(vehicle, _platform(vehicle, counts, False))
    scheduled_platform = _platform(vehicle, counts, True, plan)
    scheduled = _build(vehicle, scheduled_platform)
    if plan is not None:  # [fix_c] the fit's program is the single pool, so no composition plan is executed
        assert not hasattr(scheduled, SCHEDULE_RESIDENCY_PLAN)
    neural = [s for s in scheduled.stages if s.kind == "neural"]
    if single_pool:
        assert len(neural) == len([s for s in single.stages if s.kind == "neural"]) == _segments(vehicle)
        assert [len(s.hard_core_mapping.cores) for s in neural] == [
            len(s.hard_core_mapping.cores) for s in single.stages if s.kind == "neural"]
    assert _pinned(neural)
    assert _period_bytes(mapping_program_passes(scheduled), scheduled_platform, bits) == 0
    return vehicle, scheduled_platform, neural, bits


def _plane_fit(vehicle, scheduled_platform, neural, bits) -> None:
    """The shape-only plane states the same program: every core pinned, one census row per executed pass, 0 B."""
    specs, census, program, _walk = TV.layout_walk(vehicle)
    frag = collect_noc_fragments(softcores=specs, core_types=core_types_from_declaration(scheduled_platform["cores"]),
                                 census=census, program=program,
                                 **ChipCapabilities.from_platform_constraints(scheduled_platform).layout_kwargs())
    assert all(p.schedule_resident_cores == frozenset(range(p.cores)) for p in frag.pass_programs)
    passes = candidate_program_passes(specs, frag, semantics=StageSemantics(), timesteps=None)
    assert len(passes.rows) == len(neural)
    assert _period_bytes(passes, scheduled_platform, bits) == 0


WITNESSES = [("t0_43", [104, 104]), ("t0_03", [213, 213])]


class TestAChipThatHoldsTheProgramProgramsItOnce:
    def test_where_u1_pins_at_the_footprint_its_program_stands_on_both_planes(self):
        """t0_22 at its single-pool footprint: U1's 5 passes fit the pool side by side, so they are pinned and kept."""
        vehicle, platform, neural, bits = _builder_fit("t0_22", [26, 26], single_pool=False)
        assert len(neural) > _segments(vehicle)
        _plane_fit(vehicle, platform, neural, bits)

    @pytest.mark.slow
    @pytest.mark.parametrize("name, counts", WITNESSES)
    def test_the_reviewers_witnesses_scheduled_equals_the_single_pool_at_zero_bytes(self, name, counts):
        _builder_fit(name, counts)

    @pytest.mark.slow
    @pytest.mark.parametrize("name, counts", WITNESSES)
    def test_the_witnesses_on_the_shape_only_plane(self, name, counts):
        """[E-3] The W04 residual closed: the shape-only plane packs the builder's single pool segment by segment
        (``pack_layout(runs=...)``; t0_43 at 104 + 104: 49, 75, 18, 31, 10, 13, 3, 5 and 1 cores on both planes)."""
        _plane_fit(*_builder_fit(name, counts))


FLOOR = {"step": "floor"}


class TestTheFitRuleUnderADeclaredResidencyPlan:
    """[fix_c] The one fit rule on every plan: the floor (and a declared step index) reprogrammed every period on
    chips that hold the whole program (W03 census, fix_b: 37 of 312 fit points, every one under the floor; t0_43 at
    104 + 104: 19 passes, 21,463,040 B per period, where the single pool pins 9 passes at 0 B). Where the declared
    plan's passes are not pinned whole and the declared pool admits the unscheduled pack, that pack is the program,
    pinned, one pass per segment, on both planes. The stateless plan is the reload baseline (r = C, every core
    rewritten each period, plan section 9.3), which no fit pins."""

    def test_the_floor_at_t0_22s_footprint_is_the_single_pool_on_both_planes(self):
        vehicle, platform, neural, bits = _builder_fit("t0_22", [26, 26], plan=FLOOR)
        assert len(neural) == _segments(vehicle)
        _plane_fit(vehicle, platform, neural, bits)

    def test_a_declared_step_index_at_the_footprint_is_the_single_pool(self):
        _builder_fit("t0_22", [26, 26], plan={"step": "index", "index": 0})

    def test_the_stateless_reload_baseline_pins_no_core_at_the_footprint(self):
        vehicle = TV.template_vehicle("t0_22")
        platform = _platform(vehicle, [26, 26], True, {"step": "stateless"})
        neural = [s for s in _build(vehicle, platform).stages if s.kind == "neural"]
        assert all(getattr(s, "schedule_resident_cores", None) == frozenset() for s in neural)
        assert len(neural) > _segments(vehicle)

    @pytest.mark.slow
    @pytest.mark.parametrize("name, counts", WITNESSES)
    def test_the_reviewers_witnesses_the_floor_equals_the_single_pool_at_zero_bytes(self, name, counts):
        _builder_fit(name, counts, plan=FLOOR)


class TestThePlanesPredicate:
    def _pack(self, vehicle, count):
        from mimarsinan.mapping.layout.layout_packer import pack_layout

        specs, _census, _program, _walk = TV.layout_walk(vehicle)
        types = core_types_from_declaration([{**ct, "count": count} for ct in vehicle.config["cores"]])
        return pack_layout(softcores=specs, core_types=types,
                           allow_neuron_splitting=bool(vehicle.config.get("allow_neuron_splitting", False)),
                           allow_coalescing=bool(vehicle.config.get("allow_coalescing", False))), types

    def test_an_unscheduled_pack_that_holds_on_the_declared_pool_is_a_fit(self):
        vehicle = TV.template_vehicle("t0_22")
        assert unscheduled_pack_holds(*self._pack(vehicle, 26))

    def test_one_core_per_type_is_no_fit(self):
        vehicle = TV.template_vehicle("t0_22")
        assert not unscheduled_pack_holds(*self._pack(vehicle, 1))
