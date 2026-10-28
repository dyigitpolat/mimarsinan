"""[E-3, W04 residual] The shape-only plane's unscheduled pack is the builder's single shared pool.

The builder packs a program unscheduled (``hybrid_build_pool._build_single_pool``) segment by segment in program order,
each segment on cores of its own drawn from one pool. The shape-only plane packed every spec in one co-pack, which the
greedy packer can fail where the segment-by-segment pool holds: t0_43 at 104 + 104 cores, 205 cores in 9 segments on
the builder, infeasible as one co-pack. The W03 head census listed 40 such fit points on 14 templates
(``w03_shape_only_fit_exceptions_40.md``). With the walk's runs (``residency_rules.spec_runs``: two specs share a run
iff they share a builder segment) ``pack_layout`` packs the builder's single pool, so the fit rule, the unscheduled
feasibility and the footprint F judge one program on both planes, and where the pool does not hold both refuse by the
same name (``CoreCapacityRefusal``).

The census vehicle's IR (``templates.template_vehicle``) also took its bias row from the flat config's ``has_bias``
(absent, read as on-core bias) where the pipeline's Soft Core Mapping reads the declared cores
(``resolve_platform_mapping_params``): on chips whose cores carry no on-core bias (t0_63, t0_64, t0_66, t0_68) the
vehicle's executed plane laid out one axon fewer per core than any run of the pipeline. The vehicle maps at the
cores' parameters, as the step does.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List

import pytest

from mimarsinan.mapping.layout.layout_packer import pack_layout
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.deployment_record.build.program_census import candidate_program_passes, mapping_program_passes
from mimarsinan.mapping.noc import collect_noc_fragments, execution_stage_placements
from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities
from mimarsinan.mapping.platform.platform_constraints import resolve_platform_mapping_params
from mimarsinan.mapping.support.residency_rules import spec_runs, unscheduled_pack_holds
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from . import templates as TV

#: W03 head census fit points (integration2/explore/w03_head/census/floor_census.jsonl): per template the counts of
#: its two core types at the smallest scale on which the builder's single pool holds, and a pool it does not hold.
FIT = {"t0_13": [38, 38], "t0_02": [41, 41]}
BELOW = {"t0_02": [30, 30]}


def _cores(vehicle: TV.TemplateVehicle, counts: List[int]) -> List[Dict[str, Any]]:
    return [{**ct, "count": int(c)} for ct, c in zip(vehicle.config["cores"], counts)]


def _builder(vehicle: TV.TemplateVehicle, counts: List[int], scheduling: bool = False, plan=None):
    platform = {**TV.residency_platform(vehicle, 1.0, plan), "cores": _cores(vehicle, counts),
                "allow_scheduling": scheduling}
    with contextlib.redirect_stdout(io.StringIO()):
        return build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config), platform


def _per_stage(hybrid) -> List[int]:
    return [len(s.hard_core_mapping.cores) for s in hybrid.stages if s.kind == "neural"]


def _kw(vehicle: TV.TemplateVehicle) -> Dict[str, bool]:
    return {"allow_neuron_splitting": bool(vehicle.config.get("allow_neuron_splitting", False)),
            "allow_coalescing": bool(vehicle.config.get("allow_coalescing", False))}


@pytest.fixture(scope="module")
def t0_13():
    vehicle = TV.template_vehicle("t0_13")
    return vehicle, TV.layout_walk(vehicle)


class TestTheUnscheduledPackIsTheBuildersSinglePool:
    def test_segment_by_segment_from_one_pool_packs_what_the_builder_packs(self, t0_13):
        vehicle, (specs, _census, _program, walk) = t0_13
        hybrid, _ = _builder(vehicle, FIT["t0_13"])
        types = core_types_from_declaration(_cores(vehicle, FIT["t0_13"]))
        pack = pack_layout(softcores=specs, core_types=types, runs=spec_runs(walk), **_kw(vehicle))
        assert pack.feasible, pack.error
        assert list(pack.segment_cores) == _per_stage(hybrid) == [44, 21, 9]
        assert pack.cores_used == sum(_per_stage(hybrid))
        assert unscheduled_pack_holds(pack, types)

    def test_the_segments_take_distinct_cores_in_program_order(self, t0_13):
        vehicle, (specs, _census, _program, walk) = t0_13
        types = core_types_from_declaration(_cores(vehicle, FIT["t0_13"]))
        runs = spec_runs(walk)
        pack = pack_layout(softcores=specs, core_types=types, runs=runs, collect_placements=True, **_kw(vehicle))
        bounds, start = [], 0
        for n in pack.segment_cores:
            bounds.append((start, start + n))
            start += n
        order = sorted(set(runs))
        for origin, core in pack.placements:
            lo, hi = bounds[order.index(runs[origin])]
            assert lo <= core < hi

    def test_the_unscheduled_stats_are_the_builders_single_pool(self, t0_13):
        vehicle, (specs, census, program, walk) = t0_13
        types = core_types_from_declaration(_cores(vehicle, FIT["t0_13"]))
        stats, error = compute_mapping_stats(specs, types, allow_scheduling=False, census=census, program=program,
                                             runs=spec_runs(walk), **_kw(vehicle))
        assert error is None and stats.feasible

    @pytest.mark.parametrize("plan", [None, {"step": "floor"}], ids=["U1", "floor"])
    def test_the_fit_program_is_one_pinned_pass_per_segment_on_both_planes(self, t0_13, plan):
        """t0_13 at its fit under U1 and under the floor (W03 head census: the floor planned 12 shape-only passes
        against 3 executed and pinned)."""
        vehicle, (specs, census, program, walk) = t0_13
        hybrid, platform = _builder(vehicle, FIT["t0_13"], scheduling=True, plan=plan)
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        assert all(s.schedule_resident_cores == frozenset(range(len(s.hard_core_mapping.cores))) for s in neural)
        frag = collect_noc_fragments(softcores=specs, core_types=core_types_from_declaration(platform["cores"]),
                                     census=census, program=program,
                                     **ChipCapabilities.from_platform_constraints(platform).layout_kwargs())
        assert [p.cores for p in frag.pass_programs] == _per_stage(hybrid)
        assert all(p.schedule_resident_cores == frozenset(range(p.cores)) for p in frag.pass_programs)


class TestBelowTheFitBothPlanesRefuseByTheSameName:
    def test_a_pool_the_builder_does_not_hold_is_refused_by_name_on_both_planes(self):
        vehicle = TV.template_vehicle("t0_02")
        specs, census, program, walk = TV.layout_walk(vehicle)
        with pytest.raises(Exception) as built:
            _builder(vehicle, BELOW["t0_02"])
        assert type(built.value).__name__ == "CoreCapacityRefusal"
        types = core_types_from_declaration(_cores(vehicle, BELOW["t0_02"]))
        pack = pack_layout(softcores=specs, core_types=types, runs=spec_runs(walk), **_kw(vehicle))
        assert not pack.feasible and "CoreCapacityRefusal" in pack.error
        stats, error = compute_mapping_stats(specs, types, allow_scheduling=False, census=census, program=program,
                                             runs=spec_runs(walk), **_kw(vehicle))
        assert not stats.feasible and "CoreCapacityRefusal" in error

    def test_the_fit_point_holds_on_both_planes(self):
        vehicle = TV.template_vehicle("t0_02")
        specs, _census, _program, walk = TV.layout_walk(vehicle)
        hybrid, _ = _builder(vehicle, FIT["t0_02"])
        types = core_types_from_declaration(_cores(vehicle, FIT["t0_02"]))
        pack = pack_layout(softcores=specs, core_types=types, runs=spec_runs(walk), **_kw(vehicle))
        assert pack.feasible and list(pack.segment_cores) == _per_stage(hybrid)


class TestTheFootprint:
    def test_the_packed_tiles_on_unbounded_cores_are_the_builders_per_segment_sum(self):
        """F, the footprint the ladder reads (``packed_tiles``: one declared core shape, a fused core counting its
        members), is the builder's single pool on a pool it never exhausts, segment by segment."""
        from mimarsinan.deployment_record.introspection.scaling_ladder import packed_tiles
        from mimarsinan.deployment_record.introspection.scaling_ladder_types import ShapeCensus

        vehicle = TV.template_vehicle("t0_64")  # one core type, 128 x 128
        specs, census, program, walk = TV.layout_walk(vehicle)
        big = [20_000]
        hybrid, platform = _builder(vehicle, big)
        width = int(vehicle.config["cores"][0]["max_axons"])
        tiles = sum(-(-int(c.axons_per_core) // width) for s in hybrid.stages if s.kind == "neural"
                    for c in s.hard_core_mapping.cores)
        shape = ShapeCensus(softcores=tuple(specs), wire_census=census, host_segments=0,
                            segments=len(set(spec_runs(walk))), logical_cells=1,
                            as_mapped_cells=1, wall_s=0.0, program=program, runs=spec_runs(walk))
        assert packed_tiles(shape, platform) == tiles


class TestTheVehicleMapsAtTheCoresParameters:
    """The census vehicle's IR is the one Soft Core Mapping builds: the cores' bias capability and effective axons."""

    @pytest.mark.parametrize("name", ["t0_63", "t0_64"])
    def test_a_chip_without_on_core_bias_lays_out_the_bias_row_on_both_planes(self, name):
        vehicle = TV.template_vehicle(name)
        params = resolve_platform_mapping_params(vehicle.config["cores"],
                                                 allow_coalescing=bool(vehicle.config.get("allow_coalescing", False)))
        assert params.hardware_bias is False
        specs, _census, _program, _walk = TV.layout_walk(vehicle)
        assert [(s.input_count, s.output_count) for s in vehicle.specs] == [
            (s.input_count, s.output_count) for s in specs]

    def test_a_chip_with_on_core_bias_is_unchanged(self, t0_13):
        vehicle, (specs, _census, _program, _walk) = t0_13
        assert [(s.input_count, s.output_count) for s in vehicle.specs] == [
            (s.input_count, s.output_count) for s in specs]


#: [targeted pass, review M1] W04 witnesses on chips WITHOUT scheduling: the builder's single pool builds (t0_43 at
#: 104 + 104 cores: 9 stages; t0_03 at 213 + 213: 4 stages), while one co-pack of every spec does not pack.
UNSCHEDULED = {"t0_43": [104, 104], "t0_03": [213, 213]}


def _unscheduled(name: str):
    vehicle = TV.template_vehicle(name)
    specs, census, program, walk = TV.layout_walk(vehicle)
    hybrid, platform = _builder(vehicle, UNSCHEDULED[name])
    return vehicle, specs, census, program, walk, hybrid, platform


class TestTheUnscheduledFragmentsAreTheBuildersSinglePool:
    """The candidate plane's two consumers (``compute_mapping_stats``, ``collect_noc_fragments``) judge one program
    on a chip without scheduling: the builder's single pool (segment by segment from one pool). The fragments keep
    their one planned pass of several segments (W03: one execution stage per segment, ``test_fit_residency``,
    ``test_scaling_ladder``), now packed as that pool: each execution stage on its segment's cores, pinned where the
    builder pins (``fit_pins``, the builder's ``mark_fit_residency``)."""

    @pytest.mark.parametrize("name", sorted(UNSCHEDULED))
    def test_stats_fragments_and_builder_agree(self, name):
        _vehicle, specs, census, program, walk, hybrid, platform = _unscheduled(name)
        kw = ChipCapabilities.from_platform_constraints(platform).layout_kwargs()
        assert kw["allow_scheduling"] is False
        types = core_types_from_declaration(platform["cores"])
        stats, error = compute_mapping_stats(specs, types, census=census, program=program, runs=spec_runs(walk),
                                             **{k: kw[k] for k in ("allow_scheduling", "allow_neuron_splitting",
                                                                   "allow_coalescing", "max_schedule_passes")})
        assert error is None and stats.feasible
        frag = collect_noc_fragments(softcores=specs, core_types=types, census=census, program=program, **kw)
        neural = [s for s in hybrid.stages if s.kind == "neural"]
        assert frag.runs == tuple(spec_runs(walk))
        assert [p.cores for p in frag.pass_programs] == [sum(_per_stage(hybrid))] == [stats.total_cores]
        stages = execution_stage_placements(specs, frag.pass_placements, retimed=False, runs=frag.runs)
        assert [len({core for _spec, core in stage}) for stage in stages] == _per_stage(hybrid)
        census_rows = candidate_program_passes(specs, frag, semantics=StageSemantics(), timesteps=None).rows
        record_rows = mapping_program_passes(hybrid).rows  # the record plane's census of the built program
        assert [row.core_cells for row in census_rows] == [row.core_cells for row in record_rows]
        assert all(s.schedule_resident_cores == frozenset(range(len(s.hard_core_mapping.cores))) for s in neural)
        assert [p.schedule_resident_cores for p in frag.pass_programs] == [frozenset(range(sum(_per_stage(hybrid))))]

    def test_a_single_segment_program_keeps_its_one_pass(self, t0_13):
        """Where the program is one builder segment, the single pool is the one co-pack: one pass, as before."""
        vehicle, (specs, census, program, walk) = t0_13
        runs = [0] * len(specs)
        platform = {**TV.residency_platform(vehicle, 1.0, None), "cores": _cores(vehicle, [999, 999]),
                    "allow_scheduling": False}
        kw = ChipCapabilities.from_platform_constraints(platform).layout_kwargs()
        types = core_types_from_declaration(platform["cores"])
        one = collect_noc_fragments(softcores=specs, core_types=types, census=census, program=program, runs=runs, **kw)
        pack = pack_layout(softcores=specs, core_types=types, collect_placements=True, **_kw(vehicle))
        assert len(one.pass_programs) == 1 and one.pass_programs[0].cores == pack.cores_used
        assert one.pass_placements == (tuple(pack.placements),)

    def test_a_joint_search_candidate_on_the_unscheduled_chip_gets_a_verdict(self):
        """The search path (``validate_detailed``: stats, then the NoC fragments) on t0_43 at 104 + 104 returns a
        verdict; before, the stats admitted the candidate and the fragments raised out of the evaluation."""
        from mimarsinan.deployment_record.objectives import OBJECTIVES
        from mimarsinan.search.problems.joint.candidate_fragments import CandidateWalk
        from mimarsinan.search.problems.joint.constrain import JointConstrainMixin
        from mimarsinan.search.problems.joint.evaluate import JointEvaluateMixin
        from mimarsinan.search.problems.joint.layout_hook import JointLayoutMixin
        from mimarsinan.search.problems.joint.validate import JointValidateMixin
        from mimarsinan.search.results import ObjectiveSpec

        _vehicle, specs, census, _program, walk, _hybrid, platform = _unscheduled("t0_43")

        class Harness(JointValidateMixin, JointConstrainMixin, JointLayoutMixin, JointEvaluateMixin):
            search_mode = "joint"
            accuracy_seed = 0
            input_shape = (1, 4, 4)
            num_classes = 2
            validate_fn = constraint_fn = evaluation_budget = None
            encoding_placement = "subsume"
            pruning_fraction = onchip_min_fraction = 0.0
            spiking_mode, ttfs_cycle_schedule, per_hop_retiming = "lif", "cascaded", False
            pass_transfer = None
            active_objective_names = ("total_params", "param_utilization_pct")

            def __init__(self):
                self._cache, self._validation_cache, self._validation_errors = {}, {}, {}
                self._constraint_census, self._hw_only_cache, self._fragment_needs_cache = {}, None, {}

            @property
            def _searches_model(self):
                return True

            @property
            def active_specs(self):
                return OBJECTIVES.resolve_active(self.search_mode, self.active_objective_names)

            @property
            def objectives(self):
                return [ObjectiveSpec(s.name, s.goal) for s in self.active_specs]

            def _resolved_configuration(self, configuration):
                return configuration

            def candidate_encoding_placement(self, configuration):
                return self.encoding_placement

            def _build_raw_model(self, mc, pcfg, placement):
                return object(), 1.0

            def _ensure_mapper_repr(self, model, placement):
                return model

            def _walk_candidate(self, model, pcfg, *, collect_census=False):
                return CandidateWalk(tuple(specs), 0, census, tuple(spec_runs(walk)))

        result = Harness().validate_detailed({"model_config": {}, "platform_constraints": dict(platform)})
        assert result.is_valid, result.error_message
