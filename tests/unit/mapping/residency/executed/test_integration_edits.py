"""[U1 W9, W10] G5's integration edits IE-1 and IE-3 (report 23.6): the production callers of the planes pass the
walk's runs, the candidate's NoC fragments carry the segmentation they were planned under, and both planes' carry
census counts the builder's delayed carried slices as extra boundary-in slices.

Witnesses: fuzz seed 0 at 3 cores (one builder segment holding two depth segments: the builder runs 4 passes and
syncs them 3 times; a plane without the runs counts 2 syncs), skew3 at 3 cores (the builder pins pass 0 at a span its
latency tags do not give), late_skip at 2 cores (one delayed slice of an 8-wide producer)."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mimarsinan.deployment_record.build.program_census import (
    candidate_carry_census, candidate_program_passes, mapping_program_passes,
)
from mimarsinan.deployment_record.introspection.views import CandidateLayoutView
from mimarsinan.gui.server import routes_layout
from mimarsinan.mapping.latency.chip import PINNED_LATENCIES
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_plan import build_layout_plan
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.noc.fragments import delayed_carry_wires
from mimarsinan.mapping.noc.wire_census import census_of_walk
from mimarsinan.mapping.support.residency_rules import spec_runs
from mimarsinan.mapping.support.schedule.pass_carry import (
    boundary_transfer_bytes, carried_wire_spans, delayed_input_slices, pass_carry_census,
)
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.mapping.verification.layout_verification_scheduling import compute_mapping_stats
from mimarsinan.mapping.verification.verifier.mapping_verifier_hw import clear_verify_hardware_config_memo
from mimarsinan.mapping.verification.verifier.mapping_verifier_soft import verify_soft_core_mapping
from mimarsinan.search.problems.joint import candidate_fragments as CF
from mimarsinan.search.problems.joint.problem import JointArchHwProblem

from . import vehicles as V

SRC = Path(__file__).resolve().parents[5] / "src" / "mimarsinan"
SEMANTICS = SimpleNamespace(retimed=False, is_cycle=False, is_cascade=False, latency_group_count=None,
                            chip_latency=None)
#: The plane functions whose answer depends on the segmentation, and the production files that call them.
PLANE_FUNCTIONS = ("compute_mapping_stats", "verify_hardware_config", "collect_noc_fragments",
                   "collect_candidate_noc", "execution_stage_latencies", "execution_stage_placements")
CALLERS = ("search/problems/joint/layout_hook.py", "search/problems/joint/validate.py",
           "search/problems/joint/candidate_fragments.py", "deployment_record/introspection/views.py",
           "mapping/layout/layout_plan.py", "gui/server/routes_layout.py")


def _builder(rep, count):
    hybrid = V.build(V.ir_of(rep), count, None)
    passes = V.neural_passes(hybrid)
    segments = {int(s.schedule_segment_index) for s in passes}
    return len(passes), len(passes) - len(segments)


def _walk(rep):
    walk = LayoutIRMapping(max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS, collect_wire_census=True)
    return walk, walk.collect_layout_softcores(rep)


class TestEveryCallerPassesTheWalksRuns:
    def test_every_plane_call_of_a_production_caller_passes_runs(self):
        """Generic over the callers: a plane function called without ``runs`` plans the depth segments, which is not
        the deployed program on the divergence class (report 23.3)."""
        offenders = []
        for site in CALLERS:
            for node in ast.walk(ast.parse((SRC / site).read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
                if name in PLANE_FUNCTIONS and "runs" not in {k.arg for k in node.keywords}:
                    offenders.append(f"{site}:{node.lineno} {name}")
        assert not offenders, offenders

    def test_the_candidate_latency_reads_the_wire_census(self):
        """W10: the executed wall prices a pinned pass at its pinned span (``census=``)."""
        calls = [node for node in ast.walk(ast.parse((SRC / CALLERS[2]).read_text(encoding="utf-8")))
                 if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "execution_stage_latencies"]
        assert calls and all("census" in {k.arg for k in node.keywords} for node in calls)


class TestTheWizardPlanesDeploy:
    def test_the_soft_core_verification_carries_the_walks_runs(self):
        rep = V.random_rep(0)
        walk, _ = _walk(rep)
        result = verify_soft_core_mapping(rep, V.MAX_AXONS, V.MAX_NEURONS)
        assert result.runs == spec_runs(walk)

    def test_the_mini_view_plan_deploys_the_builders_passes_and_syncs(self):
        rep = V.random_rep(0)
        clear_verify_hardware_config_memo()
        plan = build_layout_plan(verify_soft_core_mapping(rep, V.MAX_AXONS, V.MAX_NEURONS), V.cores(3),
                                 **V.capabilities().layout_kwargs())
        assert (plan.stats.schedule_pass_count, plan.stats.schedule_sync_count) == _builder(rep, 3) == (4, 3)

    def test_the_layout_route_previews_the_builders_passes_and_syncs(self, monkeypatch):
        rep = V.random_rep(0)
        monkeypatch.setattr(routes_layout, "get_layout_result_from_request",
                            lambda body, **kwargs: verify_soft_core_mapping(rep, V.MAX_AXONS, V.MAX_NEURONS))
        app = FastAPI()
        routes_layout.register_routes(app)
        clear_verify_hardware_config_memo()
        body = {"model_repr_json": {}, "core_types": V.cores(3), "allow_scheduling": True}
        stats = TestClient(app).post("/api/hw_config_verify", json=body).json()["stats"]
        assert (stats["schedule_pass_count"], stats["schedule_sync_count"]) == _builder(rep, 3)


class TestTheIntrospectionViewDeploys:
    def test_a_view_laid_out_from_the_platform_with_the_runs_plans_the_builders_syncs(self):
        rep = V.random_rep(0)
        walk, specs = _walk(rep)
        platform = {"cores": V.cores(3), "allow_scheduling": True}
        view = CandidateLayoutView.from_platform(specs, platform, runs=spec_runs(walk))
        assert view.layout is not None
        assert (view.layout.schedule_pass_count, view.layout.schedule_sync_count) == _builder(rep, 3)


class _Model:
    def __init__(self, rep):
        self._rep = rep

    def get_mapper_repr(self):
        return self._rep


class TestTheSearchCandidatePlanesDeploy:
    PCFG = {"cores": [{"max_axons": V.MAX_AXONS + 1, "max_neurons": V.MAX_NEURONS, "count": 3, "has_bias": False}],
            "allow_scheduling": True}

    def test_the_candidate_walk_carries_its_runs_and_the_stats_plan_with_them(self):
        """The layout hook's walk (effective 25 axons, no hardware bias: the vehicles' walk) carries ``spec_runs``,
        and the packing census planned with them counts the syncs the runs give, not the depth segments' count."""
        problem = object.__new__(JointArchHwProblem)
        rep = V.random_rep(0)
        walk, specs = _walk(rep)
        candidate = problem._walk_candidate(_Model(rep), self.PCFG, collect_census=True)
        assert candidate.runs == spec_runs(walk)
        assert [s.output_count for s in candidate.softcores] == [s.output_count for s in specs]
        stats, error = problem._pack_candidate(list(candidate.softcores), self.PCFG, runs=candidate.runs)
        assert error is None
        types = CF.make_core_types(self.PCFG)
        plain, _ = compute_mapping_stats(list(candidate.softcores), types, **V.capabilities().layout_kwargs())
        assert stats.schedule_sync_count == 3 != plain.schedule_sync_count
        assert problem._collect_softcores(_Model(rep), self.PCFG)[0] == list(candidate.softcores)

    def test_the_candidate_fragments_carry_the_runs_they_were_planned_under(self):
        rep = V.random_rep(0)
        walk, specs = _walk(rep)
        noc = CF.collect_candidate_noc(softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, 3)],
                                       census=census_of_walk(walk), pcfg={"allow_scheduling": True},
                                       runs=spec_runs(walk))
        assert noc is not None and noc.runs == spec_runs(walk)
        assert V.fragment_pass_layers(noc, specs) == V.built_pass_layers(V.build(V.ir_of(rep), 3, None),
                                                                         V.ir_of(rep))

    def test_the_executed_wall_prices_a_pinned_pass_at_its_pinned_span(self):
        rep = V.branchy_repr("skew3")
        specs, census, program = V.layout_plane(rep)
        frag = V.fragments(specs, census, 3, program=program)
        pins = getattr(V.neural_passes(V.build(V.ir_of(rep), 3, None))[0].hard_core_mapping, PINNED_LATENCIES)
        spans = CF.execution_stage_latencies(specs, frag.pass_placements, retimed=False, census=census,
                                             runs=frag.runs)
        assert spans[0] == max(pins)
        steps = CF.candidate_latency_steps(specs, frag, SEMANTICS, 4)
        unpinned = CF.program_latency_steps(
            stage_max_latencies=CF.execution_stage_latencies(specs, frag.pass_placements, retimed=False),
            timesteps=4, is_cycle=False, is_cascade=False, latency_group_count=None, chip_latency=None)
        assert steps == CF.program_latency_steps(stage_max_latencies=spans, timesteps=4, is_cycle=False,
                                                 is_cascade=False, latency_group_count=None, chip_latency=None)
        assert steps != unpinned


class TestTheDelayedSlicesCrossTheBoundary:
    """IE-3: the builder replays a carried producer once per delivery delay; each slice beyond the first crosses the
    chip boundary once more. Both planes' carry census count it at the producer's width (late_skip C 2: producer B,
    8 wide, read twice by pass 1)."""

    def _planes(self):
        rep = V.branchy_repr("late_skip")
        specs, census, program = V.layout_plane(rep)
        frag = V.fragments(specs, census, 2, program=program)
        return specs, frag, V.build(V.ir_of(rep), 2, None)

    def test_the_record_plane_names_the_builders_extra_slices(self):
        _, _, u1 = self._planes()
        assert [width for _, width, _ in delayed_input_slices(u1.stages)] == [8]

    def test_both_planes_count_the_extra_boundary_in_slice(self):
        specs, frag, u1 = self._planes()
        record = pass_carry_census(u1.stages, V.T, VERBATIM)
        candidate = candidate_carry_census(specs, frag, timesteps=V.T, pass_transfer=VERBATIM)
        dense = sum(boundary_transfer_bytes(width=w, timesteps=V.T) for _, w, _, _ in carried_wire_spans(u1.stages))
        extra = boundary_transfer_bytes(width=8, timesteps=V.T)
        assert record == candidate
        assert record["boundary_in_bytes"] == dense + extra
        assert record["boundary_out_bytes"] == dense

    def test_both_planes_widen_the_boundary_before_the_reading_pass(self):
        specs, frag, u1 = self._planes()
        record = mapping_program_passes(u1).boundary_widths
        candidate = candidate_program_passes(specs, frag, semantics=SEMANTICS, timesteps=V.T).boundary_widths
        assert record == candidate
        (position, out_width, in_width), = [row for row in record if row[0] == 0]
        assert in_width == out_width + 8

    def test_the_candidate_slices_are_the_ones_its_fragments_were_planned_under(self):
        specs, frag, _ = self._planes()
        assert delayed_carry_wires(specs, frag) == delayed_carry_wires(specs, frag, frag.runs) == ((1, 8, 1),)

    def test_a_program_without_a_delay_counts_one_crossing_per_wire(self):
        rep = V.branchy_repr("skew3")
        u1 = V.build(V.ir_of(rep), 2, None)
        assert delayed_input_slices(u1.stages) == ()
        census = pass_carry_census(u1.stages, V.T, VERBATIM)
        assert census["boundary_in_bytes"] == census["boundary_out_bytes"]


def test_the_noc_fragments_without_runs_keep_the_depth_segments():
    """Byte identity of the default: a caller that passes no runs (and no program) plans as before."""
    rep = V.random_rep(0)
    walk, specs = _walk(rep)
    frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, 3)],
                                 census=census_of_walk(walk), **V.capabilities().layout_kwargs())
    assert frag.runs is None


BRANCHY = ("edge_class", "intra_to_cross", "cores_only", "deep_mismatch", "moved_to_output", "shift_streamed",
           "two_hosts", "skew3", "shift_probe", "late_skip")


def _carry_planes(rep, count):
    """Both planes' carry census and per-boundary widths of one vehicle at ``count`` cores, the census plane planned
    as the search plans it (the walk's runs); None where the build refuses by name."""
    from mimarsinan.mapping.packing.hybrid_build_scheduled import PlannedPassPackingRefusal

    try:
        u1 = V.build(V.ir_of(rep), count, None)
    except PlannedPassPackingRefusal:
        return None
    walk, specs = _walk(rep)
    frag = collect_noc_fragments(softcores=specs, core_types=[LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, count)],
                                 census=census_of_walk(walk), runs=spec_runs(walk), **V.capabilities().layout_kwargs())
    record = (pass_carry_census(u1.stages, V.T, VERBATIM), mapping_program_passes(u1).boundary_widths)
    candidate = (candidate_carry_census(specs, frag, timesteps=V.T, pass_transfer=VERBATIM),
                 candidate_program_passes(specs, frag, semantics=SEMANTICS, timesteps=V.T).boundary_widths)
    return record, candidate


class TestTheTwoPlanesCarryCensusAgree:
    """The census plane's carry census and boundary widths equal the record plane's on the deployed program, over the
    branchy vehicles (the divergence class included: shift_probe C 3 groups its specs differently in the two
    segmentations, and the census plane counted 2 carried wires against the record's 3) and the fuzz seeds."""

    def test_on_every_branchy_vehicle(self):
        compared = 0
        for kind in BRANCHY:
            for count in (2, 3, 5):
                planes = _carry_planes(V.branchy_repr(kind), count)
                if planes is not None:
                    assert planes[0] == planes[1], (kind, count)
                    compared += 1
        assert compared >= 25

    def test_on_the_fuzz_seeds(self):
        compared = 0
        for seed in range(0, 40):
            for count in (2, 3, 5):
                planes = _carry_planes(V.random_rep(seed), count)
                if planes is not None:
                    assert planes[0] == planes[1], (seed, count)
                    compared += 1
        assert compared >= 100


def _late_skip_part():
    """[MJ3] late_skip read in part: enc -> A -> B, C reads B[0:4], D reads concat(C, B[0:4]), then E. At 2 cores the
    pass holding C and D replays B once per delivery delay while reading 4 of its 8 wires (the fidelity review's probe)."""
    import torch

    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper, SubscriptMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

    torch.manual_seed(0)
    encoder = V._lif(V.IN, V.IN)
    encoder.is_encoding_layer = True
    encoder.use_cycle_accurate_trains = True
    b = V._chain(PerceptronMapper(InputMapper((V.IN,)), encoder), (8, 8), V.IN)
    read = SubscriptMapper(b, slice(0, 4))
    c = PerceptronMapper(read, V._lif(8, 4))
    d = PerceptronMapper(ConcatMapper([c, read]), V._lif(8, 12))
    rep = ModelRepresentation(PerceptronMapper(d, V._lif(4, 8)))
    mark_encoding_layers(rep)
    rep.assign_perceptron_indices()
    return rep


class TestAPartialReadOfADelayedProducer:
    """[MJ3] One width rule on both planes, the source's (G5 report 23.6): a delayed carried slice crosses the boundary
    once more at its producer's width, also where the pass reads only part of the producer."""

    def test_the_record_sizes_the_delayed_slice_at_the_producers_width(self):
        u1 = V.build(V.ir_of(_late_skip_part()), 2, None)
        (node, width, _stage), = delayed_input_slices(u1.stages)
        produced = sum(int(s.size) for st in u1.stages for s in (getattr(st, "output_map", ()) or ())
                       if int(s.node_id) == node)
        assert width == produced == 8

    def test_both_planes_agree_at_every_core_count(self):
        compared = 0
        for count in (2, 3, 4):
            planes = _carry_planes(_late_skip_part(), count)
            if planes is not None:
                assert planes[0] == planes[1], count
                compared += 1
        assert compared == 3


class TestThePlanTimeRefusalsLeaveTheProgramAsBuilt:
    """IE-2 runs the replay refusals in the hard-core mapping step, before the step pickles the program and seals its
    record: judging when a core executes (``ChipLatency``, pins honoured) must not rewrite the built cores' latencies,
    or the pickle and the record's ``static_delay_levels`` would depend on which backends a run enables."""

    @staticmethod
    def _latencies(hybrid):
        return [[core.latency for core in stage.hard_core_mapping.cores] for stage in V.neural_passes(hybrid)]

    def test_every_vehicle_keeps_its_core_latencies(self):
        from mimarsinan.models.spiking.hybrid import carry as C
        from mimarsinan.mapping.packing.hybrid_build_scheduled import PlannedPassPackingRefusal

        judged = 0
        for rep in [V.chain_repr(seed=0), V.branchy_repr("skew3"), *[V.random_rep(seed) for seed in range(12)]]:
            for count in (2, 3, 6):
                try:
                    hybrid = V.build(V.ir_of(rep), count, None)
                except PlannedPassPackingRefusal:
                    continue
                before = self._latencies(hybrid)
                config = {"spiking_family": "lif", "spiking_variant": "streamed", "spike_generation_mode": "Uniform",
                          "enable_nevresim_simulation": True}
                try:
                    C.run_pass_transfer(config, hybrid_mapping=hybrid)
                except C.ChipClockReplayRefusal:
                    pass
                assert self._latencies(hybrid) == before
                judged += 1
        assert judged >= 30
