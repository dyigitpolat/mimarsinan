"""The shape-only plane refuses by name exactly what the builder refuses (review round 3), on chain and non-chain graphs.

Both planes run one set of refusal rules (``mapping/support/residency_rules.py``) over one program graph: the IR on
the builder, the layout walk on the shape-only plane (``ProgramGraph.of_walk``). At the same key and budget, every
(vehicle, C, step) below either refuses on both planes with the same message, or plans on both with the same pass
partition (layers per pass) and the same pins. The vehicles are the chain, a skip and a two-level skew (skewed
carries, executed exactly by pass projection since U1 item 1), a balanced skip, wide branches (mixed carried and
host input), a host-op skip into a chain (late host read, two hosts), and the branchy re-segmentations (edge class,
intra to cross, cores only, deep mismatch, moved to output).

The program graph the shape-only plane reads is the layout walk's (``ProgramGraph.of_walk``): its neural producers
are the cells the walk's wire census materializes, so on the shape-only walk the search and CL5 planes use
(``LayoutIRMapping``, whose deferred views record a superset of the producers a node reads) it is still the IR's graph
but for host ops, which hold no census record (t0_30: recorded supersets at one depth); checked on these vehicles and
on tier-0 t0_01 and t0_30 walked as ``layout_hook`` walks them."""

from __future__ import annotations

from dataclasses import replace
from typing import Tuple

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.support.schedule.residency_composition import ShapeOnlyResidency, compose_shape_only_program

from . import runs as R
from . import templates as TV
from . import vehicles as V

KINDS = ["chain", "skip", "skew3", "balanced_skip", "wide_branches", "late_host_read", "two_hosts", "moved_to_output",
         "edge_class", "intra_to_cross", "cores_only", "deep_mismatch"]
COUNTS = [2, 3, 5, 6, 500]
PLANS = [{"step": "floor"}, {"step": "stateless"}, {"step": "index", "index": 1}, {"step": "index", "index": 2}]


def _rep(kind: str):
    return V.chain_repr(seed=0) if kind == "chain" else V.branchy_repr(kind)


def _outcome(call) -> Tuple:
    try:
        return ("plans",) + call()
    except ToolchainRefusal as exc:
        return ("refused", str(exc))


def _builder(ir, count, plan):
    hybrid = V.build(ir, count, plan)
    return (tuple(V.built_pass_layers(hybrid, ir)),
            tuple(stage.schedule_resident_cores for stage in V.neural_passes(hybrid)))


def _shape_only(specs, census, program, count, plan):
    frag = V.fragments(specs, census, count, plan, program=program)
    return (tuple(V.fragment_pass_layers(frag, specs)), tuple(p.schedule_resident_cores for p in frag.pass_programs))


@pytest.fixture(scope="module", params=KINDS)
def vehicle(request):
    rep = _rep(request.param)
    specs, census, program = V.layout_plane(rep)
    return request.param, specs, census, program, V.ir_of(rep)


def test_the_walk_and_the_ir_are_one_program_graph(vehicle):
    """The layout walk allocates node ids in the order the IR walk creates its nodes: one graph, one id per node."""
    _kind, specs, _census, program, ir = vehicle
    spec_of = {int(n.id): i for i, n in enumerate(n for n in ir.nodes if isinstance(n, NeuralCore))}
    of_ir = ProgramGraph.of_ir(ir, spec_of)
    assert program.ids == of_ir.ids and program.producers == of_ir.producers
    assert [s is None for s in program.specs] == [s is None for s in of_ir.specs]
    assert [specs[s].segment_id for s in program.specs if s is not None] == [
        of_ir.depth_segments()[i] for i, s in zip(of_ir.ids, of_ir.specs) if s is not None]
    program.require_specs(len(specs))


def test_both_planes_refuse_or_plan_alike(vehicle):
    kind, specs, census, program, ir = vehicle
    outcomes = {"plans": 0, "refused": 0}
    for count in COUNTS:
        for plan in PLANS:
            built = _outcome(lambda: _builder(ir, count, plan))
            planned = _outcome(lambda: _shape_only(specs, census, program, count, plan))
            assert planned == built, (kind, count, plan)
            outcomes[built[0]] += 1
    assert outcomes["plans"] + outcomes["refused"] == len(COUNTS) * len(PLANS)


class TestTheRulesTheShapeOnlyPlaneNowApplies:
    """The rows the round-2 shape-only plane planned while the builder refused (review/rv2/probes). U1 pass
    projection retires the skew and mixed-input rules: those cuts plan alike on both planes and run exact."""

    @pytest.mark.parametrize("kind", ["skip", "skew3"])
    def test_a_skewed_carry_plans_alike_on_both_planes_and_runs_exact(self, kind):
        rep = V.branchy_repr(kind)
        specs, census, program = V.layout_plane(rep)
        ir = V.ir_of(rep)
        plan = {"step": "floor"}
        assert _shape_only(specs, census, program, 2, plan) == _builder(ir, 2, plan)
        for semantics in R.ALL_RUNS:
            R.assert_runs_equal_to_the_identity_run(
                ir, 2, semantics, mapping=V.build(ir, 2, plan, retimed_level_stages=R.retimed(semantics)))

    def test_wide_branches_mixing_carried_and_host_input_plan_alike_and_run_exact(self):
        rep = V.branchy_repr("wide_branches")
        specs, census, program = V.layout_plane(rep)
        ir = V.ir_of(rep)
        plan = {"step": "index", "index": 2}
        assert _shape_only(specs, census, program, 5, plan) == _builder(ir, 5, plan)
        R.assert_runs_equal_to_the_identity_run(ir, 5, R.STREAMED_LIF, mapping=V.build(ir, 5, plan))

    @pytest.mark.parametrize("kind", ["late_host_read", "two_hosts"])
    @pytest.mark.parametrize("count", [2, 4, 500])
    def test_a_host_op_skip_into_a_chain_refuses_at_every_budget_on_both_planes(self, kind, count):
        """The boundary of review round 3: a core that reads a host op and a deeper core of its own chain moves to a
        segment boundary under the depth segments, so the key refuses it at every budget, C 500 included."""
        rep = V.branchy_repr(kind)
        specs, census, program = V.layout_plane(rep)
        for plan in ({"step": "floor"}, {"step": "stateless"}):
            with pytest.raises(ResidencyPlanError, match="change class") as caught:
                V.shape_only(specs, census, program, count, plan)
            assert isinstance(caught.value, ToolchainRefusal)
            assert "(streamed in the reference, a segment boundary re-segmented)" in str(caught.value)
            with pytest.raises(ResidencyPlanError) as built:
                V.build(V.ir_of(rep), count, plan)
            assert str(caught.value) == str(built.value)


class TestTheShapeOnlyPlaneNeedsTheProgramGraph:
    def test_without_the_walks_program_graph_it_refuses_by_name(self):
        specs, census, _program = V.layout_plane(V.chain_repr())
        with pytest.raises(ResidencyPlanError, match="program graph") as caught:
            V.fragments(specs, census, 5, {"step": "floor"})
        assert isinstance(caught.value, ToolchainRefusal)

    def test_a_graph_of_another_walk_refuses_by_name(self):
        specs, census, _program = V.layout_plane(V.chain_repr())
        _specs, _census, other = V.layout_plane(V.branchy_repr("skip"))
        with pytest.raises(ResidencyPlanError, match="not this plane's walk"):
            V.shape_only(specs, census, other, 5, {"step": "floor"})


@pytest.fixture(scope="module")
def t0_01():
    """Tier-0 t0_01 (lifsync mmixcore): its IR walk and its deferred walk as ``layout_hook`` walks it."""
    vehicle = TV.template_vehicle("t0_01")
    return vehicle, TV.layout_walk(vehicle)


class TestTheShapeOnlyWalksProgramGraph:
    """Closing round 2: the graph of the shape-only walk the search and CL5 planes run (``LayoutIRMapping``, walked as
    ``layout_hook`` walks it). Its deferred views record per node the producers of the whole view a node reads, a
    superset where the view was sliced (on the mmixcore templates t0_01, t0_06, t0_11, t0_21, t0_27, t0_28, t0_30 and
    t0_47, IR node 33 reads [1] and the recorded set is [1, ..., 12]); the program graph takes a neural node's
    producers from the cells its wire-census record materializes, so it is the IR's graph on every tier-0 template but
    t0_30, whose host ops keep recorded supersets (``G5/runs/close2/walk_graph_*.jsonl``)."""

    def test_it_is_the_irs_program_graph_on_t0_01(self, t0_01):
        vehicle, (specs, _census, program, walk) = t0_01
        spec_of = {i: s for i, s in zip(vehicle.program.ids, vehicle.program.specs) if s is not None}
        assert program == ProgramGraph.of_ir(vehicle.ir, spec_of) == vehicle.program
        def shape(spec_list):           # residency classes differ by basis (values on the IR walk), not by shape
            return [(s.input_count, s.output_count, s.latency_tag, s.segment_id, s.bank_id, s.perceptron_index)
                    for s in spec_list]

        assert shape(specs) == shape(vehicle.specs)
        recorded = [frozenset(walk._node_input_node_ids.get(i, ())) for i in program.ids]
        assert sum(r != p for r, p in zip(recorded, program.producers)) > 0       # the superset the census corrects

    def test_on_t0_30_only_host_op_producers_are_recorded_supersets_and_depths_agree(self):
        """The boundary: a host op holds no census record, so its producers are the recorded set. On t0_30 (the
        offload variant) 96 of its 99 host ops read a slice of a layer's view and record the whole view; a host op's
        producers feed only its depth (``compute_segment_ids``), and those producers share one depth there, so the
        depth segments are the IR's (the walk's spec segments, from the same records, likewise)."""
        vehicle = TV.template_vehicle("t0_30")
        specs, _census, program, _walk = TV.layout_walk(vehicle)
        spec_of = {i: s for i, s in zip(vehicle.program.ids, vehicle.program.specs) if s is not None}
        of_ir = ProgramGraph.of_ir(vehicle.ir, spec_of)
        walked, true = dict(zip(program.ids, program.producers)), dict(zip(of_ir.ids, of_ir.producers))
        neural = {i for i, s in zip(of_ir.ids, of_ir.specs) if s is not None}
        assert program.ids == of_ir.ids and program.specs == of_ir.specs
        assert all(walked[i] == true[i] for i in neural)
        assert all(walked[i] >= true[i] for i in of_ir.ids) and sum(walked[i] != true[i] for i in of_ir.ids) == 96
        assert program.depth_segments() == of_ir.depth_segments()
        program.require_depth_segments([int(s.segment_id or 0) for s in specs])

    @pytest.mark.parametrize("scale", [0.6, 1.0])
    def test_its_rule_outcomes_are_the_ir_walks(self, t0_01, scale):
        vehicle, (specs, census, program, _walk) = t0_01
        types = core_types_from_declaration(vehicle.cores(scale))

        def compose(plane_specs, plane_census, plane_program, declared):
            comp = compose_shape_only_program(plane_specs, types,
                                              ShapeOnlyResidency(declared, plane_census, plane_program),
                                              allow_coalescing=bool(vehicle.config.get("allow_coalescing", False)),
                                              allow_splitting=False, max_schedule_passes=8)
            return comp.pass_members(), comp.step, comp.facts.pass_resident_cores

        rows = 0
        for declared in ({"step": "floor"}, {"step": "stateless"}, {"step": "index", "index": 1}):
            walked = _outcome(lambda: compose(specs, census, program, declared))
            assert walked == _outcome(lambda: compose(vehicle.specs, vehicle.census, vehicle.program, declared)), (
                scale, declared)
            rows += walked[0] == "plans"
        assert rows > 0


class TestTheProgramGraphReadsTheWireCensus:
    def test_a_walk_without_the_census_refuses_by_name(self):
        walk = LayoutIRMapping(max_axons=V.MAX_AXONS, max_neurons=V.MAX_NEURONS)
        walk.collect_layout_softcores(V.chain_repr())
        with pytest.raises(ResidencyPlanError, match="wire census") as caught:
            ProgramGraph.of_walk(walk)
        assert isinstance(caught.value, ToolchainRefusal)

    @pytest.mark.parametrize("kind, edit", [("skip", "split"), ("cores_only", "merge")])
    def test_spec_segments_that_are_not_the_graphs_depth_segments_refuse_by_name(self, kind, edit):
        """The plane's segments must be its program graph's depth segments (the segments the rules check): a walk
        whose recorded producers raised a spec's segment would otherwise plan segments the rules never saw. Either
        direction refuses: a depth segment split over two spec segments, or two depth segments merged into one."""
        specs, census, program = V.layout_plane(V.branchy_repr(kind))
        assert len(set(program.depth_segments().values())) == (1 if edit == "split" else 3)
        moved = ([replace(s, segment_id=1) if i == len(specs) - 1 else s for i, s in enumerate(specs)]
                 if edit == "split" else [replace(s, segment_id=0) for s in specs])
        with pytest.raises(ResidencyPlanError, match="depth segments") as caught:
            V.shape_only(moved, census, program, 500, {"step": "floor"})
        assert isinstance(caught.value, ToolchainRefusal)
