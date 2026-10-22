"""One residency composition on both planes (acceptance 9): the built program, the shape-only fragments and the
step function agree pass for pass; pins follow the one rule; every refusal is raised by name."""

from __future__ import annotations

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.layout.segmentation import NeuralSegment
from mimarsinan.mapping.packing.hybrid_build_pool import _segment_specs, build_hybrid_hard_core_mapping
from mimarsinan.mapping.packing import schedule_residency_plan
from mimarsinan.mapping.packing.schedule_residency_plan import (
    SCHEDULE_RESIDENCY_PASS,
    compose_builder_program,
    depth_partition,
    mark_residency_plan,
)
from mimarsinan.mapping.platform.mapping_structure import MappingStrategy, ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import check_pinned_fit, dominance_matching
from mimarsinan.mapping.support.schedule.pass_planner import plan_program_passes
from mimarsinan.mapping.support.schedule.residency_composition import (
    ResidencySelection,
    ShapeOnlyResidency,
    compose_units,
)
from mimarsinan.mapping.support.schedule.residency_units import ResidencyUnits, pinned_cores, stateless_step
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType

from ..residency_fixture import load_calls
from . import vehicles as V

C_TIGHT = 5          # F = 11 tiles on the chain: Q >= 3 at this budget


def _composition(chain, count, plan, census=...):
    _rep, specs, wires, program, _ir = chain
    return V.shape_only(specs, wires if census is ... else census, program, count, plan)


def _fragments(chain, count, plan=None):
    _rep, specs, census, program, _ir = chain
    return V.fragments(specs, census, count, plan, program=program)


def _step_layers(comp):
    return [tuple(comp.units.layers[u] for u in range(a, b)) for a, b in comp.facts.phases]


@pytest.fixture(scope="module")
def chain():
    rep = V.chain_repr(seed=0)
    specs, census, program = V.layout_plane(rep)
    return rep, specs, census, program, V.ir_of(rep)


class TestBothPlanesExecuteTheStep:
    @pytest.mark.parametrize("count", [C_TIGHT, 6, 8])
    @pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}, {"step": "index", "index": 1}])
    def test_the_pass_partition_of_both_planes_is_the_steps_phases(self, chain, count, plan):
        rep, specs, census, _program, ir = chain
        comp = _composition(chain, count, plan)
        if plan["step"] == "index" and len(comp.plan.steps) < 2:
            pytest.skip("one step only at this budget")
        hybrid = V.build(ir, count, plan)
        frag = _fragments(chain, count, plan)
        assert V.built_pass_layers(hybrid, ir) == V.fragment_pass_layers(frag, specs) == _step_layers(comp)
        assert len(comp.facts.phases) == comp.step[2] >= 2                       # Q is the step's
        segment_plan, = hybrid.schedule_residency_plan
        assert (segment_plan.rotating_tiles, segment_plan.passes) == (comp.step[0], comp.step[2])
        assert segment_plan.carried_elements == comp.step[1]

    @pytest.mark.parametrize("count", [C_TIGHT, 6, 8])
    def test_the_pins_follow_the_one_rule_on_both_planes(self, chain, count):
        rep, specs, census, _program, ir = chain
        comp = _composition(chain, count, {"step": "floor"})
        r, C = comp.step[0], comp.budget
        hybrid = V.build(ir, count, {"step": "floor"})
        frag = _fragments(chain, count, {"step": "floor"})
        built = [stage.schedule_resident_cores for stage in V.neural_passes(hybrid)]
        assert built == [p.schedule_resident_cores for p in frag.pass_programs]
        packed = [len(stage.hard_core_mapping.cores) for stage in V.neural_passes(hybrid)]
        for pins, cores in zip(built, packed):
            assert len(pins) >= max(0, cores - r)                                 # (packed_q - r)^+ at least
        assert sum(len(p) for p in built) == min(C - r, sum(packed))            # the whole pin budget, no more
        for q, cores in enumerate(packed):                                        # every pass fits with the pins
            assert cores + sum(len(p) for i, p in enumerate(built) if i != q) <= C

    @pytest.mark.parametrize("count", [C_TIGHT, 6, 8])
    @pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
    def test_the_programmed_cells_of_both_planes_are_the_compositions(self, chain, count, plan):
        """Programmed bytes defined once: on this chain both packers pack every phase alike, so the composition's
        written cells (each phase packed as one pass) are the occupied cells of the cores each built pass and each
        fragment pass leaves unpinned (where they do not, ``test_packing_planes`` states the difference)."""
        rep, specs, census, _program, ir = chain
        facts = _composition(chain, count, plan).facts
        hybrid = V.build(ir, count, plan)
        frag = _fragments(chain, count, plan)
        built = [tuple(CoreOccupancy.from_hard_core(core).cells_used for i, core in
                       enumerate(stage.hard_core_mapping.cores) if i not in stage.schedule_resident_cores)
                 for stage in V.neural_passes(hybrid)]
        shape_only = [tuple(c for i, c in enumerate(p.core_cells) if i not in p.schedule_resident_cores)
                      for p in frag.pass_programs]
        assert built == shape_only == list(facts.written_cells)
        assert [s.schedule_resident_cores for s in V.neural_passes(hybrid)] == list(facts.pass_resident_cores)
        assert facts.written_core_tiles == sum(len(c) for c in built)
        for stage in V.neural_passes(hybrid):                                     # one packing: no plane difference
            assert getattr(stage, SCHEDULE_RESIDENCY_PASS).plane_differences() == ()

    def test_the_plan_records_s0s_fields(self, chain):
        rep, specs, census, _program, ir = chain
        for plan, index in (({"step": "floor"}, 0), ({"step": "stateless"}, None), ({"step": "index", "index": 1}, 1)):
            comp = _composition(chain, C_TIGHT, plan)
            segment_plan, = V.build(ir, C_TIGHT, plan).schedule_residency_plan
            assert (segment_plan.step_rule, segment_plan.step_index) == (plan["step"], index)
            assert (segment_plan.rotating_tiles, segment_plan.total_tiles, segment_plan.core_budget) == (
                comp.step[0], comp.plan.F, C_TIGHT)
            assert segment_plan.written_tiles == max(0, comp.plan.F - C_TIGHT + comp.step[0])    # W = F - C + r

    def test_the_stateless_step_pins_no_core(self, chain):
        rep, specs, census, _program, ir = chain
        hybrid = V.build(ir, C_TIGHT, {"step": "stateless"})
        assert all(stage.schedule_resident_cores == frozenset() for stage in V.neural_passes(hybrid))

    def test_every_index_builds_that_step(self, chain):
        rep, specs, census, _program, ir = chain
        comp = _composition(chain, 6, {"step": "floor"})
        for index, (r, carry, Q) in enumerate(comp.plan.steps):
            plan = V.build(ir, 6, {"step": "index", "index": index}).schedule_residency_plan[0]
            assert (plan.rotating_tiles, plan.passes, plan.carried_elements, plan.step_index) == (r, Q, carry, index)

    def test_the_shape_only_planner_serves_the_composition(self, chain):
        rep, specs, census, _program, ir = chain
        program = plan_program_passes(specs, [LayoutHardCoreType(V.MAX_AXONS, V.MAX_NEURONS, C_TIGHT)],
                                      residency=ShapeOnlyResidency({"step": "floor"}, census, _program))
        assert program.residency is not None and program.feasible
        assert program.total_pass_count == program.residency.step[2]
        assert all(not flag for flag in program.resident_flags)                 # U1's whole-pass flag stays off

    def test_absent_the_key_the_build_carries_no_residency_attribute(self, chain):
        rep, specs, census, _program, ir = chain
        hybrid = V.build(ir, C_TIGHT)
        assert not hasattr(hybrid, "schedule_residency_plan")
        assert not any(hasattr(s, "schedule_resident_cores") for s in hybrid.stages)
        frag = _fragments(chain, C_TIGHT)
        assert all(p.schedule_resident_cores is None for p in frag.pass_programs)


class TestTheCompositionIsThePortWithoutForcedCuts:
    def test_on_the_31_spread_cases_the_steps_and_phases_are_plan_steps(self):
        for call in load_calls("spread_31"):
            n = len(call.tiles)
            units = ResidencyUnits(tiles=call.tiles, live=call.live, forced=(),
                                   members=tuple((i,) for i in range(n)), layers=tuple(range(n)),
                                   segments=(0,) * n)
            for selection in (ResidencySelection("floor"), ResidencySelection("stateless")):
                comp = compose_units(units, selection, call.C, max_schedule_passes=64)
                assert comp.plan.as_ra_dict() == call.expected
                want = call.expected["steps"][0] if selection.rule == "floor" else stateless_step(comp.plan)
                assert comp.step == tuple(want)
                assert (len(comp.facts.phases), comp.facts.carried_elements) == (want[2], want[1])


class TestRefusalsByName:
    def test_a_unit_above_the_budget(self, chain):
        rep, specs, census, _program, ir = chain
        with pytest.raises(ResidencyPlanError, match="exceed the core budget"):
            V.build(ir, 2, {"step": "floor"})                                     # a 3-tile layer, C = 2

    def test_an_index_outside_the_step_function(self, chain):
        rep, specs, census, _program, ir = chain
        with pytest.raises(ResidencyPlanError, match="outside the step function"):
            V.build(ir, C_TIGHT, {"step": "index", "index": 99})

    def test_a_pass_count_above_max_schedule_passes(self, chain):
        rep, specs, census, _program, ir = chain
        with pytest.raises(ResidencyPlanError, match="max_schedule_passes"):
            V.build(ir, C_TIGHT, {"step": "floor"}, max_schedule_passes=2)

    def test_a_shape_only_entry_point_without_a_cut_census(self, chain):
        rep, specs, census, _program, ir = chain
        with pytest.raises(ResidencyPlanError, match="cut census"):
            _composition(chain, C_TIGHT, {"step": "floor"}, census=None)

    def test_an_index_one_past_the_last_step(self, chain):
        rep, specs, census, _program, ir = chain
        steps = len(_composition(chain, 6, {"step": "floor"}).plan.steps)
        V.build(ir, 6, {"step": "index", "index": steps - 1})
        with pytest.raises(ResidencyPlanError, match="outside the step function") as caught:
            V.build(ir, 6, {"step": "index", "index": steps})
        assert isinstance(caught.value, ToolchainRefusal)

    def test_exactly_max_schedule_passes_builds_and_one_fewer_refuses(self, chain):
        rep, specs, census, _program, ir = chain
        Q = _composition(chain, C_TIGHT, {"step": "floor"}).step[2]
        assert len(V.neural_passes(V.build(ir, C_TIGHT, {"step": "floor"}, max_schedule_passes=Q))) == Q
        with pytest.raises(ResidencyPlanError, match="max_schedule_passes"):
            V.build(ir, C_TIGHT, {"step": "floor"}, max_schedule_passes=Q - 1)

    def test_the_per_hop_mapping_level_split(self, chain):
        rep, specs, census, _program, ir = chain
        with pytest.raises(ResidencyPlanError, match="per_hop") as caught:
            build_hybrid_hard_core_mapping(ir_graph=ir, cores_config=V.cores(C_TIGHT), per_hop_neural_segments=True,
                                           strategy=MappingStrategy.resolve(V.capabilities({"step": "floor"})))
        assert isinstance(caught.value, ToolchainRefusal)

    def test_built_passes_that_are_not_the_planned_ones(self, chain):
        rep, specs, census, _program, ir = chain
        builder = compose_builder_program(
            [s for s in depth_partition(ir) if isinstance(s, NeuralSegment)], ir_graph=ir,
            cores_config=V.cores(C_TIGHT), residency_plan={"step": "floor"}, segment_specs=_segment_specs,
            allow_coalescing=False, allow_splitting=False, max_schedule_passes=8)
        passes = V.neural_passes(V.build(ir, C_TIGHT, {"step": "floor"}))
        with pytest.raises(ResidencyPlanError, match=f"{len(passes) - 1} built passes for {len(passes)} planned") \
                as caught:
            mark_residency_plan(passes[:-1], builder, V.cores(C_TIGHT))
        assert isinstance(caught.value, ToolchainRefusal)


    def test_the_builder_checks_the_fit_of_its_own_packing(self, chain, monkeypatch):
        """Review round 3 (V7): the builder packs each pass with the hard-core packer, which can need more physical
        cores than the composition's layout packing (t0_03 at x2 floor: 53 against 52), so it checks its own
        packing against the declared pool. Here every executed core is made to occupy two declared cores, which the
        composition (packing the phases itself) never sees: only the builder's own check can refuse it."""
        rep, specs, census, _program, ir = chain
        assert V.neural_passes(V.build(ir, C_TIGHT, {"step": "floor"}))                # the program builds
        real = schedule_residency_plan.core_type_tiles
        monkeypatch.setattr(schedule_residency_plan, "core_type_tiles", lambda *a, **k: real(*a, **k) * 2)
        with pytest.raises(ResidencyPlanError, match="declared pool") as caught:
            V.build(ir, C_TIGHT, {"step": "floor"})
        assert isinstance(caught.value, ToolchainRefusal)


class TestThePinnedRule:
    def test_largest_payload_first_ties_to_the_lower_index(self):
        assert pinned_cores([3, 9, 9, 1], 2) == frozenset({1, 2})
        assert pinned_cores([5, 5, 5], 2) == frozenset({0, 1})
        assert pinned_cores([5, 5], 0) == frozenset()

    def test_a_fused_core_counts_its_tiles(self):
        # a 2-tile fused core of the largest payload does not fit a 1-tile budget; the next core does
        assert pinned_cores([9, 5, 4], 1, core_tiles=[2, 1, 1]) == frozenset({1})
        assert pinned_cores([9, 5, 4], 3, core_tiles=[2, 1, 1]) == frozenset({0, 1})

    def test_the_fit_is_an_exact_dominance_matching(self):
        pool = {(512, 256): 48, (784, 512): 48}
        assert dominance_matching({(512, 256): 49}, pool) == 49           # a small tile on a large core
        assert dominance_matching({(784, 512): 50}, pool) == 48           # never a large tile on a small core
        assert dominance_matching({(1024, 64): 2, (256, 512): 2},
                                  {(1024, 64): 1, (256, 512): 1, (1024, 512): 1}) == 3   # incomparable kinds
        assert dominance_matching({(1024, 64): 1}, {(256, 512): 1}) == 0          # larger in one dimension only
        assert dominance_matching({(256, 512): 1}, {(1024, 64): 1}) == 0

    def test_a_pass_that_cannot_fit_with_the_pins_refuses_by_name(self):
        with pytest.raises(ResidencyPlanError, match="declared pool"):
            check_pinned_fit([[((8, 8),), ((8, 8),)], [((8, 8),), ((8, 8),)]], [frozenset({0}), frozenset({0})],
                             {(8, 8): 2})
