"""Per-core packing on the two planes (review round 2). The pass partition is one composition on both planes; the
cores of a pass are each plane's own packing: the composition packs each phase with the layout packer (``pack_layout``
on the plane's specs, as ``collect_noc_fragments`` packs a pass), the builder each pass's IR cores with the hard-core
packer. Per-core cells, programmed bytes and pins therefore agree only where both packers pack a phase alike, and each
executed pass records its difference from the composition with the cause (``schedule_residency_pass``); the record's
bytes are the marks'.

A phase packed on the declared (bounded) pool can need more tiles than its layers packed alone, so the executed
program is admitted against the declared pool (the physical bound), never refused for a layer-granular count or for
the type-blind planning budget: a program with F <= C gives the resident census on both planes."""

from __future__ import annotations

import contextlib
import io

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.mapping.crossbar_utilization import CoreOccupancy
from mimarsinan.mapping.layout.layout_types import core_types_from_declaration
from mimarsinan.mapping.noc import collect_noc_fragments
from mimarsinan.mapping.packing.schedule_residency_plan import SCHEDULE_RESIDENCY_PASS
from mimarsinan.mapping.platform.mapping_structure import ChipCapabilities, ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import check_pinned_fit
from mimarsinan.mapping.support.schedule.residency_composition import (
    ResidencySelection,
    ShapeOnlyResidency,
    compose_shape_only_program,
    compose_units,
)
from mimarsinan.mapping.support.schedule.residency_units import ResidencyUnits
from mimarsinan.pipelining.core.simulation_factory import build_hybrid_mapping_for_pipeline

from . import templates as TV


def _planes(name: str, scale: float, step: str):
    """The shape-only composition, collect_noc_fragments and the built program of one template step."""
    vehicle = TV.template_vehicle(name)
    platform = TV.residency_platform(vehicle, scale, {"step": step})
    caps = ChipCapabilities.from_platform_constraints(platform)
    types = core_types_from_declaration(platform["cores"])
    comp = compose_shape_only_program(vehicle.specs, types,
                                      ShapeOnlyResidency({"step": step}, vehicle.census, vehicle.program),
                                      allow_coalescing=caps.allow_coalescing,
                                      allow_splitting=caps.allow_neuron_splitting,
                                      max_schedule_passes=caps.max_schedule_passes)
    frag = collect_noc_fragments(softcores=vehicle.specs, core_types=types, census=vehicle.census,
                                 program=vehicle.program, **caps.layout_kwargs())
    with contextlib.redirect_stdout(io.StringIO()):
        mapping = build_hybrid_mapping_for_pipeline(vehicle.ir, platform, pipeline_config=vehicle.config)
    passes = [s for s in mapping.stages if s.kind == "neural"]
    return vehicle, comp, frag, mapping, passes


def _layers(vehicle, comp, passes):
    layer_of = {int(n.id): n.perceptron_index for n in vehicle.ir.get_neural_cores()}
    built = [tuple(sorted({layer_of[int(p["ir_node_id"])]
                           for ps in s.hard_core_mapping.soft_core_placements_per_hard_core for p in ps}))
             for s in passes]
    planned = [tuple(sorted(comp.units.layers[u] for u in range(a, b))) for a, b in comp.facts.phases]
    return built, planned


def _executed_bytes(passes, bits: int) -> int:
    return sum(params_bytes(CoreOccupancy.from_hard_core(core).cells_used, bits) for s in passes
               for i, core in enumerate(s.hard_core_mapping.cores) if i not in s.schedule_resident_cores)


class TestThePackersDivergeOnT053AtSixCores:
    """t0_53 at C 6 (x2.0): the layout packer neuron-splits a 128x64 tile of layer 2 and places 128x54 of it on the
    core that holds layer 3 (256x10), a 384x64-cell rectangle; the builder's hard-core packer keeps every layer on its
    own cores. One pass partition, two packings."""

    BITS = 5

    @pytest.mark.parametrize("step, planned, executed", [("floor", 11_040, 11_840), ("stateless", 52_000, 42_560)])
    def test_one_pass_partition_and_the_stated_byte_difference(self, step, planned, executed):
        vehicle, comp, frag, mapping, passes = _planes("t0_53", 2.0, step)
        assert int(vehicle.config["weight_bits"]) == self.BITS
        built, plan_layers = _layers(vehicle, comp, passes)
        assert built == plan_layers and len(passes) == comp.step[2] == 2           # the pass partition: equal
        assert comp.facts.programmed_bytes(lambda c: params_bytes(c, self.BITS)) == planned       # the planner's
        assert _executed_bytes(passes, self.BITS) == executed                                     # the record's
        assert [p.core_cells for p in frag.pass_programs] == [tuple(c for c, _ in p) for p in comp.facts.pass_cores]

    def test_the_floor_pins_different_cores_on_the_two_planes(self):
        _vehicle, comp, frag, _mapping, passes = _planes("t0_53", 2.0, "floor")
        assert list(comp.facts.pass_resident_cores) == [frozenset({0}), frozenset({0, 1, 2, 3})]
        assert [p.schedule_resident_cores for p in frag.pass_programs] == list(comp.facts.pass_resident_cores)
        assert [s.schedule_resident_cores for s in passes] == [frozenset({0}), frozenset({1, 2, 3, 4})]

    @pytest.mark.parametrize("step", ["floor", "stateless"])
    def test_each_executed_pass_records_its_difference_with_the_cause(self, step):
        _vehicle, comp, _frag, _mapping, passes = _planes("t0_53", 2.0, step)
        census = [getattr(s, SCHEDULE_RESIDENCY_PASS) for s in passes]
        assert [c.pass_index for c in census] == [0, 1]
        assert census[0].plane_differences() == ()                                   # layer 1 alone: one packing
        assert census[1].planned_cores == comp.facts.pass_cores[1]
        assert census[1].executed_cells == (2560, 8192, 8192, 8192, 8192)
        differences = " | ".join(census[1].plane_differences())
        assert "packer divergence" in differences and "(24576, 8192 x3, 1280)" in differences
        if step == "floor":
            assert "pinned cores differ" in differences
        assert all(c.planned_resident == comp.facts.pass_resident_cores[q] for q, c in enumerate(census))
        assert all(c.executed_resident == s.schedule_resident_cores for c, s in zip(census, passes))


class TestATypeContendedPhaseIsAdmittedOnBothPlanes:
    """t0_01 at its declared budget: two core types of 60 each, C 96, F 48 layer-granular tiles. Each layer packed
    alone (a unit) picks its best type; the whole program packed on the declared counts needs one more tile (49)."""

    def test_the_resident_program_gives_the_resident_census(self):
        _vehicle, comp, frag, mapping, passes = _planes("t0_01", 1.0, "floor")
        assert (comp.step, comp.plan.F, comp.budget) == ((0, 0, 1), 48, 96)
        assert [sum(t for _, t in p) for p in comp.facts.pass_cores] == [49]      # the bounded phase packing
        assert comp.facts.written_core_tiles == 0 and len(passes) == 1
        assert len(passes[0].schedule_resident_cores) == len(passes[0].hard_core_mapping.cores) == 49
        assert frag.pass_programs[0].schedule_resident_cores == frozenset(range(49))
        plan, = mapping.schedule_residency_plan
        assert (plan.written_tiles, plan.passes, plan.rotating_tiles) == (0, 1, 0)

    def test_the_stateless_plan_writes_one_tile_more_than_its_layer_granular_count(self):
        _vehicle, comp, frag, mapping, passes = _planes("t0_01", 1.0, "stateless")
        assert comp.facts.written_tiles == 48 and comp.facts.written_core_tiles == 49
        assert passes[0].schedule_resident_cores == frozenset() and len(passes[0].hard_core_mapping.cores) == 49
        plan, = mapping.schedule_residency_plan
        assert plan.written_tiles == max(0, plan.total_tiles - plan.core_budget + plan.rotating_tiles) == 48  # W
        census = getattr(passes[0], SCHEDULE_RESIDENCY_PASS)
        assert census.executed_tiles == 49 and census.planned_tiles == 48
        assert census.differences() == (
            "the pass packs into 49 tiles against the phase's 48 layer-granular tiles (packing the phase on the "
            "declared counts costs tiles its layers packed alone do not)",)
        assert frag.pass_programs[0].schedule_resident_cores == frozenset()


class TestTheFitIsThePhysicalPool:
    """The executed program is admitted against the declared pool (``check_pinned_fit``), the physical bound, never
    against the type-blind planning budget C or a layer-granular count."""

    def test_a_program_within_the_planning_budget_packed_beyond_it_is_resident_on_both_planes(self):
        """t0_11 at x0.6: two types of 36, C = 57 (0.8 x 72), F = 50; the phase packs into 60 cores, which the pool
        holds, so the resident census (r 0, W 0, every core pinned) is executed, as U1 deploys it."""
        _vehicle, comp, frag, mapping, passes = _planes("t0_11", 0.6, "floor")
        assert (comp.step, comp.plan.F, comp.budget) == ((0, 0, 1), 50, 57)
        assert len(passes) == 1 and len(passes[0].hard_core_mapping.cores) == 60
        assert len(passes[0].schedule_resident_cores) == 60 and comp.facts.written_core_tiles == 0
        assert frag.pass_programs[0].schedule_resident_cores == frozenset(range(60))
        assert mapping.schedule_residency_plan[0].written_tiles == 0

    def test_the_stateless_plan_pins_no_core_where_a_phase_outgrows_the_planning_budget(self):
        """Review round 3: the stateless plan is the reload baseline (r = C, no pinned budget): every core is
        rewritten each period on both planes, also where the phase packed on the declared counts outgrows C
        (t0_11 at x0.6: 60 cores packed against C 57; the forced (60 - 57)^+ would pin 3 and understate reload)."""
        _vehicle, comp, frag, mapping, passes = _planes("t0_11", 0.6, "stateless")
        assert comp.step[0] == comp.budget == 57 and comp.plan.F == 50
        assert [sum(t for _, t in p) for p in comp.facts.pass_cores] == [60]
        assert comp.facts.pass_resident_cores == (frozenset(),) and comp.facts.written_core_tiles == 60
        assert frag.pass_programs[0].schedule_resident_cores == frozenset()
        assert len(passes) == 1 and passes[0].schedule_resident_cores == frozenset()
        assert len(passes[0].hard_core_mapping.cores) == 60
        assert mapping.schedule_residency_plan[0].written_tiles == 50                    # the plan's W = F - C + r

    def test_more_physical_cores_than_planned_tiles_is_not_a_refusal(self):
        check_pinned_fit([[((8, 8),), ((8, 8),), ((8, 8),)]], [frozenset()], {(8, 8): 4})

    def test_the_composition_refuses_by_name_what_the_builder_refuses(self):
        """The shape-only composition runs the builder's fit check on its own phase packing: two 2-tile layers at
        C 3 whose phases each pack into three 1-tile cores cannot hold the pins of the other pass on 3 cores."""
        units = ResidencyUnits(tiles=(2, 2), live=(1, 0), forced=(), members=((0,), (1,)), layers=(0, 1),
                               segments=(0, 0))

        def pack_phase(members):
            return ((64, 1),) * 3, (((8, 8),),) * 3

        compose_units(units, ResidencySelection("floor"), 3, max_schedule_passes=8)          # unpacked: plans
        with pytest.raises(ResidencyPlanError, match="declared pool") as caught:
            compose_units(units, ResidencySelection("floor"), 3, max_schedule_passes=8, pack_phase=pack_phase,
                          pool={(8, 8): 3})
        assert isinstance(caught.value, ToolchainRefusal)

    def test_a_pass_the_declared_pool_cannot_place_refuses_by_name(self):
        with pytest.raises(ResidencyPlanError, match="declared pool") as caught:
            check_pinned_fit([[((16, 8),), ((8, 8),)]], [frozenset()], {(8, 8): 4})
        assert isinstance(caught.value, ToolchainRefusal)
