"""Toolchain residency units: layer identity, unit tiles, live sets at layer cuts, the step function, the census."""

from __future__ import annotations

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.support.schedule.pass_cut import CutNode
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_rules import ProgramGraph
from mimarsinan.mapping.support.schedule.residency_composition import compose_residency_program
from mimarsinan.mapping.support.schedule.residency_front import plan_steps
from mimarsinan.mapping.packing.residency_tiles import core_type_tiles, unit_tile_count
from mimarsinan.mapping.support.schedule.residency_units import (
    cut_nodes_from_census,
    pin_passes,
    pinned_tiles,
    plan_period_census,
    plan_period_facts,
    residency_steps,
    residency_units,
    stateless_step,
)

from mimarsinan.mapping.support.period_census import PassPeriodCensus

from .residency_fixture import load_calls

CORE = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=4)]


def _spec(i, layer, segment=0, axons=16, neurons=8):
    return LayoutSoftCoreSpec(input_count=axons, output_count=neurons, perceptron_index=layer,
                              segment_id=segment, latency_tag=i, name=f"sc{i}")


def _chain(layers, segments=None, skip=()):
    """One spec per entry of ``layers``; spec i reads spec i-1 (same segment), plus ``skip`` edges (src, dst)."""
    segments = segments or [0] * len(layers)
    specs = [_spec(i, layer, segments[i]) for i, layer in enumerate(layers)]
    nodes = []
    for i in range(len(specs)):
        sources = {i - 1} if i and segments[i - 1] == segments[i] else set()
        sources |= {a for a, b in skip if b == i}
        nodes.append(CutNode(core_id=i, latency=i, sources=tuple(sorted(sources)), out_width=8))
    return specs, nodes, segments


class TestUnits:
    def test_one_unit_per_layer_in_program_order(self):
        specs, nodes, seg = _chain([0, 0, 1, 2, 2, 2])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.layers == (0, 1, 2)
        assert units.members == ((0, 1), (2,), (3, 4, 5))
        assert units.tiles == (2, 1, 3)
        assert units.forced == ()

    def test_live_sets_are_the_cut_widths_at_layer_cuts(self):
        specs, nodes, seg = _chain([0, 1, 2], skip=((0, 2),))
        units = residency_units(specs, nodes, CORE, segments=seg)
        # after layer 0: its output (8) feeds layers 1 and 2; after layer 1: its output plus the skip
        assert units.live == (8, 16, 0)

    def test_a_segment_boundary_is_a_forced_cut_with_no_counted_live_set(self):
        specs, nodes, seg = _chain([0, 1, 2, 3], segments=[0, 0, 1, 1])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.forced == (2,)
        assert units.live == (8, 0, 8, 0)

    def test_a_layer_order_that_is_not_a_program_order_plans_in_the_unit_graphs_order(self):
        # [F2] layer 0 holds specs 0 and 2, and spec 2 reads spec 1 of layer 1: no pass order runs layer 0 first, so
        # the units run in a topological order of the unit graph (a pass may hold part of a layer); a strongly
        # connected set of units would be one unit (``test_unit_order``)
        specs, nodes, seg = _chain([0, 1, 0])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (1,), (2,)) and units.layers == (0, 1, 0)
        assert units.live == (8, 8, 0)

    def test_a_core_without_a_layer_joins_the_layer_it_feeds(self):
        specs, nodes, seg = _chain([0, None, 1])
        units = residency_units(specs, nodes, CORE, segments=seg)
        assert units.members == ((0,), (1, 2))

    def test_a_core_with_no_layer_identity_refuses_by_name(self):
        specs, nodes, seg = _chain([0, None])
        with pytest.raises(ResidencyPlanError) as caught:
            residency_units(specs, nodes, CORE, segments=seg)
        assert isinstance(caught.value, ToolchainRefusal) and isinstance(caught.value, ValueError)

    def test_unit_tiles_are_the_layer_packed_alone_with_an_unbounded_pool(self):
        specs = [LayoutSoftCoreSpec(input_count=8, output_count=4, perceptron_index=0, latency_tag=0)
                 for _ in range(10)]
        # two 8x4 softcores of one layer share a 16x8 core diagonally: ten need five, more than the pool of 4
        assert unit_tile_count(specs, CORE, allow_coalescing=False, allow_splitting=False) == 5

    def test_the_census_plane_reads_sources_from_the_wire_census(self):
        class Census:
            pair_wires = {(0, 1): 3, (1, 2): 8, (0, 2): 1}

        specs = [_spec(0, 0), _spec(1, 1), _spec(2, 2, segment=1)]
        nodes = cut_nodes_from_census(specs, Census())
        assert [n.sources for n in nodes] == [(), (0,), ()]   # the cross-segment pairs are host boundaries
        assert [n.out_width for n in nodes] == [8, 8, 8]


class TestPhysicalTiles:
    """A fused core (axons summed over cores of one type) occupies its components, so units count physical tiles."""

    TYPES = [LayoutHardCoreType(1024, 512, 4), LayoutHardCoreType(512, 256, 4)]

    def test_a_declared_core_is_one_tile(self):
        assert core_type_tiles(512, 256, self.TYPES) == ((512, 256),)

    def test_a_fused_core_is_its_components(self):
        assert core_type_tiles(3072, 512, self.TYPES) == ((1024, 512),) * 3
        assert core_type_tiles(2048, 512, self.TYPES, components=[1024, 1024]) == ((1024, 512),) * 2

    def test_an_ambiguous_decomposition_refuses_by_name(self):
        types = [LayoutHardCoreType(256, 64, 1), LayoutHardCoreType(512, 64, 1)]
        with pytest.raises(ResidencyPlanError, match="decomposes"):
            core_type_tiles(1024, 64, types)


class TestStepFunction:
    def test_a_unit_above_the_budget_refuses_by_name(self):
        with pytest.raises(ResidencyPlanError):
            residency_steps([3, 9, 2], [1, 1, 0], 8)

    def test_it_equals_plan_steps_on_the_spread_cases(self):
        for call in load_calls("spread_31")[:8]:
            assert residency_steps(call.tiles, call.live, call.C).as_ra_dict() == call.expected


class TestPlanPeriodFacts:
    """Acceptance 6 at the facts level: plan step -> census -> r, W, Q and carry (S0's type joins at staging)."""

    @pytest.mark.parametrize("index", range(0, 31, 3))
    def test_a_step_round_trips_through_its_census(self, index):
        call = load_calls("spread_31")[index]
        plan = residency_steps(call.tiles, call.live, call.C)
        for step in list(plan.steps) + [stateless_step(plan)]:
            r, carry, Q = step
            facts = plan_period_facts(call.tiles, call.live, call.C, step)
            assert (facts.r, facts.Q, facts.carried_elements) == (r, Q, carry)
            assert facts.written_tiles == plan.F - call.C + r                  # W = F - C + r (RA)
            assert all(t - p <= r for t, p in zip(facts.pass_tiles, facts.pass_pinned))
            assert sum(facts.pass_pinned) == call.C - r
        floor = plan_period_facts(call.tiles, call.live, call.C, plan.steps[0])
        tile_bytes = 256 * 256 // 8
        assert floor.written_tiles * tile_bytes == (plan.F - call.C + plan.steps[0][0]) * tile_bytes

    def test_a_program_that_fits_gives_the_resident_census_at_the_floor(self):
        plan = plan_steps([2, 3], [4, 0], 8)
        facts = plan_period_facts([2, 3], [4, 0], 8, plan.steps[0])
        assert (facts.written_tiles, facts.Q, facts.carried_elements) == (0, 1, 0)

    def test_the_pin_budget_is_filled_in_program_order(self):
        assert pinned_tiles([5, 3, 4], r=3, C=7) == (3, 0, 1)       # forced (2, 0, 1), the spare 1 to pass 0
        assert pinned_tiles([5, 3, 4], r=4, C=7) == (3, 0, 0)       # forced (1, 0, 0), the spare 2 to pass 0
        assert pinned_tiles([5, 3, 4], r=7, C=7) == (0, 0, 0)

    def test_no_pinned_budget_pins_no_tile_however_a_pass_packs(self):
        """Review round 3: at r = C (the stateless plan, the reload baseline) the pinned budget C - r is 0, so no
        tile is pinned, also where a phase packed on the declared counts outgrows C (heterogeneous core types)."""
        assert pinned_tiles([9, 3], r=7, C=7) == (0, 0)
        assert pin_passes([[(10, 1)] * 9, [(10, 1)] * 3], r=7, C=7) == (frozenset(), frozenset())
        assert pinned_tiles([9, 3], r=6, C=7) == (3, 0)                   # a pinned budget: the forced (9 - 6)^+


class TestPlanPeriodCensus:
    """Acceptance 6 on S0's type: plan step -> ``PassPeriodCensus`` -> r, W, Q and carry (the carry through the
    census's carried bytes per boundary), the label carried verbatim, and W x TB programmed at the floor."""

    TILE_CELLS, BYTES_PER_ELEMENT = 256 * 256, 1

    def _census(self, call, step, **kw):
        return plan_period_census(call.tiles, call.live, call.C, step, bytes_of_cells=lambda cells: cells // 8,
                                  tile_cells=self.TILE_CELLS, bytes_per_element=self.BYTES_PER_ELEMENT, **kw)

    @pytest.mark.parametrize("index", range(0, 31, 3))
    def test_a_step_round_trips_through_s0s_census(self, index):
        call = load_calls("spread_31")[index]
        plan = residency_steps(call.tiles, call.live, call.C)
        for step in list(plan.steps) + [stateless_step(plan)]:
            facts = plan_period_facts(call.tiles, call.live, call.C, step)
            census = self._census(call, step, label="planned and priced (test)")
            assert isinstance(census, PassPeriodCensus) and census.label == "planned and priced (test)"
            assert PassPeriodCensus.from_dict(census.to_dict()) == census
            written = sum(p.cores - p.resident_cores for p in census.passes)
            carried = (census.carry_out_bytes + census.carry_in_bytes) // self.BYTES_PER_ELEMENT
            assert (written, len(census.passes), carried) == (facts.written_tiles, facts.Q, facts.carried_elements)
            assert (written, len(census.passes), carried) == (plan.F - call.C + step[0], step[2], step[1])
            if written:
                assert call.C - sum(p.resident_cores for p in census.passes) == step[0]          # r
            assert [b.carry_out_bytes for b in census.boundaries] == [b.carry_in_bytes for b in census.boundaries]
        floor = self._census(call, plan.steps[0])
        tile_bytes = self.TILE_CELLS // 8
        assert sum(p.payload_bytes for p in floor.passes) == (plan.F - call.C + plan.steps[0][0]) * tile_bytes

    def test_a_forced_cut_starts_a_segment_and_carries_no_boundary(self):
        tiles, live, C = [2, 2, 2], [5, 0, 0], 3
        census = plan_period_census(tiles, live, C, (3, 0, 3), forced=(2,), bytes_of_cells=int, tile_cells=4,
                                    bytes_per_element=2)
        assert [p.segment_index for p in census.passes] == [0, 0, 1]
        assert [(b.after_pass, b.carry_out_bytes, b.carry_in_bytes) for b in census.boundaries] == [(0, 10, 10)]

    def test_packed_cores_give_the_occupied_cell_bytes(self):
        tiles, live, C = [2, 1, 2], [5, 5, 0], 3
        plan = residency_steps(tiles, live, C)
        census = plan_period_census(tiles, live, C, plan.steps[0], unit_cores=TestProgrammedCells.UNIT_CORES,
                                    bytes_of_cells=lambda cells: 2 * cells, tile_cells=16, bytes_per_element=1)
        facts = plan_period_facts(tiles, live, C, plan.steps[0], unit_cores=TestProgrammedCells.UNIT_CORES)
        assert [p.written_cells_backing_store for p in census.passes] == [sum(c) for c in facts.written_cells]
        assert sum(p.payload_bytes for p in census.passes) == facts.programmed_bytes(lambda cells: 2 * cells) == 36
        assert [(p.cores, p.resident_cores) for p in census.passes] == [(2, 1), (1, 0), (2, 1)]

    def test_a_program_that_fits_is_resident_at_the_floor(self):
        plan = plan_steps([2, 3], [4, 0], 8)
        census = plan_period_census([2, 3], [4, 0], 8, plan.steps[0], bytes_of_cells=int, tile_cells=4,
                                    bytes_per_element=1)
        assert [(p.resident, p.payload_bytes) for p in census.passes] == [(True, 0)]

    def test_the_stateless_plan_of_a_fit_runs_one_pass_that_nothing_displaces(self):
        """At F <= C the stateless plan is one pass (Q = 1): no other pass rewrites its cores, so the executed
        rotation writes nothing per period, as the RA ``price`` reads a fit (resident) under either arm."""
        plan = plan_steps([2, 3], [4, 0], 8)
        assert stateless_step(plan) == (8, 0, 1)
        census = plan_period_census([2, 3], [4, 0], 8, stateless_step(plan), bytes_of_cells=int, tile_cells=4,
                                    bytes_per_element=1)
        assert [(p.resident, p.resident_cores, p.payload_bytes) for p in census.passes] == [(True, 5, 0)]

    def test_a_segmented_fit_rotates_its_segments_unless_the_floor_pins_them(self):
        """A forced cut makes the stateless plan of a fit two passes, each loaded from the first core: both rewrite
        every period; the floor pins every tile of both (the resident census)."""
        plan = residency_steps([2, 3], [0, 0], 8, forced=(1,))
        stateless = plan_period_census([2, 3], [0, 0], 8, stateless_step(plan), forced=(1,), bytes_of_cells=int,
                                       tile_cells=4, bytes_per_element=1)
        assert [(p.segment_index, p.payload_bytes) for p in stateless.passes] == [(0, 8), (1, 12)]
        floor = plan_period_census([2, 3], [0, 0], 8, plan.steps[0], forced=(1,), bytes_of_cells=int, tile_cells=4,
                                   bytes_per_element=1)
        assert [(p.resident, p.payload_bytes) for p in floor.passes] == [(True, 0), (True, 0)]


class TestProgrammedCells:
    """Programmed bytes, defined once: the occupied cells of every core a pass writes (its packed cores less the
    pinned ones), each core priced by the record's ``params_bytes``."""

    UNIT_CORES = (((10, 1), (6, 1)), ((9, 1),), ((4, 1), (3, 1)))

    def test_a_unit_carries_the_cells_and_tiles_of_its_packed_cores(self):
        specs = [LayoutSoftCoreSpec(input_count=8, output_count=4, perceptron_index=0, latency_tag=0)
                 for _ in range(3)]
        units = residency_units(specs, [CutNode(i, 0, (), 4) for i in range(3)], CORE, segments=[0, 0, 0])
        # two 8x4 softcores share a 16x8 core diagonally (its used rectangle 16x8), the third uses 8x4 of one; each
        # core is a tile group of the layer [W03], in order of its first spec
        assert units.members == ((0,), (1, 2)) and units.cores == (((32, 1),), ((128, 1),)) and units.tiles == (1, 1)

    def test_the_written_cells_are_the_cores_the_pinned_rule_leaves(self):
        tiles, live, C = [2, 1, 2], [5, 5, 0], 3
        plan = residency_steps(tiles, live, C)
        floor = plan_period_facts(tiles, live, C, plan.steps[0], unit_cores=self.UNIT_CORES)
        assert floor.phases == ((0, 1), (1, 2), (2, 3)) and floor.pass_pinned == (1, 0, 1)
        assert floor.pass_resident_cores == (frozenset({0}), frozenset(), frozenset({0}))   # largest payload
        assert floor.written_cells == ((6,), (9,), (3,))
        assert floor.written_core_tiles == floor.written_tiles == plan.F - C + plan.steps[0][0]
        assert floor.programmed_bytes(lambda cells: 2 * cells) == 36
        merged = plan_period_facts(tiles, live, C, plan.steps[1], unit_cores=self.UNIT_CORES)
        assert merged.written_cells == ((6, 9), (4, 3))
        stateless = plan_period_facts(tiles, live, C, stateless_step(plan), unit_cores=self.UNIT_CORES)
        assert stateless.written_cells == ((10, 6, 9), (4, 3))                           # every core rewritten

    def test_a_fused_core_writes_its_component_tiles(self):
        """Review round 3 (V19): W_exec counts the physical tiles of the written cores, a fused core its
        components, not the cores."""
        plan = residency_steps([3], [0], 3)
        facts = plan_period_facts([3], [0], 3, stateless_step(plan), unit_cores=(((100, 2), (50, 1)),))
        assert facts.pass_resident_cores == (frozenset(),)
        assert facts.written_cells == ((100, 50),) and facts.written_core_tiles == 3

    def test_facts_stated_by_tile_counts_hold_no_programmed_cells(self):
        facts = plan_period_facts([2, 1, 2], [5, 5, 0], 3, (1, 20, 3))
        with pytest.raises(ResidencyPlanError, match="no programmed cells"):
            facts.programmed_bytes(int)

    def test_a_phase_packed_jointly_writes_at_most_the_planned_tiles(self):
        # layers 0 and 1 read the input (latency 0) and share one 16x8 core when packed as one pass; layer 2 (two
        # 8x8 specs, two tile groups [W03]) reads both. F = 4 planned tiles at C = 2; the stateless plan's first
        # phase holds layers 0 and 1, which pack into one core, so it writes 3 tiles, not W = 4.
        specs = [LayoutSoftCoreSpec(input_count=8, output_count=4 if i < 2 else 8, perceptron_index=min(i, 2),
                                    latency_tag=int(i >= 2), segment_id=0, name=f"s{i}") for i in range(4)]
        nodes = [CutNode(0, 0, (), 4), CutNode(1, 0, (), 4), CutNode(2, 1, (0, 1), 8), CutNode(3, 1, (0, 1), 8)]
        program = ProgramGraph(ids=(0, 1, 2, 3), specs=(0, 1, 2, 3),
                               producers=(frozenset(), frozenset(), frozenset({0, 1}), frozenset({0, 1})))
        comp = compose_residency_program(specs, nodes, [LayoutHardCoreType(16, 8, 2)], segments=[0] * 4,
                                         residency_plan={"step": "stateless"}, program=program,
                                         allow_coalescing=False, allow_splitting=False, max_schedule_passes=8)
        facts = comp.facts
        assert comp.units.members == ((0,), (1,), (2,), (3,)) and comp.units.layers == (0, 1, 2, 2)
        assert facts.phases == ((0, 2), (2, 4)) and facts.pass_tiles == (2, 2)
        assert facts.pass_cores == (((128, 1),), ((64, 1), (64, 1)))
        assert facts.written_cells == ((128,), (64, 64))
        assert facts.written_core_tiles == 3 < facts.written_tiles == 4                    # W_exec < W
