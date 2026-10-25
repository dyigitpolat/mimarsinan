"""[W03] The plan census states the carry the executed program moves: one crossing rule on every plane.

Tile-granular units cut inside a layer, so a carried wire can live across several pass boundaries (a layer's input,
read by its tile groups in later passes). The record census and the candidate census count such a wire once per
direction: read out after its producing pass and re-injected before its last consuming pass
(``period_census.boundary_crossings``). The plan census of a composition states the same crossings from its units'
carried spans. The planner's own carry, the step function's objective (E85), stays the live set at every cut in both
directions, and a plan stated by tile counts alone (the E85 fixture, the ladder's units) keeps that statement.
"""

from __future__ import annotations

import random

import pytest

from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import boundary_widths_of, mapping_program_passes
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.support.period_census import boundary_crossings
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.residency_units import (
    facts_period_census,
    plan_period_census,
    plan_period_facts,
    residency_steps,
    residency_units,
)

from . import certified as CF
from . import vehicles as V

BITS, BPE = 8, 2
CORE = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=8)]


def _carry(census):
    """Per boundary with a nonzero carry, (bytes read out, bytes injected)."""
    return {b.after_pass: (b.carry_out_bytes, b.carry_in_bytes) for b in census.boundaries
            if b.carry_out_bytes or b.carry_in_bytes}


def _fan(sizes):
    """``sizes[L]`` one-tile specs in layer L, each reading every spec of layer L - 1 (one segment)."""
    specs, nodes, previous = [], [], []
    for layer, size in enumerate(sizes):
        mine = []
        for _ in range(size):
            i = len(specs)
            specs.append(LayoutSoftCoreSpec(input_count=16, output_count=8, perceptron_index=layer, segment_id=0,
                                            latency_tag=layer, name=f"sc{i}"))
            nodes.append(CutNode(core_id=i, latency=layer, sources=tuple(previous), out_width=8))
            mine.append(i)
        previous = mine
    return specs, nodes


class TestOneCrossingRule:
    def test_a_wire_is_read_out_once_and_injected_once_however_many_boundaries_it_crosses(self):
        assert boundary_crossings([(64, 0, 3), (64, 1, 3), (64, 2, 3)]) == ((0, 64, 0), (1, 64, 0), (2, 64, 192))

    def test_the_record_and_candidate_rule_is_this_rule(self):
        rng = random.Random(7)
        for _ in range(200):
            spans = []
            for _ in range(rng.randint(0, 6)):
                produced = rng.randint(0, 5)
                spans.append((rng.randint(1, 9), produced, produced + rng.randint(1, 4)))
            assert boundary_widths_of(spans) == boundary_crossings(spans)


class TestTheUnitsCarryTheirSpans:
    def test_the_spans_are_the_cut_s_carried_wires_on_units(self):
        specs, nodes = _fan([1, 3, 1])
        units = residency_units(specs, nodes, CORE, segments=[0] * len(specs))
        assert units.members == ((0,), (1,), (2,), (3,), (4,))
        cut = PassCut.over(nodes, [list(m) for m in units.members])
        assert units.spans == tuple((w.width, w.produced_in, w.last_consumed_in) for w in cut.carried)
        # spec 0 is read by the three specs of layer 1; each of those by the last spec
        assert units.spans == ((8, 0, 3), (8, 1, 4), (8, 2, 4), (8, 3, 4))

    def test_one_unit_per_pass_crosses_as_the_record_counts(self):
        specs, nodes = _fan([1, 3, 1])
        units = residency_units(specs, nodes, CORE, segments=[0] * len(specs))
        step = residency_steps(units.tiles, units.live, 1).step(0)
        facts = plan_period_facts(units.tiles, units.live, 1, step, unit_cores=units.cores, unit_spans=units.spans)
        assert facts.phases == ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))
        assert facts.boundary_crossings == ((8, 0), (8, 0), (8, 8), (8, 24))
        census = facts_period_census(facts, bytes_of_cells=int, tile_cells=0, bytes_per_element=BPE)
        assert _carry(census) == {0: (16, 0), 1: (16, 0), 2: (16, 16), 3: (16, 48)}
        # the planner's carry stays the live set at every cut, both ways (E85): 8, 16, 24 and 24 elements live
        assert facts.boundary_elements == (8, 16, 24, 24) and facts.carried_elements == 2 * 72 == step[1]

    def test_a_plan_stated_by_tile_counts_keeps_the_live_set_each_way(self):
        census = plan_period_census([1, 1, 1], [3, 4, 0], 1, (1, 14, 3), bytes_of_cells=int, tile_cells=4,
                                    bytes_per_element=BPE)
        assert _carry(census) == {0: (6, 6), 1: (8, 8)}


def _builder_census(ir, count, plan):
    comp = CF.builder_composition(ir, {"cores": V.cores(count), "allow_scheduling": True,
                                       "max_schedule_passes": 64}, plan)
    return comp, facts_period_census(comp.facts, forced=comp.units.forced,
                                     bytes_of_cells=lambda cells: params_bytes(cells, BITS), tile_cells=0,
                                     bytes_per_element=BPE)


class TestThePlanCensusEqualsTheRecordCensus:
    """On the seeded LIF chain (units of 3, 3, 2, 2 and 1 tiles in layers) at every budget below the fit, the
    executed floor and stateless steps: the plan census's carry per boundary equals the record's, and somewhere a wire
    lives across several boundaries, where the live set each way would state more than the program moves."""

    @pytest.mark.parametrize("plan", [{"step": "floor"}, {"step": "stateless"}])
    def test_at_every_budget_below_the_fit(self, plan):
        ir = V.ir_of(V.chain_repr(seed=0))
        multi = 0
        for count in range(1, 11):
            hybrid = V.build(ir, count, plan, max_schedule_passes=64)
            record = period_census(mapping_program_passes(hybrid), weight_bits=BITS,
                                   capacity={(V.MAX_AXONS, V.MAX_NEURONS): count}, bytes_of=lambda w: w * BPE).census
            comp, census = _builder_census(ir, count, plan)
            assert len(census.passes) == len(record.passes) == len(V.neural_passes(hybrid))
            assert _carry(census) == _carry(record), count
            live_each_way = {q: (v * BPE, v * BPE) for q, v in enumerate(comp.facts.boundary_elements) if v}
            multi += live_each_way != _carry(record)
        assert multi > 0                                                     # the regression bites
