"""[W03] The plan census states the carry the executed program moves: one crossing rule on every plane.

Tile-granular units cut inside a layer, so a carried wire can live across several pass boundaries (a layer's input,
read by its tile groups in later passes). The record census and the candidate census count such a wire once per
direction: read out after its producing pass and re-injected before its last consuming pass
(``period_census.boundary_crossings``). The plan census of a composition states the same crossings from its units'
carried spans. The planner's own carry, the step function's objective (E85), stays the live set at every cut in both
directions, and a plan stated by tile counts alone (the E85 fixture, the ladder's units) keeps that statement.

[F2] The builder replays a carried producer once per delivery delay (``pass_timing``): a pass that reads a producer
of an earlier pass at k delays re-injects it k - 1 more times, which the record states (``delayed_input_slices``) and
the candidate states (``delayed_carry_wires``). The plan census states the same replays from its units' delayed
reads, so the three planes apply one crossing rule with one replay term at every boundary.
"""

from __future__ import annotations

import random

import pytest

from mimarsinan.deployment_record.build.payload_sizes import params_bytes
from mimarsinan.deployment_record.build.program_census import (
    boundary_widths_of,
    candidate_program_passes,
    mapping_program_passes,
)
from mimarsinan.deployment_record.build.write_census import period_census
from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.pass_timing import project_planned_passes
from mimarsinan.mapping.support.period_census import boundary_crossings, phase_crossings
from mimarsinan.mapping.support.schedule.pass_cut import CutNode, PassCut
from mimarsinan.mapping.support.schedule.residency_units import (
    facts_period_census,
    plan_period_census,
    plan_period_facts,
    residency_steps,
    residency_units,
)

from mimarsinan.search.problems.joint.candidate_fragments import StageSemantics

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


def _skip3():
    """Spec 0 -> spec 1 -> spec 2, and spec 2 reads spec 0 too (an identity skip): the fused segment raises spec 0 to
    its deepest reader's latency minus one, so spec 1 reads it one cycle late and spec 2 on time (delays -1 and 0)."""
    specs = [LayoutSoftCoreSpec(input_count=16, output_count=8, perceptron_index=i, segment_id=0, latency_tag=i,
                                name=f"sc{i}") for i in range(3)]
    nodes = [CutNode(0, 0, (), 8), CutNode(1, 1, (0,), 8), CutNode(2, 2, (0, 1), 8)]
    return specs, nodes


def _projected(units, phases, specs, nodes):
    """The crossings of ``phases`` by the candidate plane's rule: ``project_planned_passes`` on the passes' specs and
    the cut nodes' wires, its extra slices re-injected through ``boundary_crossings``."""
    members = [[i for u in range(a, b) for i in units.members[u]] for a, b in phases]
    phase_of = {u: q for q, (a, b) in enumerate(phases) for u in range(a, b)}
    wires = [(int(s), int(n.core_id)) for n in nodes for s in n.sources]
    segment = {i: s for m, s in zip(units.members, units.segments) for i in m}
    projection = project_planned_passes([int(s.output_count) for s in specs], members, wires,
                                        [segment[i] for i in range(len(specs))])
    replayed = [(w, q) for q, extra in enumerate(projection.extra_slices) for _p, w in extra]
    by = {p: (o, i) for p, o, i in boundary_crossings(
        [(w, phase_of[p], phase_of[c]) for w, p, c in units.spans if phase_of[p] != phase_of[c]], replayed)}
    return tuple(by.get(q, (0, 0)) for q in range(len(phases) - 1))


class TestThePlanPlaneStatesTheReplays:
    def test_a_producer_read_at_two_delays_in_one_pass_is_re_injected_once_more(self):
        specs, nodes = _skip3()
        units = residency_units(specs, nodes, CORE, segments=[0, 0, 0])
        assert units.delayed_reads == ((8, 0, 1, 0, -1), (8, 0, 2, 0, 0))
        # cut after spec 0: pass 1 reads spec 0 at two delays, so spec 0's width is re-injected twice
        assert phase_crossings(units.spans, ((0, 1), (1, 3)), units.delayed_reads) == ((8, 16),)
        assert phase_crossings(units.spans, ((0, 1), (1, 3))) == ((8, 8),)
        # one pass per spec: each pass reads spec 0 at one delay, no replay
        assert phase_crossings(units.spans, ((0, 1), (1, 2), (2, 3)), units.delayed_reads) == ((8, 0), (8, 16))
        assert _projected(units, ((0, 1), (1, 3)), specs, nodes) == ((8, 16),)

    def test_a_program_whose_readers_read_each_producer_at_one_delay_states_no_delayed_reads(self):
        specs, nodes = _fan([1, 3, 1])
        assert residency_units(specs, nodes, CORE, segments=[0] * len(specs)).delayed_reads == ()

    @pytest.mark.parametrize("seed", range(40))
    def test_the_replays_are_the_candidate_planes_projection_on_every_phase_partition(self, seed):
        rng = random.Random(seed)
        n = rng.randint(3, 12)
        specs = [LayoutSoftCoreSpec(input_count=16, output_count=rng.choice([4, 8]), perceptron_index=i, segment_id=0,
                                    latency_tag=i, name=f"sc{i}") for i in range(n)]
        nodes = [CutNode(i, i, tuple(sorted(a for a in range(i) if rng.random() < 0.35)), int(specs[i].output_count))
                 for i in range(n)]
        units = residency_units(specs, nodes, CORE, segments=[0] * n)
        for _ in range(6):
            cuts = sorted(rng.sample(range(1, len(units.members)), rng.randint(0, len(units.members) - 1)))
            bounds = [0, *cuts, len(units.members)]
            phases = tuple(zip(bounds, bounds[1:]))
            assert phase_crossings(units.spans, phases, units.delayed_reads) == _projected(units, phases, specs,
                                                                                           nodes)


#: (vehicle, step, cores) where the builder replays a carried producer (the replay probe of the residency unit).
REPLAYING = [("late_skip", "floor", 2), ("late_skip", "stateless", 3), ("skew3", "floor", 3), ("deep_skip", "floor", 5),
             ("skip", "floor", 2), ("random_38", "stateless", 2)]


def _vehicle(name):
    rep = V.random_rep(int(name.split("_")[1])) if name.startswith("random_") else V.branchy_repr(name)
    specs, census, program = V.layout_plane(rep)
    return specs, census, program, V.ir_of(rep)


class TestThePlanesAgreeWhereTheBuilderReplays:
    @pytest.mark.parametrize("name,step,count", REPLAYING)
    def test_the_record_plan_and_candidate_planes_state_one_carry_at_every_boundary(self, name, step, count):
        specs, census, program, ir = _vehicle(name)
        plan = {"step": step}
        hybrid = V.build(ir, count, plan, max_schedule_passes=64)
        passes = V.neural_passes(hybrid)
        assert sum(len(s.input_map) - len({int(x.node_id) for x in s.input_map}) for s in passes) > 0   # it bites
        capacity = {(V.MAX_AXONS, V.MAX_NEURONS): count}
        record = period_census(mapping_program_passes(hybrid), weight_bits=BITS, capacity=capacity,
                               bytes_of=lambda w: w * BPE).census
        comp, planned = _builder_census(ir, count, plan)
        frag = V.fragments(specs, census, count, plan, program=program, max_schedule_passes=64)
        candidate = period_census(candidate_program_passes(specs, frag, semantics=StageSemantics(), timesteps=V.T),
                                  weight_bits=BITS, capacity=capacity, bytes_of=lambda w: w * BPE).census
        assert len(record.passes) == len(planned.passes) == len(candidate.passes) == len(passes)
        assert _carry(planned) == _carry(record) == _carry(candidate)
        assert comp.units.delayed_reads
