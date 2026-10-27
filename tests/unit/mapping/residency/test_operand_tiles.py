"""[F3] The residency planner on programs with dynamic operand banks (N2): it states which cores hold operand banks
and prices their tiles as rewritten every inference.

An operand bank's payload is written every inference (N2 design section 2.7), so the one pinned rule never pins a core
that holds one (``pinned_cores(operand_cores=...)``). Before F3 no caller passed those cores, so the builder refused
every residency plan for a program with operand banks. The planner now states, per packed core, the cells of the
operand-bank regions it holds (``ResidencyUnits.producer``, a phase's packing ``pack_specs``, the builder's hard
cores), and prices them:

- a pass's operand tiles are never among its pins (``pinned_tiles``), so W counts them every period;
- a phase's operand tiles rotate, so a step at r admits only phases whose operand tiles are at most r (the fronts):
  mincarry(r) stays non-increasing and feasibility monotone in r;
- the plan plane's census states each pass's producer cells (written every inference, resident or not) apart from
  its static cells, as the record plane does (``mapping_program_passes``).

Without operand banks every function is unchanged (seeded identity)."""

from __future__ import annotations

import itertools
import random

import pytest

from mimarsinan.mapping.layout.layout_types import LayoutHardCoreType, LayoutSoftCoreSpec
from mimarsinan.mapping.packing.canonical import OPERAND_ROLE
from mimarsinan.mapping.packing.residency_tiles import pack_specs
from mimarsinan.mapping.support.residency_front_store import front_mincarry_store, front_store
from mimarsinan.mapping.support.schedule.pass_cut import CutNode
from mimarsinan.mapping.support.schedule.residency_front import (
    FRONTS,
    Q_SPAN,
    phase_partition,
    plan_steps,
    q_fit,
)
from mimarsinan.mapping.support.schedule.residency_units import (
    pin_passes,
    pinned_tiles,
    plan_period_facts,
    residency_steps,
    residency_units,
    stateless_step,
    units_period_census,
)


def _brute(sizes, live, C, r, qmax, forced, operand):
    """Every contiguous partition under the planner's constraints, a phase's operand tiles at most r."""
    n, best = len(sizes), None
    for mask in itertools.product((0, 1), repeat=n - 1):
        cuts = [i + 1 for i, b in enumerate(mask) if b]
        if any(p not in cuts for p in forced):
            continue
        bounds = [0, *cuts, n]
        phases = list(zip(bounds, bounds[1:]))
        widths = [sum(sizes[a:b]) for a, b in phases]
        if max(widths) > C or len(phases) > qmax or sum(max(0, R - r) for R in widths) > C - r:
            continue
        if any(sum(operand[a:b]) > r for a, b in phases):
            continue
        key = (sum(2 * live[p - 1] for p in cuts), len(phases))
        best = key if best is None or key < best else best
    return best


def _program(rng, n_hi=10):
    n = rng.randint(1, n_hi)
    sizes = [rng.choice([1, 1, 1, 2, 3]) if rng.random() < 0.85 else rng.randint(1, 8) for _ in range(n)]
    live = [rng.choice([0, 0, 1, 2, 4, 7]) for _ in range(n - 1)] + [0]
    forced = sorted(rng.sample(range(1, n), rng.randint(0, min(3, n - 1)))) if n > 1 else []
    C = rng.randint(max(sizes), max(max(sizes), sum(sizes) + 2))
    operand = [rng.choice([0, 0, s]) if rng.random() < 0.5 else rng.randint(0, s) for s in sizes]
    return sizes, live, forced, C, operand


class TestThePinnedRuleOnOperandTiles:
    def test_an_operand_tile_is_never_a_pin_and_the_spare_goes_to_static_tiles(self):
        assert pinned_tiles([5, 3, 4], r=3, C=7) == (3, 0, 1)
        # pass 0 holds 3 operand tiles: its forced 2 pins are its 2 static tiles, the spare pin goes to pass 1
        assert pinned_tiles([5, 3, 4], r=3, C=7, operand=(3, 0, 0)) == (2, 1, 1)
        assert pinned_tiles([5, 3, 4], r=7, C=7, operand=(3, 0, 0)) == (0, 0, 0)

    def test_forced_pins_never_exceed_a_passes_static_tiles(self):
        assert pinned_tiles([5], r=2, C=7, operand=(4,)) == (1,)

    def test_without_operand_tiles_the_rule_is_unchanged(self):
        rng = random.Random(0)
        for _ in range(500):
            tiles = [rng.randint(1, 9) for _ in range(rng.randint(1, 6))]
            C = rng.randint(max(tiles), sum(tiles) + 3)
            r = rng.randint(0, C)
            assert pinned_tiles(tiles, r, C, operand=(0,) * len(tiles)) == pinned_tiles(tiles, r, C)

    def test_the_pinned_rule_never_pins_a_core_holding_an_operand_bank(self):
        cores = [(90, 1), (50, 1), (80, 1), (10, 1)]
        assert pin_passes([cores], r=1, C=5) == (frozenset({0, 1, 2, 3}),)
        assert pin_passes([cores], r=1, C=5, producer=[(90, 0, 0, 0)]) == (frozenset({1, 2, 3}),)
        assert pin_passes([cores], r=1, C=5, producer=[()]) == pin_passes([cores], r=1, C=5)


class TestTheStepFunctionPricesOperandTiles:
    @pytest.mark.parametrize("seed", range(3))
    def test_every_front_equals_a_brute_force_where_a_phases_operand_tiles_are_at_most_r(self, seed):
        rng = random.Random(100 + seed)
        for _ in range(300):
            sizes, live, forced, C, operand = _program(rng)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, Q_SPAN])
            for r in sorted({0, rng.randint(0, C), C}):
                want = _brute(sizes, live, C, r, qmax, forced, operand)
                assert front_mincarry_store(sizes, live, C, r, qmax, forced, operand=operand) == want
                for name in sorted(FRONTS):
                    assert FRONTS[name](sizes, live, C, r, qmax, forced, operand=operand) == want, name
                phases = phase_partition(sizes, live, C, r, qmax, forced, operand=operand)
                assert (phases is None) == (want is None)
                if phases is not None:
                    assert all(sum(operand[a:b]) <= r for a, b in phases)

    @pytest.mark.parametrize("seed", range(3))
    def test_without_operand_tiles_the_step_function_and_its_phases_are_unchanged(self, seed):
        rng = random.Random(200 + seed)
        for _ in range(200):
            sizes, live, forced, C, _operand = _program(rng, 14)
            zero = (0,) * len(sizes)
            plain = plan_steps(sizes, live, C, forced=forced)
            stated = plan_steps(sizes, live, C, forced=forced, operand=zero)
            assert (plain is None) == (stated is None)
            if plain is None:
                continue
            assert stated.as_ra_dict() == plain.as_ra_dict()
            qmax = plain.qmax
            for r in {0, C, rng.randint(0, C)}:
                assert front_store(sizes, live, C, r, qmax, forced, operand=zero).path() == front_store(
                    sizes, live, C, r, qmax, forced).path()

    @pytest.mark.parametrize("seed", range(2))
    def test_the_floor_admits_every_units_operand_tiles_and_the_function_is_monotone(self, seed):
        rng = random.Random(300 + seed)
        for _ in range(200):
            sizes, live, forced, C, operand = _program(rng, 12)
            plan = plan_steps(sizes, live, C, forced=forced, operand=operand)
            if plan is None:
                continue
            steps = plan.steps
            assert steps[0][0] >= max(operand)
            assert all(a[1] > b[1] and a[0] < b[0] for a, b in zip(steps, steps[1:]))
            for step in list(steps) + [stateless_step(plan)]:
                facts = plan_period_facts(sizes, live, C, step, forced=forced, unit_operand=operand)
                ops = [sum(operand[a:b]) for a, b in facts.phases]
                assert all(o <= step[0] for o in ops)
                assert all(p <= t - o for p, t, o in zip(facts.pass_pinned, facts.pass_tiles, ops))
                assert facts.written_tiles >= sum(operand)               # every operand tile written every period

    def test_a_program_that_fits_with_operand_tiles_rotates_them(self):
        """F <= C: one phase holds every tile, but its 2 operand tiles are never pinned, so the floor is r = 2 and
        writes exactly them every period (the closed form of a fit, r = 0 and W = 0, holds without operand tiles)."""
        assert plan_steps([2, 3], [4, 0], 8).steps == ((0, 0, 1),)
        plan = plan_steps([2, 3], [4, 0], 8, operand=(2, 0))
        assert plan.steps == ((2, 0, 1),) and plan.stateless == (0, 1)
        facts = plan_period_facts([2, 3], [4, 0], 8, plan.steps[0], unit_operand=(2, 0))
        assert facts.pass_pinned == (3,) and facts.written_tiles == 2

    def test_a_unit_whose_operand_tiles_exceed_every_r_below_it_moves_the_floor(self):
        tiles, live, C = [1, 4, 1], [3, 3, 0], 4
        assert residency_steps(tiles, live, C).steps[0][0] == 1
        plan = residency_steps(tiles, live, C, operand=(0, 3, 0))
        assert plan.steps[0][0] == 3


CORE = [LayoutHardCoreType(max_axons=16, max_neurons=8, count=8)]


def _operand_program():
    """Layer 0: two static 8x4 specs (one 16x8 core, diagonal); layer 1: an operand bank's two instances (bank 7),
    8x4 each, which share one core the same way; layer 2: a static 16x8 spec reading both."""
    specs = [LayoutSoftCoreSpec(input_count=8, output_count=4, perceptron_index=0, latency_tag=0, name="s0"),
             LayoutSoftCoreSpec(input_count=8, output_count=4, perceptron_index=0, latency_tag=0, name="s1"),
             LayoutSoftCoreSpec(input_count=8, output_count=4, perceptron_index=1, latency_tag=1, bank_id=7,
                                core_role=OPERAND_ROLE, name="o0"),
             LayoutSoftCoreSpec(input_count=8, output_count=4, perceptron_index=1, latency_tag=1, bank_id=7,
                                core_role=OPERAND_ROLE, name="o1"),
             LayoutSoftCoreSpec(input_count=16, output_count=8, perceptron_index=2, latency_tag=2, name="s2")]
    nodes = [CutNode(0, 0, (), 4), CutNode(1, 0, (), 4), CutNode(2, 1, (0,), 4), CutNode(3, 1, (1,), 4),
             CutNode(4, 2, (2, 3), 8)]
    return specs, nodes


class TestUnitsStateTheirOperandCores:
    def test_a_packed_core_states_the_cells_of_the_operand_bank_regions_it_holds(self):
        specs, _ = _operand_program()
        cores, tiles, producer = pack_specs(specs[2:4], CORE, allow_coalescing=False, allow_splitting=False)
        assert cores == ((128, 1),) and producer == (64,)            # two 8x4 regions on a 16x8 rectangle
        cores, tiles, producer = pack_specs(specs[:2], CORE, allow_coalescing=False, allow_splitting=False)
        assert cores == ((128, 1),) and producer == ()

    def test_the_units_state_each_cores_producer_cells_and_operand_tiles(self):
        specs, nodes = _operand_program()
        units = residency_units(specs, nodes, CORE, segments=[0] * 5)
        assert units.members == ((0, 1), (2, 3), (4,)) and units.tiles == (1, 1, 1)
        assert units.producer == ((0,), (64,), (0,)) and units.operand_tiles == (0, 1, 0)

    def test_a_program_without_operand_banks_states_none(self):
        specs, nodes = _operand_program()
        static = [LayoutSoftCoreSpec(input_count=s.input_count, output_count=s.output_count,
                                     perceptron_index=s.perceptron_index, latency_tag=s.latency_tag, name=s.name)
                  for s in specs]
        units = residency_units(static, nodes, CORE, segments=[0] * 5)
        assert units.producer == () and units.operand_tiles == ()

    def test_the_unit_census_writes_the_producer_cells_every_inference_and_never_keeps_an_operand_core(self):
        specs, nodes = _operand_program()
        units = residency_units(specs, nodes, CORE, segments=[0] * 5)
        plan = residency_steps(units.tiles, units.live, 2, forced=units.forced, operand=units.operand_tiles)
        assert plan.steps[0][0] >= 1
        for step in list(plan.steps) + [stateless_step(plan)]:
            facts = plan_period_facts(units.tiles, units.live, 2, step, forced=units.forced, unit_cores=units.cores,
                                      unit_operand=units.operand_tiles, unit_producer=units.producer)
            for producer, kept in zip(facts.pass_producer, facts.pass_resident_cores):
                assert not any(producer[i] for i in kept)                     # never pinned
            census = units_period_census(units, 2, step, bytes_of_cells=int, tile_cells=128, bytes_per_element=1)
            passes = census.passes
            assert [p.written_cells_producer for p in passes] == [sum(p) for p in facts.pass_producer]
            assert sum(p.written_cells_producer for p in passes) == 64       # once per inference, resident or not
            # every written core's static cells, the operand core's rectangle less its regions (64) among them
            assert [p.written_cells_backing_store for p in passes] == [sum(c) for c in facts.written_cells]
            if any(p.written_cells_backing_store for p in passes):
                holding = next(q for q, p in enumerate(passes) if p.written_cells_producer)
                assert 64 in facts.written_cells[holding]
