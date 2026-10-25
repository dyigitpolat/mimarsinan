"""[W03] The store front: the residency front in one vectorized step per position (tile-granular programs have
hundreds of units), the pointer front's phases and tie order, a per-segment pass cap, and the step function walked
on demand (the floor needs only its first step)."""

from __future__ import annotations

import itertools
import random

import pytest

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.mapping_structure import ResidencyPlanError
from mimarsinan.mapping.support.residency_front_store import front_mincarry_store, front_store
from mimarsinan.mapping.support.schedule.residency_front import (
    Q_SPAN,
    binding_cap,
    front_mincarry_py,
    phase_partition,
    plan_steps,
    q_fit,
)
from mimarsinan.mapping.support.schedule.residency_units import plan_period_facts, residency_steps


def _reference_phases(sizes, live, C, r, qmax, forced):
    """The pointer front of E85 (each kept tuple with its start and entry, candidates stably sorted by (e, c, q)),
    followed back from the last position's minimal (carry, phases): the phases the planner executed before W03."""
    n = len(sizes)
    prefix = [0]
    for s in sizes:
        prefix.append(prefix[-1] + s)
    cuts, starts, k, last = sorted(forced), [0] * (n + 1), 0, 0
    for j in range(1, n + 1):
        while k < len(cuts) and cuts[k] < j:
            last, k = cuts[k], k + 1
        starts[j] = last
    fronts = [[] for _ in range(n + 1)]
    fronts[0] = [(0, 0, 0, -1, -1)]
    for j in range(1, n + 1):
        cand, cut = [], 2 * live[j - 1] if j < n else 0
        for i in range(j - 1, starts[j] - 1, -1):
            width = prefix[j] - prefix[i]
            if width > C:
                break
            for kk, (e, c, q, _, _) in enumerate(fronts[i]):
                if e + max(0, width - r) <= C - r and q + 1 <= qmax:
                    cand.append((e + max(0, width - r), c + cut, q + 1, i, kk))
        cand.sort(key=lambda t: t[:3])
        keep = []
        for t in cand:
            if not any(x[0] <= t[0] and x[1] <= t[1] and x[2] <= t[2] for x in keep):
                keep.append(t)
        fronts[j] = keep
    if not fronts[n]:
        return None
    j, kk = n, min(range(len(fronts[n])), key=lambda x: (fronts[n][x][1], fronts[n][x][2]))
    out = []
    while j > 0:
        _, _, _, i, prev = fronts[j][kk]
        out.append((i, j))
        j, kk = i, prev
    return tuple(reversed(out))


def _segment_phases(phases, forced, n):
    bounds = [0, *sorted(set(forced)), n]
    return [sum(1 for a, _ in phases if lo <= a < hi) for lo, hi in zip(bounds, bounds[1:])]


def _brute(sizes, live, C, r, qmax, forced, cap=None):
    """Every contiguous partition under the planner's constraints (and at most ``cap`` phases per segment)."""
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
        if cap is not None and max(_segment_phases(phases, forced, n)) > cap:
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
    return sizes, live, forced, C


class TestTheStoreEqualsThePointerFront:
    @pytest.mark.parametrize("seed", range(4))
    def test_carry_phases_and_the_executed_phases_on_seeded_programs(self, seed):
        rng = random.Random(seed)
        for _ in range(400):
            sizes, live, forced, C = _program(rng, 16)
            qmax = q_fit(sizes, C, forced) + rng.choice([0, 1, 2, Q_SPAN])
            for r in sorted({0, rng.randint(0, C), C}):
                want = front_mincarry_py(sizes, live, C, r, qmax, forced)
                assert front_mincarry_store(sizes, live, C, r, qmax, forced) == want
                assert phase_partition(sizes, live, C, r, qmax, forced) == _reference_phases(
                    sizes, live, C, r, qmax, forced)

    def test_the_store_holds_every_position_front_in_position_order(self):
        store = front_store([1, 2, 1, 1], [3, 0, 2, 0], 4, 1, 6, (2,))
        positions = [int(p) for p in store.position]
        assert positions == sorted(positions) and store.off[0] == 0 and store.off[-1] == len(positions)
        assert all(store.off[j] <= store.off[j + 1] for j in range(len(store.off) - 1))


class TestASegmentPassCap:
    @pytest.mark.parametrize("seed", range(3))
    def test_the_capped_front_equals_a_capped_brute_force(self, seed):
        rng = random.Random(10 + seed)
        for _ in range(300):
            sizes, live, forced, C = _program(rng)
            qmax = q_fit(sizes, C, forced) + Q_SPAN
            cap = rng.randint(1, 4)
            for r in sorted({0, rng.randint(0, C), C}):
                want = _brute(sizes, live, C, r, qmax, forced, cap)
                assert front_mincarry_store(sizes, live, C, r, qmax, forced, segment_cap=cap) == want
                phases = front_store(sizes, live, C, r, qmax, forced, segment_cap=cap).path()
                assert (phases is None) == (want is None)
                if phases is not None:
                    assert max(_segment_phases(phases, forced, len(sizes))) <= cap

    def test_a_cap_that_cannot_bind_leaves_the_plan_as_it_was(self):
        rng = random.Random(20)
        for _ in range(200):
            sizes, live, forced, C = _program(rng)
            qmax = q_fit(sizes, C, forced) + Q_SPAN
            if binding_cap(sizes, C, forced, qmax, len(sizes)) is None:
                capped = plan_steps(sizes, live, C, forced=forced, segment_cap=len(sizes))
                assert capped.as_ra_dict() == plan_steps(sizes, live, C, forced=forced).as_ra_dict()

    def test_the_floor_under_a_binding_cap_is_the_smallest_r_with_a_capped_plan(self):
        rng = random.Random(21)
        checked = 0
        for _ in range(300):
            sizes, live, forced, C = _program(rng)
            qmax = q_fit(sizes, C, forced) + Q_SPAN
            cap = rng.randint(1, 3)
            plan = plan_steps(sizes, live, C, forced=forced, segment_cap=cap)
            feasible = [r for r in range(C + 1) if _brute(sizes, live, C, r, qmax, forced, cap) is not None]
            if not feasible:
                assert plan is None
                continue
            checked += 1
            r = feasible[0]
            assert plan.step(0) == (r, *_brute(sizes, live, C, r, qmax, forced, cap))
            assert plan.stateless == _brute(sizes, live, C, C, qmax, forced, cap)
        assert checked > 100

    def test_the_planned_phases_keep_the_cap_and_the_composition_never_exceeds_it(self):
        tiles, live = [1] * 9, [1] * 8 + [0]
        plan = residency_steps(tiles, live, 3, max_passes=3)
        facts = plan_period_facts(tiles, live, 3, plan.step(0), max_passes=3)
        assert len(facts.phases) == 3 == plan.step(0)[2]

    def test_a_segment_that_needs_more_passes_than_the_cap_refuses_by_name(self):
        with pytest.raises(ResidencyPlanError, match="max_schedule_passes = 2 per segment") as caught:
            residency_steps([2, 2, 2], [0, 0, 0], 2, max_passes=2)
        assert isinstance(caught.value, ToolchainRefusal) and "{0: 3}" in str(caught.value)


class TestTheStepFunctionOnDemand:
    @staticmethod
    def _counted():
        calls = []

        def front(*args, **kwargs):
            calls.append(args[3])
            return front_mincarry_py(*args, **kwargs)
        return front, calls

    def test_the_floor_probes_only_the_bisection_and_the_stateless_plan(self):
        tiles, live = [1] * 24, [(7 * i) % 5 for i in range(23)] + [0]
        front, calls = self._counted()
        plan = plan_steps(tiles, live, 8, front=front)
        floor = plan.step(0)
        before = len(set(calls))
        assert before <= 2 + (8).bit_length() + 1
        steps = plan.steps
        assert steps[0] == floor and len(set(calls)) > before and len(steps) >= 2
        assert [plan.step(i) for i in range(len(steps))] == list(steps) and plan.step(len(steps)) is None

    def test_on_demand_steps_equal_the_whole_walk(self):
        rng = random.Random(30)
        for _ in range(100):
            sizes, live, forced, C = _program(rng, 12)
            lazy, eager = plan_steps(sizes, live, C, forced=forced), plan_steps(sizes, live, C, forced=forced)
            assert [lazy.step(i) for i in range(len(eager.steps) + 1)] == list(eager.steps) + [None]
