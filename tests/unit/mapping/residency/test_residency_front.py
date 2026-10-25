"""The toolchain residency planner equals the RA planner (E85 port, E297 front) on the exported fixture."""

from __future__ import annotations

import itertools
import json
import random

import pytest

from mimarsinan.mapping.support.schedule.residency_front import (
    FRONTS,
    Q_SPAN,
    STORE_MIN_UNITS,
    front_mincarry_np,
    front_mincarry_py,
    phase_partition,
    plan_steps,
    q_fit,
    select_front,
)

from .residency_fixture import (
    FixtureDigestError,
    canonical,
    check_digest,
    fixture_bytes,
    load_calls,
    planner_calls,
)


def _agrees(call, front=None) -> bool:
    got = plan_steps(call.tiles, call.live, call.C, front=front)
    if call.expected is None:
        return got is None
    return got is not None and got.as_ra_dict() == call.expected


class TestFixtureIntegrity:
    @pytest.mark.parametrize("name, count", [("spread_31", 31), ("x10ta_3341", 3341), ("deep_9", 9)])
    def test_each_fixture_matches_its_recorded_digest(self, name, count):
        assert len(load_calls(name)) == count

    def test_a_one_entry_edit_fails_the_digest(self):
        data = fixture_bytes("spread_31")
        document = json.loads(data)
        assert canonical(document) == data          # the edit below is the only change
        document["calls"][7]["expected"]["steps"][0][1] += 1
        with pytest.raises(FixtureDigestError):
            check_digest("spread_31", canonical(document))


class TestFrontEqualsTheRaPlanner:
    def test_the_dispatched_planner_on_a_seeded_250_call_subset(self):
        calls = random.Random(0).sample(planner_calls(), 250)
        assert sum(c.expected is None for c in calls) > 0      # refusals are in the subset too
        assert [c.case for c in calls if not _agrees(c)] == []

    @pytest.mark.parametrize("branch", sorted(FRONTS))
    def test_each_branch_on_the_31_spread_cases(self, branch):
        calls = load_calls("spread_31")
        assert [c.case for c in calls if not _agrees(c, FRONTS[branch])] == []

    def test_the_dispatch_threshold(self):
        """[W03] The store front from 100 units (re-measured on tile-granular units); the vectorized front a branch."""
        from mimarsinan.mapping.support.residency_front_store import front_mincarry_store
        assert select_front(STORE_MIN_UNITS - 1) is front_mincarry_py
        assert select_front(STORE_MIN_UNITS) is front_mincarry_store
        assert FRONTS["vectorized"] is front_mincarry_np


@pytest.mark.slow
class TestFrontEqualsTheRaPlannerEverywhere:
    @pytest.mark.timeout(900)
    @pytest.mark.parametrize("branch", sorted(FRONTS))
    def test_each_branch_on_all_3372_calls(self, branch):
        assert [c.case for c in planner_calls() if not _agrees(c, FRONTS[branch])] == []

    @pytest.mark.timeout(900)
    @pytest.mark.parametrize("branch", sorted(FRONTS))
    def test_each_branch_on_the_9_deep_cases(self, branch):
        assert [c.case for c in load_calls("deep_9") if not _agrees(c, FRONTS[branch])] == []


def _seeded_calls():
    return random.Random(1).sample([c for c in planner_calls() if c.expected is not None], 200)


def _check_invariants(call, rs) -> None:
    """Over the r values ``rs`` (r*, r* - 1 and C always among them): feasibility monotone in r, r* the smallest
    feasible r, and every step and the stateless plan within [Q_fit, Q_fit + 8]."""
    plan = plan_steps(call.tiles, call.live, call.C)
    front = select_front(len(call.tiles))
    r_star = plan.steps[0][0]
    feasible = {r: front(call.tiles, call.live, call.C, r, plan.qmax) is not None
                for r in sorted({*rs, r_star, max(r_star - 1, 0), call.C})}
    flags = [feasible[r] for r in sorted(feasible)]
    assert flags == sorted(flags), call.case                     # monotone in r
    assert feasible[r_star]
    assert r_star == 0 or not feasible[r_star - 1], call.case    # the smallest feasible r
    assert plan.qmax == plan.q_fit + Q_SPAN
    for _, _, q in plan.steps + ((None, None, plan.stateless[1]),):
        assert plan.q_fit <= q <= plan.q_fit + Q_SPAN, call.case


class TestPlannerInvariants:
    """Acceptance 3: monotone feasibility, r* the smallest feasible r, Q within [Q_fit, Q_fit + 8]."""

    def test_on_200_seeded_calls_at_sampled_r(self):
        rng = random.Random(1)
        for call in _seeded_calls():
            _check_invariants(call, rng.sample(range(call.C + 1), min(16, call.C + 1)))

    def test_at_every_r_on_the_seeded_calls_of_a_small_budget(self):
        small = [c for c in _seeded_calls() if c.C <= 64]
        assert len(small) >= 25
        for call in small:
            _check_invariants(call, range(call.C + 1))

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_at_every_r_on_all_200_seeded_calls(self):
        for call in _seeded_calls():
            _check_invariants(call, range(call.C + 1))


def _brute_force(sizes, live, C, r, qmax, forced):
    """Every contiguous partition, filtered by the planner's constraints; the minimal (carry, phases)."""
    n, best = len(sizes), None
    for mask in itertools.product((0, 1), repeat=n - 1):
        cuts = [i + 1 for i, b in enumerate(mask) if b]
        if any(p not in cuts for p in forced):
            continue
        bounds = [0, *cuts, n]
        phases = [sum(sizes[a:b]) for a, b in zip(bounds, bounds[1:])]
        if max(phases) > C or len(phases) > qmax:
            continue
        if sum(max(0, R - r) for R in phases) > C - r:
            continue
        key = (sum(2 * live[p - 1] for p in cuts), len(phases))
        best = key if best is None or key < best else best
    return best


class TestForcedCuts:
    """The one generalization of the port: a segment boundary is a cut every phase respects."""

    def test_every_front_equals_a_brute_force_with_and_without_forced_cuts(self):
        rng = random.Random(2)
        for _ in range(150):
            n = rng.randint(2, 9)
            sizes = [rng.randint(1, 6) for _ in range(n)]
            live = [rng.randint(0, 9) for _ in range(n)]
            C = rng.randint(max(sizes), sum(sizes) + 2)
            forced = tuple(sorted(rng.sample(range(1, n), rng.randint(0, min(2, n - 1)))))
            r, qmax = rng.randint(0, C), rng.randint(1, n + 1)
            want = _brute_force(sizes, live, C, r, qmax, forced)
            for front in FRONTS.values():
                assert front(sizes, live, C, r, qmax, forced) == want

    def test_the_partition_attains_the_front_and_respects_every_constraint(self):
        rng = random.Random(3)
        for _ in range(150):
            n = rng.randint(2, 9)
            sizes = [rng.randint(1, 6) for _ in range(n)]
            live = [rng.randint(0, 9) for _ in range(n)]
            C = rng.randint(max(sizes), sum(sizes) + 2)
            forced = tuple(sorted(rng.sample(range(1, n), rng.randint(0, min(2, n - 1)))))
            r, qmax = rng.randint(0, C), rng.randint(1, n + 1)
            want = front_mincarry_py(sizes, live, C, r, qmax, forced)
            phases = phase_partition(sizes, live, C, r, qmax, forced)
            if want is None:
                assert phases is None
                continue
            assert phases[0][0] == 0 and phases[-1][1] == n
            assert all(a[1] == b[0] for a, b in zip(phases, phases[1:]))
            cuts = [end for _, end in phases[:-1]]
            assert set(forced) <= set(cuts)
            widths = [sum(sizes[a:b]) for a, b in phases]
            assert max(widths) <= C and sum(max(0, R - r) for R in widths) <= C - r
            assert (sum(2 * live[p - 1] for p in cuts), len(phases)) == want

    def test_q_fit_restarts_at_a_forced_cut(self):
        assert q_fit([2, 2, 2], 6) == 1
        assert q_fit([2, 2, 2], 6, forced=(1,)) == 2
        assert q_fit([2, 2, 2], 6, forced=(1, 2)) == 3
        assert q_fit([7, 1], 6) is None

    def test_a_plan_with_forced_cuts_never_merges_across_them(self):
        plan = plan_steps([2, 2, 2, 2], [5, 5, 5, 5], 6, forced=(2,))
        assert plan is not None and plan.q_fit == 2
        for r, carry, q in plan.steps:
            phases = phase_partition([2, 2, 2, 2], [5, 5, 5, 5], 6, r, plan.qmax, (2,))
            assert 2 in [end for _, end in phases[:-1]] and len(phases) == q
