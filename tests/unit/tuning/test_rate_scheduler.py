"""RateScheduler policy properties against mock attempt surfaces (P4)."""

import math

import pytest

from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler

EPS = 2 ** -6  # 0.015625


def _smooth():
    """Every proposed rate is feasible (commits at the proposed value)."""
    return lambda target: target


def _cliff(alpha_star):
    """Feasible iff target <= alpha_star; else rolls back to committed."""
    state = {"committed": 0.0}

    def attempt(target):
        if target <= alpha_star + 1e-12:
            state["committed"] = target
        return state["committed"]

    return attempt, state


def test_invalid_policy_raises():
    with pytest.raises(ValueError):
        RateScheduler(epsilon=EPS, policy="nope")


def test_greedy_reaches_one_in_a_single_round_when_smooth():
    calls = []
    attempt = lambda t: (calls.append(t), t)[1]
    final = RateScheduler(epsilon=EPS).run(0.0, attempt)
    assert final == pytest.approx(1.0)
    assert calls == [1.0]  # one greedy jump


def test_cliff_bisects_to_edge_and_never_commits_past():
    attempt, state = _cliff(0.5)
    calls = []
    wrapped = lambda t: (calls.append(t), attempt(t))[1]
    final = RateScheduler(epsilon=EPS).run(0.0, wrapped)
    # committed lands within epsilon below the cliff, never above it
    assert final <= 0.5 + 1e-9
    assert 0.5 - EPS <= final <= 0.5
    assert max(c for c in calls if c <= 0.5 + 1e-12) <= 0.5


def test_probe_count_per_round_is_log_bounded():
    attempt, _ = _cliff(0.0)  # nothing feasible above 0 → pure bisection underflow
    calls = []
    wrapped = lambda t: (calls.append(t), attempt(t))[1]
    RateScheduler(epsilon=EPS).run(0.0, wrapped)
    # one round, bisecting the unit gap down to epsilon
    assert len(calls) <= math.ceil(math.log2(1.0 / EPS)) + 1


def test_uniform_ladder_takes_even_steps():
    commits = []
    attempt = lambda t: (commits.append(t), t)[1]
    final = RateScheduler(
        epsilon=EPS, policy="uniform_ladder", initial_step=0.25
    ).run(0.0, attempt)
    assert final == pytest.approx(1.0)
    assert commits == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_one_shot_only_does_not_bisect():
    attempt, _ = _cliff(0.5)  # 1.0 fails
    calls = []
    wrapped = lambda t: (calls.append(t), attempt(t))[1]
    final = RateScheduler(epsilon=EPS, policy="one_shot_only").run(0.0, wrapped)
    assert calls == [1.0]      # single attempt, no bisection fallback
    assert final == 0.0


def test_max_rounds_bounds_the_outer_loop():
    attempt = lambda t: t  # smooth, but cap rounds at 1 via uniform ladder
    final = RateScheduler(
        epsilon=EPS, policy="uniform_ladder", initial_step=0.25, max_rounds=2
    ).run(0.0, attempt)
    assert final == pytest.approx(0.5)  # 2 rounds × 0.25
