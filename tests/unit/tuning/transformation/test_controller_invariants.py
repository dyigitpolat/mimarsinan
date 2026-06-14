"""RateScheduler invariants across the mock-axis zoo (report IV.2, the crux).

Validates that one controller behaves correctly across heterogeneous profiles:
reaches/never-exceeds the feasible edge, monotone committed progress, bounded
probes — deterministically, in milliseconds, no GPU.
"""

import math

import pytest

from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler
from . import mock_axis_zoo as zoo

EPS = 2 ** -6


def _instrument(attempt):
    """Wrap an attempt to record the committed trajectory and probe count."""
    trajectory = []

    def wrapped(target):
        result = attempt(target)
        trajectory.append(result)
        return result

    return wrapped, trajectory


@pytest.mark.parametrize(
    "factory, lo, hi",
    [
        (lambda: zoo.smooth_monotone(), 1.0, 1.0),
        (lambda: zoo.cliff(0.5), 0.5 - EPS, 0.5),
        (lambda: zoo.cliff(0.73), 0.73 - EPS, 0.73),
        (lambda: zoo.plateau_then_drop(0.25), 0.25 - EPS, 0.25),
        (lambda: zoo.recovery_limited(0.6), 0.6 - EPS, 0.6),
        (lambda: zoo.adversarial_timing(0.3), 1.0, 1.0),
        (lambda: zoo.non_monotone(0.4, 0.6), 1.0, 1.0),
    ],
)
def test_scheduler_reaches_feasible_edge(factory, lo, hi):
    attempt, _expected = factory()
    wrapped, trajectory = _instrument(attempt)

    final = RateScheduler(epsilon=EPS, max_rounds=60).run(0.0, wrapped)

    # lands within [lo, hi] — never commits past the feasible edge
    assert lo - 1e-9 <= final <= hi + 1e-9
    # I2: committed progress is monotone non-decreasing
    assert trajectory == sorted(trajectory)
    # I1: never exceeds the upper feasible bound at any probe
    assert max(trajectory) <= hi + 1e-9


@pytest.mark.parametrize("alpha_star", [0.1, 0.5, 0.9])
def test_cliff_probe_count_is_bounded(alpha_star):
    attempt, _ = zoo.cliff(alpha_star)
    wrapped, trajectory = _instrument(attempt)
    RateScheduler(epsilon=EPS, max_rounds=60).run(0.0, wrapped)
    # bisection per round is log-bounded; rounds bounded by 1/epsilon
    per_round_bound = math.ceil(math.log2(1.0 / EPS)) + 1
    assert len(trajectory) <= per_round_bound * math.ceil(1.0 / EPS)


def test_recovery_limited_returns_partial_result():
    attempt, _ = zoo.recovery_limited(0.6)
    final = RateScheduler(epsilon=EPS, max_rounds=60).run(0.0, attempt)
    assert 0.6 - EPS <= final <= 0.6  # a valid, recovered partial (I1), not 1.0


def test_last_successful_step_reaches_edge_too():
    attempt, _ = zoo.cliff(0.5)
    final = RateScheduler(
        epsilon=EPS, policy="last_successful_step", max_rounds=60
    ).run(0.0, attempt)
    assert 0.5 - EPS <= final <= 0.5
