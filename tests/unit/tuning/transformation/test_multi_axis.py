"""MultiAxisDriver / VectorRateScheduler: interleaved value-domain continuation.

Drives the driver over the mock-axis zoo and asserts (a) each axis reaches its
feasible edge, (b) the committed vector is monotone per axis (I2), and (c) the
hard value-domain guard rejects a blend/LIF/TTFS axis. Pure-python, no GPU.
"""

import pytest

from mimarsinan.tuning.orchestration.multi_axis_driver import (
    MultiAxisDriver,
    ValueAxisSpec,
    VectorRateScheduler,
    assert_value_domain_axes,
)

from . import mock_axis_zoo as zoo

_MIN_STEP = 1e-3
_EDGE_TOL = 2 * _MIN_STEP


def _spec(name, attempt):
    return ValueAxisSpec(name=name, attempt=attempt)


def _is_monotone(seq):
    return all(b >= a - 1e-12 for a, b in zip(seq, seq[1:]))


# ---------------------------------------------------------------------------
# Each axis reaches its feasible edge
# ---------------------------------------------------------------------------

def test_smooth_axes_all_reach_one():
    a0, e0 = zoo.smooth_monotone()
    a1, e1 = zoo.smooth_monotone()
    driver = MultiAxisDriver(
        [_spec("a0", a0), _spec("a1", a1)], min_step=_MIN_STEP
    )
    final = driver.run()
    assert final[0] == pytest.approx(e0, abs=_EDGE_TOL)
    assert final[1] == pytest.approx(e1, abs=_EDGE_TOL)
    assert all(v >= 1.0 - _EDGE_TOL for v in final)


def test_three_axes_each_reach_their_edge():
    a0, e0 = zoo.cliff(0.5)
    a1, e1 = zoo.recovery_limited(0.75)
    a2, e2 = zoo.smooth_monotone()
    driver = MultiAxisDriver(
        [_spec("cliff", a0), _spec("recov", a1), _spec("smooth", a2)],
        min_step=_MIN_STEP,
    )
    final = driver.run()
    assert final[0] == pytest.approx(e0, abs=_EDGE_TOL)
    assert final[1] == pytest.approx(e1, abs=_EDGE_TOL)
    assert final[2] == pytest.approx(e2, abs=_EDGE_TOL)


def test_plateau_and_timing_edges():
    a0, e0 = zoo.plateau_then_drop(0.625)
    a1, e1 = zoo.adversarial_timing()  # off-grid edge -> converges within min_step
    driver = MultiAxisDriver(
        [_spec("plateau", a0), _spec("timing", a1)], min_step=_MIN_STEP
    )
    final = driver.run()
    assert final[0] == pytest.approx(e0, abs=_EDGE_TOL)
    assert final[1] == pytest.approx(e1, abs=_EDGE_TOL)
    # The off-grid edge is approached from below, never overshot in the commit.
    assert final[1] <= e1 + 1e-9


def test_non_monotone_axis_reaches_edge_with_monotone_commit():
    attempt, edge = zoo.non_monotone(0.875)
    driver = MultiAxisDriver([_spec("nm", attempt)], min_step=_MIN_STEP)
    final = driver.run()
    assert final[0] == pytest.approx(edge, abs=_EDGE_TOL)
    # The commit path stays monotone even on a non-monotone feasibility surface
    # (the monotone-commit guard). The worktree's richer _StatefulAxis tracked a
    # non-monotone probe readout via probe_log; this repo's zoo exposes plain
    # closures, so we assert the committed trace directly.
    assert _is_monotone(driver.traces[0])


# ---------------------------------------------------------------------------
# I2: the committed vector is monotone per axis
# ---------------------------------------------------------------------------

def test_committed_trace_is_monotone_per_axis():
    a0, _ = zoo.cliff(0.5)
    a1, _ = zoo.plateau_then_drop(0.625)
    a2, _ = zoo.smooth_monotone()
    driver = MultiAxisDriver(
        [_spec("a0", a0), _spec("a1", a1), _spec("a2", a2)], min_step=_MIN_STEP
    )
    driver.run()
    for trace in driver.traces:
        assert _is_monotone(trace), trace
    for v in driver.committed_vector:
        assert 0.0 <= v <= 1.0


def test_vector_scheduler_committed_never_retreats_on_rollback():
    # An axis whose probe value can read below the proposed target must not pull
    # the committed vector backward.
    a0, _ = zoo.cliff(0.5)
    sched = VectorRateScheduler([a0], names=["cliff"], min_step=_MIN_STEP)
    sched.run()
    assert _is_monotone(sched.traces[0])
    assert sched.committed_vector[0] == pytest.approx(0.5, abs=_EDGE_TOL)


# ---------------------------------------------------------------------------
# Least-sensitive axis first
# ---------------------------------------------------------------------------

def test_least_sensitive_axis_advances_first():
    # A cliff at 0.25 is the *most* sensitive (least headroom once it caps);
    # the smooth axis is least sensitive and should reach 1.0 while the cliff
    # stays pinned at its edge.
    a_cliff, e_cliff = zoo.cliff(0.25)
    a_smooth, _ = zoo.smooth_monotone()
    driver = MultiAxisDriver(
        [_spec("cliff", a_cliff), _spec("smooth", a_smooth)], min_step=_MIN_STEP
    )
    final = driver.run()
    assert final[0] == pytest.approx(e_cliff, abs=_EDGE_TOL)
    assert final[1] >= 1.0 - _EDGE_TOL


# ---------------------------------------------------------------------------
# Hard value-domain guard: never blend / LIF / TTFS
# ---------------------------------------------------------------------------

class _NonValueAxis:
    """A finalize-bearing axis (e.g. LIF/TTFS) that interleaving must refuse."""

    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
        self.is_value_domain = False

    def attempt(self, target):
        return target


@pytest.mark.parametrize("domain", ["lif", "ttfs", "blend"])
def test_guard_rejects_non_value_domain_axis(domain):
    bad = _NonValueAxis(f"{domain}_axis", domain)
    with pytest.raises(ValueError, match="value-domain only"):
        MultiAxisDriver([bad])
    with pytest.raises(ValueError, match="value-domain only"):
        assert_value_domain_axes([bad])


def test_guard_rejects_when_mixed_with_value_axes():
    good, _ = zoo.smooth_monotone()
    bad = _NonValueAxis("lif_axis", "lif")
    with pytest.raises(ValueError, match="lif_axis"):
        MultiAxisDriver([_spec("good", good), bad])


def test_guard_accepts_plain_callable_and_value_spec():
    plain, _ = zoo.smooth_monotone()
    spec = ValueAxisSpec(name="v", attempt=zoo.smooth_monotone()[0])
    # Neither raises.
    assert_value_domain_axes([plain, spec])
    driver = MultiAxisDriver([plain, spec], min_step=_MIN_STEP)
    final = driver.run()
    assert all(v >= 1.0 - _EDGE_TOL for v in final)
