"""Metamorphic controller relations (report IV.5).

Metamorphic testing asserts relations between the outputs of *related* runs when
the absolute ground truth is unavailable: shrink a tolerance, scale a sample
count, tighten a gate, or rerun an identical config, and the controller must move
in the contractually mandated direction. These hold across the mock-axis zoo with
no real model, no GPU, in milliseconds — the report's IV.5 follow-on (the IV.2
crux) for the ``RateScheduler`` / ``AcceptanceSensor`` services.
"""

import pytest

from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor
from . import mock_axis_zoo as zoo

# A non-dyadic cliff so the bisection lands strictly *below* the edge: smaller
# epsilon resolves a finer increment, exposing the monotone epsilon→committed
# relation. A dyadic edge (e.g. 0.5) is hit exactly at any epsilon and would
# make the relation vacuous.
_CLIFF_ALPHA = 0.37
_FINE_EPS = [2 ** -3, 2 ** -4, 2 ** -5, 2 ** -6, 2 ** -8, 2 ** -10]


def _run_cliff(alpha, epsilon):
    attempt, _ = zoo.cliff(alpha)
    return RateScheduler(epsilon=epsilon, max_rounds=2000).run(0.0, attempt)


# ── Relation M1: shrinking epsilon weakly increases the committed cliff edge ──
# Finer resolution can only resolve the feasible edge more precisely; it can
# never *retreat* from a coarser commit, and never overshoot the true edge.

def test_smaller_epsilon_weakly_increases_cliff_committed():
    descending = sorted(_FINE_EPS, reverse=True)  # coarse → fine
    finals = [_run_cliff(_CLIFF_ALPHA, eps) for eps in descending]
    for coarser, finer in zip(finals, finals[1:]):
        assert finer >= coarser - 1e-12  # monotone non-decreasing as eps shrinks
    assert all(f <= _CLIFF_ALPHA + 1e-9 for f in finals)  # never past the edge
    assert finals[-1] > finals[0]  # the finest strictly improves on the coarsest


@pytest.mark.parametrize("alpha", [0.29, 0.37, 0.61, 0.83])
def test_smaller_epsilon_weakly_increases_for_many_cliffs(alpha):
    finals = [_run_cliff(alpha, eps) for eps in sorted(_FINE_EPS, reverse=True)]
    for coarser, finer in zip(finals, finals[1:]):
        assert finer >= coarser - 1e-12
        assert finer <= alpha + 1e-9


# ── Relation M2: one_shot_only never commits past a cliff ─────────────────────
# A single full jump to 1.0 is the only probe; on a cliff it always fails, so the
# committed rate stays at the prior value (0.0) — never above the feasible edge.

@pytest.mark.parametrize("alpha", [0.1, 0.37, 0.5, 0.73, 0.9])
def test_one_shot_only_never_commits_past_a_cliff(alpha):
    attempt, _ = zoo.cliff(alpha)
    final = RateScheduler(
        epsilon=2 ** -6, policy="one_shot_only", max_rounds=80
    ).run(0.0, attempt)
    assert final <= alpha + 1e-9


def test_one_shot_only_commits_when_the_full_jump_is_feasible():
    # The same policy DOES reach 1.0 on a fully feasible (smooth) axis — proving
    # M2 is about cliff infeasibility, not a policy that can never advance.
    attempt, _ = zoo.smooth_monotone()
    final = RateScheduler(
        epsilon=2 ** -6, policy="one_shot_only", max_rounds=80
    ).run(0.0, attempt)
    assert final == pytest.approx(1.0)


# ── Relation M3: larger N shrinks the paired drop SE at fixed discordance ──────
# Replicating a fixed concordant/discordant *pattern* M times holds the
# discordance fraction (and thus delta_hat) constant while N grows M-fold; the
# paired SE = sqrt(b10+b01)/N must shrink ~1/sqrt(M).

def _replicated_pair(reps, *, b10=4, b01=1, concordant=5):
    ref, cand = [], []
    for _ in range(reps):
        ref += [True] * b10;  cand += [False] * b10   # ref-right, cand-wrong
        ref += [False] * b01; cand += [True] * b01    # ref-wrong, cand-right
        ref += [True] * concordant; cand += [True] * concordant
    return ref, cand


def test_larger_n_reduces_paired_se_at_fixed_discordance():
    prev_se = None
    prev_delta = None
    for reps in (1, 2, 4, 8, 16):
        ref, cand = _replicated_pair(reps)
        delta, se = AcceptanceSensor.paired_drop_se(ref, cand)
        if prev_se is not None:
            assert se < prev_se  # strictly smaller SE with more samples
            assert delta == pytest.approx(prev_delta)  # discordance fraction held
        prev_se, prev_delta = se, delta


def test_paired_se_scales_as_inverse_sqrt_n():
    ref1, cand1 = _replicated_pair(1)
    ref4, cand4 = _replicated_pair(4)
    _, se1 = AcceptanceSensor.paired_drop_se(ref1, cand1)
    _, se4 = AcceptanceSensor.paired_drop_se(ref4, cand4)
    assert se4 == pytest.approx(se1 / 2.0)  # 4× the data → SE halves


# ── Relation M4: tightening k never turns a reject into an accept ─────────────
# ``paired_is_rollback`` rejects iff delta > k·se. Lowering k can only make the
# threshold stricter, so a rejection at a looser k stays a rejection at any
# tighter k — the reject set is downward-closed in k.

def test_tightening_k_never_flips_reject_to_accept():
    ref, cand = _replicated_pair(8)  # delta/se ≈ 3.79, a clear-ish drop
    ks = [10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]  # loosest → tightest
    rejects = [AcceptanceSensor.paired_is_rollback(ref, cand, k) for k in ks]
    for looser, tighter in zip(rejects, rejects[1:]):
        if looser:  # rejected at the looser k → must still reject when tighter
            assert tighter


@pytest.mark.parametrize("reps", [2, 4, 8, 16])
def test_reject_set_is_downward_closed_in_k(reps):
    ref, cand = _replicated_pair(reps)
    # Sweep k descending; once it rejects it must keep rejecting (no flip back).
    ks = [12.0, 8.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.1]
    seen_reject = False
    for k in ks:
        rej = AcceptanceSensor.paired_is_rollback(ref, cand, k)
        if seen_reject:
            assert rej
        seen_reject = seen_reject or rej


# ── Relation M5: an identical scheduler config replays an identical trajectory ─
# Determinism across two runs of the same (axis, policy, epsilon) — no hidden
# state leaks between runs of the pure controller.

def _trajectory(factory, **scheduler_kwargs):
    attempt, _ = factory()
    trajectory = []

    def recording(target):
        result = attempt(target)
        trajectory.append((target, result))
        return result

    final = RateScheduler(**scheduler_kwargs).run(0.0, recording)
    return final, trajectory


@pytest.mark.parametrize(
    "factory",
    [
        lambda: zoo.cliff(0.37),
        lambda: zoo.smooth_monotone(),
        lambda: zoo.non_monotone(0.4, 0.6),
        lambda: zoo.adversarial_timing(0.3),
        lambda: zoo.recovery_limited(0.6),
    ],
)
def test_identical_config_yields_identical_trajectory(factory):
    f1, t1 = _trajectory(factory, epsilon=2 ** -6, max_rounds=80)
    f2, t2 = _trajectory(factory, epsilon=2 ** -6, max_rounds=80)
    assert f1 == f2
    assert t1 == t2
