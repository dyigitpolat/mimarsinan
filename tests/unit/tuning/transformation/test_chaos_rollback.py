"""Chaos / rollback robustness (report IV.6).

Three failure-injection scenarios prove the controller's safety contract holds
when an axis misbehaves:

(a) a recovery "divergence" — the post-recovery metric collapses — must restore
    the tunable params *bitwise* via the existing ``_adaptation`` rollback path
    (``_restore_state(pre_state)``), leaving the committed rate at its prior value.
(b) ``CheckpointGuard.bracket()`` + ``restore`` gives a bitwise restore after an
    arbitrary mutation (the snapshot/restore primitive the rollback path stands on).
(c) a pathological non-monotone feasibility surface still drives ``RateScheduler``
    to *terminate* with a valid committed rate (invariants I1/I2) — it never spins
    or commits past a feasible edge.

These are pure-Python and GPU-free; the model mutation in (a)/(b) is a tiny CPU
tensor add, not a real transform.
"""

import torch
import torch.nn as nn

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    make_scripted_run_tuner,
    default_config,
)
from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard
from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler
from . import mock_axis_zoo as zoo

EPS = 2 ** -6


def _snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def _assert_bitwise_equal(model, snapshot, label):
    for k, v in model.state_dict().items():
        assert torch.equal(v, snapshot[k]), f"{label}: param {k} not bitwise-restored"


def _mutate(model, delta=1.0):
    with torch.no_grad():
        for p in model.parameters():
            p.add_(delta)


# ---------------------------------------------------------------------------
# (a) recovery divergence -> bitwise rollback via the live _adaptation path
# ---------------------------------------------------------------------------

def test_recovery_divergence_restores_params_bitwise(tmp_path):
    """A destructive cycle whose post_acc collapses rolls back bit-for-bit.

    The cycle's ``_update_and_evaluate`` mutates every parameter (a stand-in for
    a transform that diverges under recovery); the post-recovery validation then
    collapses far below the pre-cycle metric, so ``_adaptation`` must restore the
    snapshot taken at cycle entry and keep the committed rate at 0.0.
    """
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()

    tuner = make_scripted_run_tuner(
        pipeline, model, instant_fn=lambda r: 0.9, post_fn=lambda r: 0.9,
    )
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9

    def _diverge(rate):
        _mutate(tuner.model, delta=1.0)  # destructive transform
        return 0.85                      # instant acc is non-catastrophic

    tuner._update_and_evaluate = _diverge
    # pre_cycle_acc=0.9, post_acc=0.05 -> far below pre - tolerance -> rollback.
    post_seq = iter([0.9, 0.05])
    tuner.trainer.validate_n_batches = lambda n: next(post_seq, 0.05)
    tuner.trainer.train_steps_until_target = lambda *a, **k: None
    tuner._last_post_acc = None

    before = _snapshot(model)
    result = tuner._adaptation(0.5)

    assert result == tuner._committed_rate == 0.0
    _assert_bitwise_equal(model, before, "recovery divergence rollback")


def test_committed_cycle_then_divergence_restores_to_committed(tmp_path):
    """After a clean commit, a later diverging cycle rolls back to the committed
    state — not to the original — proving rollback restores the *cycle-entry*
    snapshot, not a stale baseline."""
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()

    tuner = make_scripted_run_tuner(
        pipeline, model, instant_fn=lambda r: 0.9, post_fn=lambda r: 0.9,
    )
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9
    tuner.trainer.train_steps_until_target = lambda *a, **k: None

    # Cycle 1: a clean, non-destructive commit at rate 0.5 (no mutation).
    tuner._update_and_evaluate = lambda rate: 0.9
    commit_seq = iter([0.9, 0.9])
    tuner.trainer.validate_n_batches = lambda n: next(commit_seq, 0.9)
    tuner._last_post_acc = None
    assert tuner._adaptation(0.5) == 0.5
    assert tuner._committed_rate == 0.5
    committed_state = _snapshot(model)

    # Cycle 2: a destructive attempt whose post_acc collapses -> rollback to the
    # committed state captured at the entry of cycle 2.
    def _diverge(rate):
        _mutate(tuner.model, delta=2.0)
        return 0.85

    tuner._update_and_evaluate = _diverge
    rollback_seq = iter([0.9, 0.02])
    tuner.trainer.validate_n_batches = lambda n: next(rollback_seq, 0.02)
    tuner._last_post_acc = None

    result = tuner._adaptation(0.75)
    assert result == tuner._committed_rate == 0.5
    _assert_bitwise_equal(model, committed_state, "rollback to committed state")


# ---------------------------------------------------------------------------
# (b) CheckpointGuard.bracket() + restore -> bitwise restore after a mutation
# ---------------------------------------------------------------------------

class _Stub:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device


def test_checkpoint_guard_bracket_restores_bitwise():
    """``bracket()`` snapshots on enter; restoring its handle after an arbitrary
    mutation is bit-for-bit identical (stronger than the ``allclose`` check in
    ``test_checkpoint_guard.test_bracket_restores_on_rollback``)."""
    model = make_tiny_supermodel()
    guard = CheckpointGuard(_Stub(model))

    before = _snapshot(model)
    with guard.bracket() as handle:
        _mutate(model, delta=3.14159)
        assert any(
            not torch.equal(v, before[k]) for k, v in model.state_dict().items()
        ), "mutation inside bracket did not change params"
        guard.restore(handle)
    _assert_bitwise_equal(model, before, "checkpoint bracket restore")


def test_checkpoint_guard_bracket_handle_independent_of_live_model():
    """The bracket handle is a true snapshot — mutating the live model after the
    snapshot leaves the handle unchanged, so restore is always to the entry state."""
    model = make_tiny_supermodel()
    guard = CheckpointGuard(_Stub(model))

    with guard.bracket() as handle:
        captured = {k: v.clone() for k, v in handle.state.items()}
        _mutate(model, delta=5.0)
        for k, v in handle.state.items():
            assert torch.equal(v, captured[k]), (
                f"handle param {k} aliased the live model and mutated"
            )


# ---------------------------------------------------------------------------
# (c) pathological non-monotone surface still terminates with a valid commit
# ---------------------------------------------------------------------------

def test_non_monotone_surface_terminates_with_valid_commit():
    """A surface feasible in [0, low] U [high, 1] (a gap in the middle) still
    drives ``RateScheduler`` to a valid, terminated commit (I1/I2).

    Greedy-to-one jumps straight to the feasible endpoint 1.0, so this surface
    commits at 1.0. The point is robustness: the controller does not spin or
    commit a rate the surface rejected. A dense-grid "safe mode" that would walk
    *into* the middle gap and stop at ``low`` is the future V9 handling; today's
    greedy controller is correct because it only ever commits rates the surface
    accepted.
    """
    attempt, _expected = zoo.non_monotone(0.4, 0.6)

    trajectory = []

    def instrumented(target):
        result = attempt(target)
        trajectory.append(result)
        return result

    final = RateScheduler(epsilon=EPS, max_rounds=60).run(0.0, instrumented)

    # Terminates with a valid committed rate in [0, 1].
    assert 0.0 <= final <= 1.0 + 1e-9
    # Greedy reaches the feasible endpoint.
    assert final == 1.0
    # I2: committed progress is monotone non-decreasing.
    assert trajectory == sorted(trajectory)
    # I1: every committed probe value was a rate the surface actually accepted.
    for committed in trajectory:
        assert committed <= 0.4 + 1e-9 or committed >= 0.6 - 1e-9 or committed == 0.0


def test_non_monotone_bisection_stops_at_gap_edge_when_endpoint_blocked():
    """If the feasible endpoint 1.0 is itself blocked, bisection still terminates
    at the lower feasible edge (``low``) without committing into the gap — a
    valid partial commit, never an infinite loop (I1/I2)."""
    state = {"c": 0.0}

    def attempt(target):
        # Feasible only in [0, 0.4]; everything above the gap edge is rejected.
        if target <= 0.4 + 1e-12:
            state["c"] = target
        return state["c"]

    trajectory = []

    def instrumented(target):
        result = attempt(target)
        trajectory.append(result)
        return result

    final = RateScheduler(epsilon=EPS, max_rounds=60).run(0.0, instrumented)

    assert 0.4 - EPS <= final <= 0.4 + 1e-9  # at the feasible edge, not in the gap
    assert trajectory == sorted(trajectory)  # I2
    assert max(trajectory) <= 0.4 + 1e-9     # I1: never committed past the edge
