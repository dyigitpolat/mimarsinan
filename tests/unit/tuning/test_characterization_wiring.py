"""D1: characterize() wired into the live driver (off by default).

The pure characterization function was unit-tested but never called in ``src``;
the monotonicity verdict and ``epsilon_hint`` never configured a real run. These
tests cover the live wiring: enabled, a non-monotone axis flips the *actual*
scheduler into dense_grid safe mode; disabled, it is a bit-exact no-op.
"""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_driver import AdaptationDriver
from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import (
    SmoothAdaptationTuner,
)


def _tuner(tmp_path, *, enable, grid=None, drops=None):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["tuning_enable_characterization"] = enable
    if grid is not None:
        cfg["tuning_characterization_grid"] = grid
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    drops = drops or {}

    class _T(SmoothAdaptationTuner):
        def _update_and_evaluate(self, rate):
            return 0.9 - drops.get(round(rate, 2), 0.0)  # baseline 0.9

        def _find_lr(self):
            return 0.001

    tuner = _T(pipeline, make_tiny_supermodel(), 0.9, 0.001)
    tuner._validation_baseline = 0.9
    tuner._rollback_tolerance = 0.02
    tuner.trainer.train_steps_until_target = lambda *a, **k: None
    return tuner


def test_characterization_off_by_default_is_noop(tmp_path):
    # the key is absent → no Profile is built, the search path is unchanged.
    tuner = _tuner(tmp_path, enable=False)
    assert tuner._maybe_characterize() is None
    tuner.close()


def test_nonmonotone_axis_flips_live_scheduler_to_dense_grid(tmp_path):
    # drop dips in the middle then rises (a re-aligning quant grid) → non-monotone.
    drops = {0.0: 0.0, 0.25: 0.06, 0.5: 0.01, 0.75: 0.07, 1.0: 0.1}
    tuner = _tuner(tmp_path, enable=True, grid=[0.0, 0.25, 0.5, 0.75, 1.0], drops=drops)

    profile = tuner._maybe_characterize()
    assert profile is not None and profile.monotonic is False

    # The LIVE scheduler the run builds from this profile is dense_grid safe mode,
    # not the default greedy_to_one.
    scheduler = AdaptationDriver.build_scheduler(
        epsilon=profile.epsilon_hint,
        max_rounds=10,
        skip_one_shot=False,
        initial_step=profile.epsilon_hint,
        policy_override="dense_grid" if not profile.monotonic else None,
    )
    assert scheduler.policy == "dense_grid"
    tuner.close()


def test_monotone_axis_keeps_greedy_and_shrinks_epsilon(tmp_path):
    # smooth increasing drop → monotone; epsilon_hint set, policy stays greedy.
    drops = {0.0: 0.0, 0.25: 0.02, 0.5: 0.04, 0.75: 0.06, 1.0: 0.08}
    tuner = _tuner(tmp_path, enable=True, grid=[0.0, 0.25, 0.5, 0.75, 1.0], drops=drops)
    profile = tuner._maybe_characterize()
    assert profile is not None and profile.monotonic is True
    scheduler = AdaptationDriver.build_scheduler(
        epsilon=profile.epsilon_hint, max_rounds=10, skip_one_shot=False,
        initial_step=profile.epsilon_hint,
        policy_override="dense_grid" if not profile.monotonic else None,
    )
    assert scheduler.policy == "greedy_to_one"
    tuner.close()


def test_characterization_restores_committed_state(tmp_path):
    # the sweep is non-destructive: rate is back to committed (0) afterwards.
    drops = {0.0: 0.0, 0.5: 0.05, 1.0: 0.1}
    tuner = _tuner(tmp_path, enable=True, grid=[0.0, 0.5, 1.0], drops=drops)
    tuner._maybe_characterize()
    assert tuner._committed_rate == 0.0
    tuner.close()


def test_dense_grid_walks_small_steps_not_greedy_jump():
    targets = []

    def attempt(t):
        targets.append(t)
        return t  # always feasible → each small step commits

    sched = RateScheduler(epsilon=1e-3, policy="dense_grid", initial_step=0.1, max_rounds=64)
    out = sched.run(0.0, attempt)
    assert out == pytest.approx(1.0)
    assert targets[0] == pytest.approx(0.1)  # NOT a greedy jump to 1.0
    steps = [b - a for a, b in zip([0.0] + targets, targets)]
    assert max(steps) <= 0.1 + 1e-9
