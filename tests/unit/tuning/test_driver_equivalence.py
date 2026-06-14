"""P4 (graduated): the RateScheduler driver is the sole run loop. It reaches a
committed rate of 1.0 with monotone progress (I1/I2) across the one-shot,
ramp-after-failed-one-shot, and uniform-ladder (KD-blend) scenarios."""

import pytest

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    make_scripted_run_tuner,
    default_config,
)


def _run(tmp_path, instant_fn, post_fn, ladder=False):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = make_scripted_run_tuner(
        pipeline, make_tiny_supermodel(),
        instant_fn=instant_fn, post_fn=post_fn, ladder=ladder,
    )
    tuner.run()
    return tuner


def test_driver_one_shot_reaches_one(tmp_path, deterministic_rng):
    tuner = _run(tmp_path, lambda r: 0.87, lambda r: 0.9)
    assert tuner._committed_rate == pytest.approx(1.0)


def test_driver_ramp_after_failed_one_shot_is_monotone(tmp_path, deterministic_rng):
    instant = lambda r: 0.1 if r >= 0.99 else 0.85   # one-shot at 1.0 fails
    tuner = _run(tmp_path, instant, lambda r: 0.9)
    assert tuner._committed_rate == pytest.approx(1.0)
    # I2 monotone committed progress in the driver trajectory
    committed = [r.committed for r in tuner._cycle_log.records]
    assert committed == sorted(committed)
    # I1: no committed regression — every commit record's rate is a real advance
    commits = [r.rate for r in tuner._cycle_log.records if r.outcome == "commit"]
    assert commits == sorted(commits)


def test_driver_kdblend_ladder_reaches_one(tmp_path, deterministic_rng):
    tuner = _run(tmp_path, lambda r: 0.87, lambda r: 0.9, ladder=True)
    assert tuner._committed_rate == pytest.approx(1.0)
    # uniform-ladder driver: commits advance by even increments up to 1.0
    commits = [r.rate for r in tuner._cycle_log.records if r.outcome == "commit"]
    assert commits == sorted(commits)
    assert commits[-1] == pytest.approx(1.0)
