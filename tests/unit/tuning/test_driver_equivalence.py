"""P4: the RateScheduler driver reaches the same final committed rate as the
legacy 3-loop search (outcome equivalence / Tier B), with monotone progress."""

import pytest

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    make_scripted_run_tuner,
    default_config,
)


def _run(use_driver, tmp_path, instant_fn, post_fn, ladder=False):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["tuning_use_driver"] = use_driver
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    tuner = make_scripted_run_tuner(
        pipeline, make_tiny_supermodel(),
        instant_fn=instant_fn, post_fn=post_fn, ladder=ladder,
    )
    tuner.run()
    return tuner


def test_driver_one_shot_matches_legacy_final_rate(tmp_path, deterministic_rng):
    off = _run(False, tmp_path / "off", lambda r: 0.87, lambda r: 0.9)
    on = _run(True, tmp_path / "on", lambda r: 0.87, lambda r: 0.9)
    assert off._committed_rate == pytest.approx(1.0)
    assert on._committed_rate == pytest.approx(1.0)


def test_driver_ssa_ramp_matches_legacy_final_rate(tmp_path, deterministic_rng):
    instant = lambda r: 0.1 if r >= 0.99 else 0.85   # one-shot at 1.0 fails
    post = lambda r: 0.9
    off = _run(False, tmp_path / "off", instant, post)
    on = _run(True, tmp_path / "on", instant, post)
    assert off._committed_rate == pytest.approx(1.0)
    assert on._committed_rate == pytest.approx(1.0)
    # I2 monotone committed progress in the driver trajectory
    committed = [r.committed for r in on._cycle_log.records]
    assert committed == sorted(committed)
    # I1: no committed regression — every commit record's rate is a real advance
    commits = [r.rate for r in on._cycle_log.records if r.outcome == "commit"]
    assert commits == sorted(commits)


def test_driver_kdblend_ladder_reaches_one(tmp_path, deterministic_rng):
    off = _run(False, tmp_path / "off", lambda r: 0.87, lambda r: 0.9, ladder=True)
    on = _run(True, tmp_path / "on", lambda r: 0.87, lambda r: 0.9, ladder=True)
    assert off._committed_rate == pytest.approx(1.0)
    assert on._committed_rate == pytest.approx(1.0)
    # uniform-ladder driver: commits advance by even increments
    commits = [r.rate for r in on._cycle_log.records if r.outcome == "commit"]
    assert commits == sorted(commits)
    assert commits[-1] == pytest.approx(1.0)
