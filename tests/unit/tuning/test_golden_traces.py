"""Golden decision-trace baselines (P0).

These pin the bit-exact decision behavior the later refactor phases must
reproduce: single-cycle outcomes (the P2 AcceptanceSensor contract) and
full run()-level trajectories (the P4 RateScheduler contract). Record/refresh
with ``MIMARSINAN_RECORD_GOLDEN=1 pytest tests/unit/tuning/test_golden_traces.py``.
"""

import os

import pytest

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    make_scripted_run_tuner,
    default_config,
    assert_trace_matches,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner

_GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _pipeline(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["degradation_tolerance"] = 0.05
    return MockPipeline(config=cfg, working_directory=str(tmp_path))


class _SeqTuner(SmoothAdaptationTuner):
    """Direct-``_adaptation`` driver with scripted instant + validate values."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._committed_rate = 0.0
        self._instant = 0.85
        self._validate_seq = []
        self._idx = 0
        self.trainer.train_steps_until_target = lambda *a, **k: None

    def _update_and_evaluate(self, rate):
        return self._instant

    def _find_lr(self):
        return 0.001

    def _validate_n(self, _n):
        i = self._idx
        self._idx += 1
        return self._validate_seq[i] if i < len(self._validate_seq) else self._validate_seq[-1]

    def drive(self, rate, instant, validate_seq):
        self._instant = instant
        self._validate_seq = validate_seq
        self._idx = 0
        self._last_post_acc = None  # force fresh pre_cycle_acc from the sequence
        self.trainer.validate_n_batches = self._validate_n
        return self._adaptation(rate)


def test_golden_single_cycle_outcomes(tmp_path, deterministic_rng):
    tuner = _SeqTuner(_pipeline(tmp_path), make_tiny_supermodel(), 0.9, 0.001)
    tuner._rollback_tolerance = 0.05
    tuner._validation_baseline = 0.9

    o1 = tuner.drive(0.3, instant=0.85, validate_seq=[0.9, 0.9])    # commit
    o2 = tuner.drive(0.5, instant=0.85, validate_seq=[0.9, 0.1])    # rollback
    o3 = tuner.drive(0.7, instant=0.10, validate_seq=[0.9])         # catastrophic

    outcomes = [r.outcome for r in tuner._cycle_log.records]
    assert outcomes == ["commit", "rollback", "catastrophic"]
    assert (o1, o2, o3) == (0.3, 0.3, 0.3)  # commit→0.3, then rollback/catastrophic return committed

    assert_trace_matches(
        tuner._cycle_log, os.path.join(_GOLDEN_DIR, "single_cycle_outcomes.json")
    )


def test_golden_run_one_shot_success(tmp_path, deterministic_rng):
    tuner = make_scripted_run_tuner(
        _pipeline(tmp_path), make_tiny_supermodel(),
        instant_fn=lambda r: 0.87, post_fn=lambda r: 0.9,
    )
    tuner.run()
    assert tuner._committed_rate >= 1.0 - 1e-6
    assert [r.outcome for r in tuner._cycle_log.records] == ["commit"]
    assert_trace_matches(
        tuner._cycle_log, os.path.join(_GOLDEN_DIR, "run_one_shot_success.json")
    )


def test_golden_run_driver_ramp(tmp_path, deterministic_rng):
    # Greedy attempt at 1.0 catastrophically fails → the driver bisects into the
    # cliff (greedy-to-gap + bisect), climbing monotonically toward 1.0.
    tuner = make_scripted_run_tuner(
        _pipeline(tmp_path), make_tiny_supermodel(),
        instant_fn=lambda r: 0.1 if r >= 0.99 else 0.85,
        post_fn=lambda r: 0.9,
    )
    tuner.run()
    assert tuner._committed_rate >= 1.0 - 1e-6
    outcomes = [r.outcome for r in tuner._cycle_log.records]
    assert outcomes[0] == "catastrophic"  # the failed greedy one-shot
    assert "commit" in outcomes
    assert_trace_matches(
        tuner._cycle_log, os.path.join(_GOLDEN_DIR, "run_driver_ramp.json")
    )


def test_golden_run_kdblend_ladder(tmp_path, deterministic_rng):
    tuner = make_scripted_run_tuner(
        _pipeline(tmp_path), make_tiny_supermodel(),
        instant_fn=lambda r: 0.87, post_fn=lambda r: 0.9, ladder=True,
    )
    tuner.run()
    assert tuner._committed_rate >= 1.0 - 1e-6
    records = tuner._cycle_log.records
    # skip_one_shot=True → the ladder starts below 1.0 (not a one-shot jump).
    assert records[0].rate < 1.0
    rates = [r.rate for r in records if r.outcome == "commit"]
    assert rates[-1] == pytest.approx(1.0)
    # growth=1.0 → uniform increments (the policy distinguishing it from SSA's
    # grow-1.5); the final step may be capped at the 1.0 boundary. The exact
    # ladder (budget-dependent on this fixture) is pinned by the golden file.
    incs = [rates[i + 1] - rates[i] for i in range(len(rates) - 1)]
    if len(incs) >= 2:
        assert all(inc <= incs[0] + 1e-6 for inc in incs)
    assert_trace_matches(
        tuner._cycle_log, os.path.join(_GOLDEN_DIR, "run_kdblend_ladder.json")
    )
