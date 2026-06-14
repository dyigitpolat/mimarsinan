"""Diagnostic: the full-transformation (rate 1.0) probe trajectory.

After each commit, ``tuning_full_transform_probe`` measures the value-domain
rate-1.0 accuracy from the committed state and records the drop
(committed_acc - full_acc). A shrinking drop means the gradual ramp is pulling
the model toward 1.0-viability; a flat/growing drop means it is not."""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


class _ProbeTuner(SmoothAdaptationTuner):
    """Scripts the rate-1.0 probe accuracy as a function of the committed rate."""

    def __init__(self, pipeline, model, target, lr, *, full_acc_fn):
        super().__init__(pipeline, model, target, lr)
        self._committed_rate = 0.0
        self._validation_baseline = 0.9
        self._rollback_tolerance = 0.05
        self._full_acc_fn = full_acc_fn
        self.trainer.train_steps_until_target = lambda *a, **k: None
        self.trainer.validate_n_batches = lambda n: 0.9  # committed-rate acc

    def _find_lr(self):
        return 0.001

    def _update_and_evaluate(self, rate):
        # rate == 1.0 is the probe; sub-1.0 is the cycle's instant eval (commits).
        if rate >= 1.0 - 1e-9:
            return self._full_acc_fn(self._committed_rate)
        return 0.9


def _make(tmp_path, *, probe, full_acc_fn):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["tuning_full_transform_probe"] = probe
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    return _ProbeTuner(pipeline, make_tiny_supermodel(), 0.9, 0.001, full_acc_fn=full_acc_fn)


def test_probe_off_by_default_records_nothing(tmp_path, deterministic_rng):
    tuner = _make(tmp_path, probe=False, full_acc_fn=lambda a: 0.5)
    tuner._adaptation(0.3)
    tuner._adaptation(0.6)
    assert tuner._full_transform_log == []
    tuner.close()


def test_probe_records_drop_after_each_commit(tmp_path, deterministic_rng):
    # full_acc climbs with the committed rate → the drop shrinks (converging).
    tuner = _make(tmp_path, probe=True, full_acc_fn=lambda a: 0.6 + 0.3 * a)
    tuner._adaptation(0.3)
    tuner._adaptation(0.6)
    tuner._adaptation(0.9)
    log = tuner._full_transform_log
    assert len(log) == 3
    assert [round(r["committed"], 2) for r in log] == [0.3, 0.6, 0.9]
    # committed_acc is the 0.9 cycle metric; full_acc is the scripted 1.0 probe.
    assert all(r["committed_acc"] == pytest.approx(0.9) for r in log)
    assert log[0]["full_acc"] < log[-1]["full_acc"]      # endpoint improved
    assert log[0]["drop"] > log[-1]["drop"]              # drop shrank
    tuner.close()


def test_trend_converging_vs_flat(tmp_path, deterministic_rng, capsys):
    converging = _make(tmp_path, probe=True, full_acc_fn=lambda a: 0.6 + 0.3 * a)
    for r in (0.3, 0.6, 0.9):
        converging._adaptation(r)
    converging._log_full_transform_trend()
    assert "CONVERGING" in capsys.readouterr().out
    converging.close()

    flat = _make(tmp_path, probe=True, full_acc_fn=lambda a: 0.6)  # never improves
    for r in (0.3, 0.6, 0.9):
        flat._adaptation(r)
    with pytest.warns(UserWarning, match="not be pulling the model toward 1.0"):
        flat._log_full_transform_trend()
    assert "FLAT/DIVERGING" in capsys.readouterr().out
    flat.close()


def test_probe_restores_committed_state(tmp_path, deterministic_rng):
    """The probe must not leave the rate at 1.0 — it restores the committed state."""
    tuner = _make(tmp_path, probe=True, full_acc_fn=lambda a: 0.5)
    tuner._adaptation(0.4)
    assert tuner._committed_rate == pytest.approx(0.4)
    tuner.close()
