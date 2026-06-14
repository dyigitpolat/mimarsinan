"""AdaptationAxis is the sole rate-application path (P1 graduated): a manager-rate
tuner drives its full ramp through the axis and reaches a committed rate of 1.0.

The byte-for-byte equivalence with the (now-deleted) legacy inline path is frozen
into the golden decision traces (test_golden_traces.py)."""

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.axes import ManagerRateAxis
from mimarsinan.tuning.adaptation_rate_tuner import AdaptationRateTuner


class _RateTuner(AdaptationRateTuner):
    rate_attr = "quantization_rate"

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr, adaptation_manager)
        self._applied = 0.0
        # one-shot at 1.0 fails (instant low) → exercise the ramp; gradual commits.
        post_fn = lambda r: 0.1 if r >= 0.99 else 0.9
        self.trainer.validate_n_batches = lambda n: post_fn(self._applied)
        self.trainer.validate = lambda: post_fn(self._applied)
        self.trainer.train_steps_until_target = lambda *a, **k: None
        self.trainer.test = lambda: post_fn(1.0)

    def _find_lr(self):
        return 0.001

    def _apply_rate(self, rate):
        self._applied = float(rate)
        super()._apply_rate(rate)


def _run(tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    tuner = _RateTuner(pipeline, model, 0.9, 0.001, manager)
    tuner.run()
    return tuner


def test_axis_is_the_sole_rate_path(tmp_path, deterministic_rng):
    tuner = _run(tmp_path)
    assert isinstance(tuner._axis, ManagerRateAxis)
    assert tuner._committed_rate == pytest.approx(1.0)
    # the ramp ran (one-shot at 1.0 failed first) and committed at least once.
    outcomes = [r.outcome for r in tuner._cycle_log.records]
    assert "commit" in outcomes
