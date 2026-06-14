"""P1 gate: routing rate application through an AdaptationAxis (tuning_use_axis)
produces a byte-identical decision trace vs the legacy inline path."""

import dataclasses

import pytest

from conftest import MockPipeline, make_tiny_supermodel, default_config
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
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


def _run(use_axis, tmp_path):
    cfg = default_config()
    cfg["tuning_budget_scale"] = 1.0
    cfg["tuning_use_axis"] = use_axis
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    tuner = _RateTuner(pipeline, model, 0.9, 0.001, manager)
    tuner.run()
    return tuner


def _records_no_timing(trace):
    return [
        dataclasses.replace(r, elapsed_sec=0.0, seeds=None) for r in trace.records
    ]


def test_axis_routing_matches_legacy_trace(tmp_path, deterministic_rng):
    off = _run(False, tmp_path / "off")
    on = _run(True, tmp_path / "on")

    assert off._axis is None
    assert on._axis is not None
    assert _records_no_timing(off._cycle_log) == _records_no_timing(on._cycle_log)
    assert off._committed_rate == on._committed_rate == pytest.approx(1.0)
