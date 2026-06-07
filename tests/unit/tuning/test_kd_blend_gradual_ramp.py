"""KD blend tuners must ramp ANN->SNN genuinely gradually.

The SmartSmoothAdaptation philosophy: the transformation advances in small
committed increments, each recovered by training DURING the ramp — not a
one-shot jump to rate 1.0 (recovered later in stabilization) and not 0.5-sized
cliffs. These tests pin the gradual contract for the LIF and TTFS-cycle
adaptation steps (the 2026-06-08 incident: TTFS Cycle Fine-Tuning ramped
0 -> 0.5 -> 1.0 in two cliffs, each licensed to lose ~3 pp).
"""

from __future__ import annotations

import pytest

from conftest import make_tiny_supermodel

from mimarsinan.pipelining.pipeline_steps.adaptation.lif_adaptation_step import (
    LIFAdaptationStep,
)
from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
    TTFSCycleAdaptationStep,
)
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
    KDBlendAdaptationTuner,
)


def _seed(mock_pipeline, *, spiking_mode, target=0.0, **config):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = spiking_mode
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    # A realistic training budget so min_step (check_interval / total steps)
    # does not swallow the gradual ladder on the tiny fixture.
    mock_pipeline.config["training_epochs"] = 50
    mock_pipeline.config.setdefault("simulation_steps", 8)
    mock_pipeline.config.update(config)
    mock_pipeline._target_metric = target
    mock_pipeline.seed("model", model, step_name="Activation Analysis")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")
    return model, am


def _run(mock_pipeline, step_cls, name):
    step = step_cls(mock_pipeline)
    step.name = name
    mock_pipeline.prepare_step(step)
    step.run()
    return step


def _assert_gradual(cycle_log):
    """The gradual contract (ladder geometry itself is pinned by the
    SmartSmoothAdaptation unit tests): no one-shot 1.0 jump, a small first
    increment, and several committed intermediate rates — the transformation
    advances and recovers DURING the ramp."""
    assert cycle_log, "tuner must record adaptation cycles"
    rates = [e["rate"] for e in cycle_log]
    # No one-shot: the first proposed rate is a small increment, not 1.0/0.5.
    assert rates[0] <= 0.2, f"first proposed rate {rates[0]} is a cliff, not a ramp"
    intermediate_commits = {
        round(e["committed"], 6)
        for e in cycle_log
        if e["outcome"] == "commit" and 1e-6 < e["committed"] < 1.0 - 1e-6
    }
    assert len(intermediate_commits) >= 3, (
        f"expected >= 3 committed intermediate rates, got {sorted(intermediate_commits)}"
    )
    assert len(cycle_log) >= 4


class TestGradualContract:
    def test_kd_blend_tuners_skip_one_shot(self):
        assert KDBlendAdaptationTuner._skip_one_shot is True

    def test_kd_blend_tuners_request_small_uniform_steps(self):
        assert KDBlendAdaptationTuner._initial_ramp_step <= 0.125 + 1e-9
        assert KDBlendAdaptationTuner._ramp_step_growth == pytest.approx(1.0)


class TestLIFGradualRamp:
    def test_lif_adaptation_ramps_gradually(self, mock_pipeline):
        _seed(
            mock_pipeline, spiking_mode="lif",
            cycle_accurate_lif_forward=True,
        )
        step = _run(mock_pipeline, LIFAdaptationStep, "LIF Adaptation")
        _assert_gradual(step.tuner._cycle_log)


class TestTTFSCycleGradualRamp:
    def test_ttfs_cycle_cascaded_ramps_gradually(self, mock_pipeline):
        _seed(
            mock_pipeline, spiking_mode="ttfs_cycle_based",
            ttfs_cycle_schedule="cascaded",
        )
        step = _run(
            mock_pipeline, TTFSCycleAdaptationStep, "TTFS Cycle Fine-Tuning",
        )
        _assert_gradual(step.tuner._cycle_log)
