"""Unit tests for TTFSCycleAdaptationStep.

Final-polish fine-tuning for ``spiking_mode == 'ttfs_cycle_based'``: blends each
perceptron's activation to the exact single-spike TTFS kernel (TTFSCycleActivation)
with KD recovery, and marks the adaptation_manager so the clamp/quant decorators
are subsumed.
"""

import pytest

from conftest import (
    MockPipeline,
    default_config,
    make_tiny_supermodel,
)

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.pipelining.pipeline_steps.adaptation.ttfs_cycle_adaptation_step import (
    TTFSCycleAdaptationStep,
)
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner


def _seed_ttfs_cycle_step(mock_pipeline, *, target_metric=0.5):
    model = make_tiny_supermodel()
    am = AdaptationManager()
    mock_pipeline.config["spiking_mode"] = "ttfs_cycle_based"
    mock_pipeline.config["activation_quantization"] = True
    mock_pipeline.config["tuning_budget_scale"] = 1.0
    mock_pipeline.config.setdefault("simulation_steps", 16)
    mock_pipeline._target_metric = target_metric

    mock_pipeline.seed("model", model, step_name="Activation Quantization")
    mock_pipeline.seed("adaptation_manager", am, step_name="Activation Quantization")
    return model, am


def _run_step(mock_pipeline):
    step = TTFSCycleAdaptationStep(mock_pipeline)
    step.name = "TTFS Cycle Fine-Tuning"
    mock_pipeline.prepare_step(step)
    step.run()
    return step


class TestTunerCreation:
    def test_tuner_created(self, mock_pipeline):
        _seed_ttfs_cycle_step(mock_pipeline)
        step = _run_step(mock_pipeline)
        assert step.tuner is not None
        assert isinstance(step.tuner, TTFSCycleAdaptationTuner)


class TestValidate:
    def test_validate_returns_float_in_range(self, mock_pipeline):
        _seed_ttfs_cycle_step(mock_pipeline)
        step = _run_step(mock_pipeline)
        result = step.validate()
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


class TestFinalState:
    def test_ttfs_active_set_and_activation_is_ttfs(self, mock_pipeline):
        model, am = _seed_ttfs_cycle_step(mock_pipeline)
        _run_step(mock_pipeline)
        assert am.ttfs_active is True
        for p in model.get_perceptrons():
            # base_activation is the blend, fully ramped to the TTFS target.
            assert p.base_activation.rate == pytest.approx(1.0)
            assert p.base_activation.activation_type == "TTFS"
            assert isinstance(p.base_activation.target_activation, TTFSCycleActivation)

    def test_decorators_subsumed_no_clamp_quant_wrapping(self, mock_pipeline):
        # With ttfs_active, update_activation must not wrap the blend in clamp/quant
        # decorators (the TTFS kernel does that internally).
        model, am = _seed_ttfs_cycle_step(mock_pipeline)
        am.clamp_rate = 1.0
        am.quantization_rate = 1.0
        _run_step(mock_pipeline)
        for p in model.get_perceptrons():
            # activation is the bare blend (TransformedActivation with no decorators,
            # or the blend itself) — assert no decorators are applied.
            decorators = getattr(p.activation, "decorators", [])
            assert len(decorators) == 0
