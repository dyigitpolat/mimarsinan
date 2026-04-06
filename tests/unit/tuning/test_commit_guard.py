"""Tests for ActivationAdaptationTuner._after_run() commit behavior.

_after_run() always commits base_activation to ReLU (LeakyGradReLU) and runs
recovery training.  The pipeline requires ReLU-compatible activations after
this step, so there is no rollback -- only commit + train.
"""

import pytest
import torch

from conftest import MockPipeline, make_tiny_supermodel, default_config

from mimarsinan.tuning.tuners.activation_adaptation_tuner import (
    ActivationAdaptationTuner,
)
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.models.layers import TransformedActivation
from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    has_non_relu_activations,
)


def _model_with_gelu():
    model = make_tiny_supermodel()
    for p in model.get_perceptrons():
        p.base_activation = make_activation("GELU")
        p.base_activation_name = "GELU"
        p.set_activation(TransformedActivation(p.base_activation, []))
    return model


class TestCommitBehavior:
    @pytest.fixture
    def setup(self, tmp_path):
        cfg = default_config()
        cfg["tuning_budget_scale"] = 1.0
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        model = _model_with_gelu()
        am = AdaptationManager()
        tuner = ActivationAdaptationTuner(pipeline, model, 0.9, 0.001, am)
        tuner._rollback_tolerance = 0.05
        return tuner, model

    def test_always_commits_to_relu(self, setup):
        """After _after_run(), all activations must be ReLU-compatible."""
        tuner, model = setup
        tuner._after_run()
        assert not has_non_relu_activations(model)

    def test_committed_metric_is_set(self, setup):
        """_after_run() must set _committed_metric."""
        tuner, model = setup
        result = tuner._after_run()
        assert tuner._committed_metric is not None
        assert result == tuner._committed_metric

    def test_validate_returns_committed_metric(self, setup):
        """After _after_run(), validate() returns the cached committed metric."""
        tuner, model = setup
        tuner._after_run()
        metric = tuner._committed_metric
        assert tuner.validate() == metric
        assert tuner.validate() == metric

    def test_floor_derived_from_degradation_tolerance(self, setup):
        """The floor should be target * (1 - degradation_tolerance)."""
        tuner, _ = setup
        dt = float(tuner.pipeline.config.get("degradation_tolerance", 0.05))
        expected_floor = 0.9 * (1.0 - dt)
        assert tuner.target_adjuster.floor == pytest.approx(expected_floor, rel=0.01)
