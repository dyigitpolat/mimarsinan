"""Tests for ActivationAdaptationStep and activation_utils.

When activation_quantization is False, ActivationAdaptationStep runs instead of
ClampAdaptationStep. It must apply activation_scales and (if needed) replace
non-ReLU with ReLU without setting clamp_rate, so Normalization Fusion →
Soft Core Mapping stays exact.
"""

import pytest

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
)

from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    HOST_SIDE_TYPES,
    RELU_COMPATIBLE_TYPES,
    needs_clamp_adaptation,
)
from mimarsinan.pipelining.pipeline_steps.activation_adaptation_step import (
    ActivationAdaptationStep,
)
from mimarsinan.models.perceptron_mixer.perceptron import make_activation
from mimarsinan.models.layers import TransformedActivation


def _model_with_gelu():
    """Model with one GELU perceptron (chip-targeted, can be adapted to ReLU)."""
    model = make_tiny_supermodel()
    perceptrons = model.get_perceptrons()
    assert len(perceptrons) >= 1
    p = perceptrons[0]
    p.base_activation = make_activation("GELU")
    p.base_activation_name = "GELU"
    p.set_activation(TransformedActivation(p.base_activation, []))
    return model


class TestActivationUtils:
    """needs_clamp_adaptation (used for ReLU adaptation decision)."""

    def test_needs_clamp_adaptation_false_when_all_relu(self):
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.base_activation = make_activation("ReLU")
            p.base_activation_name = "ReLU"
        assert needs_clamp_adaptation(model) is False

    def test_needs_clamp_adaptation_true_when_any_gelu(self):
        """GELU perceptrons are chip-targeted (adapted to ReLU in ActivationAdaptationStep).
        They are included in get_perceptrons() and trigger clamp adaptation."""
        model = _model_with_gelu()
        assert needs_clamp_adaptation(model) is True

    def test_needs_clamp_adaptation_true_when_any_leaky_relu(self):
        model = make_tiny_supermodel()
        perceptrons = model.get_perceptrons()
        perceptrons[0].base_activation = make_activation("LeakyReLU")
        perceptrons[0].base_activation_name = "LeakyReLU"
        perceptrons[0].set_activation(
            TransformedActivation(perceptrons[0].base_activation, [])
        )
        assert needs_clamp_adaptation(model) is True

    def test_constants_defined(self):
        assert "ReLU" in RELU_COMPATIBLE_TYPES
        assert "LeakyGradReLU" in RELU_COMPATIBLE_TYPES
        assert "Identity" in HOST_SIDE_TYPES


class TestActivationAdaptationStepClampRateUnchanged:
    """ActivationAdaptationStep must not set clamp_rate (stays 0)."""

    def test_clamp_rate_unchanged_after_step(self, mock_pipeline):
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        model = make_tiny_supermodel()
        am = AdaptationManager()
        assert am.clamp_rate == 0.0

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")
        mock_pipeline.seed(
            "activation_scales",
            [1.0] * len(model.get_perceptrons()),
            step_name="Activation Analysis",
        )

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        assert am.clamp_rate == 0.0, (
            "ActivationAdaptationStep must not set clamp_rate; "
            "when act_q is False clamp stays no-op."
        )

    def test_activation_scales_applied(self, mock_pipeline):
        model = make_tiny_supermodel()
        am = type("AM", (), {"clamp_rate": 0.0})()
        n = len(model.get_perceptrons())
        scales = [2.0] * n if n >= 1 else [1.0]

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")
        mock_pipeline.seed("activation_scales", scales, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        for p, s in zip(model.get_perceptrons(), scales):
            assert p.activation_scale.item() == s
