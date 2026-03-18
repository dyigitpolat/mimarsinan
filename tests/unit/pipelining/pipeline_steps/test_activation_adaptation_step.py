"""Tests for ActivationAdaptationStep and activation_utils.

ActivationAdaptationStep always runs after Activation Analysis. It uses
ActivationAdaptationTuner (SmartSmoothAdaptation) to gradually blend non-ReLU
chip-targeted activations toward ReLU. When all activations are already
ReLU-compatible, it is a no-op.

This step does NOT apply activation_scales -- that is the responsibility of
downstream steps (Clamp Adaptation, etc.).
"""

import pytest

from conftest import (
    MockPipeline,
    make_tiny_supermodel,
    default_config,
)

from mimarsinan.pipelining.pipeline_steps.activation_utils import (
    RELU_COMPATIBLE_TYPES,
    has_non_relu_activations,
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
    """has_non_relu_activations — checks for non-ReLU chip-targeted perceptrons."""

    def test_false_when_all_relu(self):
        model = make_tiny_supermodel()
        for p in model.get_perceptrons():
            p.base_activation = make_activation("ReLU")
            p.base_activation_name = "ReLU"
        assert has_non_relu_activations(model) is False

    def test_true_when_any_gelu(self):
        """GELU perceptrons are chip-targeted (adapted to ReLU in ActivationAdaptationStep).
        They are included in get_perceptrons() and trigger activation adaptation."""
        model = _model_with_gelu()
        assert has_non_relu_activations(model) is True

    def test_true_when_any_leaky_relu(self):
        model = make_tiny_supermodel()
        perceptrons = model.get_perceptrons()
        perceptrons[0].base_activation = make_activation("LeakyReLU")
        perceptrons[0].base_activation_name = "LeakyReLU"
        perceptrons[0].set_activation(
            TransformedActivation(perceptrons[0].base_activation, [])
        )
        assert has_non_relu_activations(model) is True

    def test_constants_defined(self):
        assert "ReLU" in RELU_COMPATIBLE_TYPES
        assert "LeakyGradReLU" in RELU_COMPATIBLE_TYPES


class TestActivationAdaptationStepRates:
    """ActivationAdaptationStep must not set clamp_rate (stays 0)."""

    def test_clamp_rate_unchanged_after_step(self, mock_pipeline):
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        model = make_tiny_supermodel()
        am = AdaptationManager()
        assert am.clamp_rate == 0.0
        assert am.activation_adaptation_rate == 0.0

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        assert am.clamp_rate == 0.0, (
            "ActivationAdaptationStep must not set clamp_rate; "
            "when act_q is False clamp stays no-op."
        )
        assert am.activation_adaptation_rate == 0.0, (
            "activation_adaptation_rate should be reset to 0 after adaptation."
        )

    def test_scales_not_applied(self, mock_pipeline):
        """ActivationAdaptationStep must NOT apply activation_scales."""
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        model = make_tiny_supermodel()
        am = AdaptationManager()
        n = len(model.get_perceptrons())

        original_scales = [p.activation_scale.item() for p in model.get_perceptrons()]

        mock_pipeline.config.update(default_config())
        mock_pipeline.config["activation_quantization"] = False
        mock_pipeline.config["tuner_epochs"] = 1
        mock_pipeline.seed("model", model, step_name="Activation Analysis")
        mock_pipeline.seed("adaptation_manager", am, step_name="Activation Analysis")

        step = ActivationAdaptationStep(mock_pipeline)
        step.name = "Activation Adaptation"
        mock_pipeline.prepare_step(step)
        step.run()

        for p, orig in zip(model.get_perceptrons(), original_scales):
            assert p.activation_scale.item() == orig, (
                "ActivationAdaptationStep should not modify activation_scale"
            )
