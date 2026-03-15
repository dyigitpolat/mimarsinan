"""Pipeline step ordering: Activation Analysis + Clamp Adaptation must always run.

The chip only supports ReLU activation. Activation adaptation steps convert any
base activation (LeakyReLU, GELU, Identity) to a ReLU-compatible form by training
the model with ClampDecorator. These steps must run regardless of whether full
activation_quantization (Shifting + Quantization) is enabled.
"""

import pytest

from mimarsinan.pipelining.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.pipelining.pipeline_steps.activation_analysis_step import ActivationAnalysisStep
from mimarsinan.pipelining.pipeline_steps.clamp_adaptation_step import ClampAdaptationStep
from mimarsinan.pipelining.pipeline_steps.activation_shift_step import ActivationShiftStep
from mimarsinan.pipelining.pipeline_steps.activation_quantization_step import ActivationQuantizationStep


def _step_names(config: dict) -> list[str]:
    return [name for name, _ in get_pipeline_step_specs(config)]


def _step_classes(config: dict) -> list[type]:
    return [cls for _, cls in get_pipeline_step_specs(config)]


class TestActivationAdaptationAlwaysPresent:
    """Activation Analysis + Clamp Adaptation must be in the pipeline regardless of config."""

    @pytest.mark.parametrize("act_q", [True, False])
    def test_adaptation_steps_present(self, act_q):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": act_q,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Analysis" in names, (
            f"Activation Analysis missing when activation_quantization={act_q}. "
            "This step is required to determine activation_scale for clamping."
        )
        assert "Clamp Adaptation" in names, (
            f"Clamp Adaptation missing when activation_quantization={act_q}. "
            "This step is required to convert activations to ReLU-compatible form."
        )

    @pytest.mark.parametrize("act_q", [True, False])
    def test_adaptation_before_normalization_fusion(self, act_q):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": act_q,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        clamp_idx = names.index("Clamp Adaptation")
        fusion_idx = names.index("Normalization Fusion")
        assert clamp_idx < fusion_idx, (
            "Clamp Adaptation must come before Normalization Fusion"
        )


class TestActivationQuantizationConditional:
    """Shifting + Quantization steps should only appear when activation_quantization=True."""

    def test_quantization_present_when_enabled(self):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": True,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Shifting" in names
        assert "Activation Quantization" in names

    def test_quantization_absent_when_disabled(self):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Shifting" not in names
        assert "Activation Quantization" not in names


class TestStepOrderingInvariants:
    """General pipeline ordering invariants."""

    @pytest.mark.parametrize("spiking_mode", ["rate", "ttfs", "ttfs_quantized"])
    def test_adaptation_always_present_for_all_modes(self, spiking_mode):
        config = {
            "configuration_mode": "user",
            "spiking_mode": spiking_mode,
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Analysis" in names
        assert "Clamp Adaptation" in names

    def test_adaptation_present_for_torch_models(self):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "torch_custom",
        }
        names = _step_names(config)
        assert "Activation Analysis" in names
        assert "Clamp Adaptation" in names
        # Torch Mapping should come before adaptation
        assert names.index("Torch Mapping") < names.index("Activation Analysis")
