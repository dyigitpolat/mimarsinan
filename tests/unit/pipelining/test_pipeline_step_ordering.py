"""Pipeline step ordering: Activation Analysis + adaptation step must always run.

The chip only supports ReLU activation. Adaptation step selection:
  - act_q=True (any spiking mode): Clamp Adaptation (ClampDecorator, required for quantization).
  - act_q=False + ttfs/ttfs_quantized: Clamp Adaptation (TTFS saturates relu(V)/θ at 1.0;
    model must be trained to operate within [0, 1] to avoid large information loss).
  - act_q=False + rate: Activation Adaptation (no-quant) — ReLU replacement + scales, no clamp.
"""

import pytest

from mimarsinan.pipelining.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.pipelining.pipeline_steps.activation_analysis_step import ActivationAnalysisStep
from mimarsinan.pipelining.pipeline_steps.activation_adaptation_step import ActivationAdaptationStep
from mimarsinan.pipelining.pipeline_steps.clamp_adaptation_step import ClampAdaptationStep
from mimarsinan.pipelining.pipeline_steps.activation_shift_step import ActivationShiftStep
from mimarsinan.pipelining.pipeline_steps.activation_quantization_step import ActivationQuantizationStep


def _step_names(config: dict) -> list[str]:
    return [name for name, _ in get_pipeline_step_specs(config)]


def _step_classes(config: dict) -> list[type]:
    return [cls for _, cls in get_pipeline_step_specs(config)]


class TestActivationAdaptationAlwaysPresent:
    """Activation Analysis always runs; Clamp or Activation Adaptation by spiking mode + act_q."""

    def test_activation_analysis_always_present(self):
        for act_q in (True, False):
            config = {
                "configuration_mode": "user",
                "spiking_mode": "ttfs",
                "activation_quantization": act_q,
                "weight_quantization": False,
                "model_type": "mlp_mixer",
            }
            names = _step_names(config)
            assert "Activation Analysis" in names, (
                f"Activation Analysis missing when activation_quantization={act_q}."
            )

    def test_clamp_adaptation_when_act_q_true(self):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": True,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Clamp Adaptation" in names
        assert "Activation Adaptation" not in names

    def test_clamp_adaptation_for_ttfs_even_when_act_q_false(self):
        """TTFS saturates relu(V)/θ at 1.0 → Clamp Adaptation runs regardless of act_q."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Clamp Adaptation" in names
        assert "Activation Adaptation" not in names

    def test_activation_adaptation_for_rate_when_act_q_false(self):
        """Rate mode + act_q=False → Activation Adaptation (no clamp needed)."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": "rate",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Adaptation" in names
        assert "Clamp Adaptation" not in names

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
        fusion_idx = names.index("Normalization Fusion")
        # For ttfs, both act_q=True and act_q=False use Clamp Adaptation
        assert "Clamp Adaptation" in names
        clamp_idx = names.index("Clamp Adaptation")
        assert clamp_idx < fusion_idx


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

    @pytest.mark.parametrize("spiking_mode", ["ttfs", "ttfs_quantized"])
    def test_clamp_adaptation_for_ttfs_when_act_q_false(self, spiking_mode):
        """TTFS saturates at 1.0 → Clamp Adaptation runs even when act_q=False."""
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
        assert "Activation Adaptation" not in names

    def test_activation_adaptation_for_rate_when_act_q_false(self):
        """Rate mode + act_q=False → Activation Adaptation (no saturation)."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": "rate",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Analysis" in names
        assert "Activation Adaptation" in names
        assert "Clamp Adaptation" not in names

    def test_clamp_adaptation_present_for_torch_models_ttfs_when_act_q_false(self):
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
        assert "Activation Adaptation" not in names
        # Torch Mapping should come before adaptation
        assert names.index("Torch Mapping") < names.index("Activation Analysis")
