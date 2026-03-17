"""Pipeline step ordering: Activation Analysis + Activation Adaptation + optional Clamp.

Activation Adaptation always runs immediately after Activation Analysis (ReLU
replacement when needed, scales always). When act_q is True or spiking is
TTFS, Clamp Adaptation runs after Activation Adaptation.
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
    """Activation Adaptation always runs after Activation Analysis; Clamp when act_q or TTFS."""

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

    def test_activation_adaptation_follows_activation_analysis(self):
        """Activation Adaptation immediately follows Activation Analysis in all configs."""
        for spiking in ("rate", "ttfs"):
            for act_q in (True, False):
                config = {
                    "configuration_mode": "user",
                    "spiking_mode": spiking,
                    "activation_quantization": act_q,
                    "weight_quantization": False,
                    "model_type": "mlp_mixer",
                }
                names = _step_names(config)
                assert "Activation Analysis" in names
                assert "Activation Adaptation" in names
                idx_analysis = names.index("Activation Analysis")
                idx_adapt = names.index("Activation Adaptation")
                assert idx_adapt == idx_analysis + 1, (
                    f"Activation Adaptation should immediately follow Activation Analysis "
                    f"(spiking={spiking}, act_q={act_q})."
                )

    def test_clamp_adaptation_when_act_q_true(self):
        """act_q=True: both Activation Adaptation and Clamp Adaptation, in that order."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": True,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Adaptation" in names
        assert "Clamp Adaptation" in names
        assert names.index("Activation Adaptation") < names.index("Clamp Adaptation")

    def test_clamp_adaptation_for_ttfs_even_when_act_q_false(self):
        """TTFS + act_q=False: both Activation Adaptation and Clamp Adaptation, in that order."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Adaptation" in names
        assert "Clamp Adaptation" in names
        assert names.index("Activation Adaptation") < names.index("Clamp Adaptation")

    def test_activation_adaptation_for_rate_when_act_q_false(self):
        """Rate mode + act_q=False → Activation Adaptation only (no Clamp)."""
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
        assert "Activation Adaptation" in names
        assert names.index("Activation Adaptation") < fusion_idx
        assert "Clamp Adaptation" in names
        assert names.index("Clamp Adaptation") < fusion_idx


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
        """TTFS saturates at 1.0 → both Activation Adaptation and Clamp Adaptation."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": spiking_mode,
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Analysis" in names
        assert "Activation Adaptation" in names
        assert "Clamp Adaptation" in names
        assert names.index("Activation Adaptation") < names.index("Clamp Adaptation")

    def test_activation_adaptation_for_rate_when_act_q_false(self):
        """Rate mode + act_q=False → Activation Adaptation only (no saturation)."""
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
        assert "Activation Adaptation" in names
        assert "Clamp Adaptation" in names
        assert names.index("Activation Adaptation") < names.index("Clamp Adaptation")
        assert names.index("Torch Mapping") < names.index("Activation Analysis")
