"""Pipeline step ordering: activation preconditioning before spiking tuning.

Activation Adaptation always runs after Activation Analysis. Clamp/Shift/
Activation Quantization precondition cycle-accurate LIF and TTFS-cycle tuning;
for analytical/rate modes the activation-quantization flag remains the gate.
"""

import pytest

from mimarsinan.pipelining.core.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.pipelining.pipeline_steps.adaptation.activation_analysis_step import ActivationAnalysisStep
from mimarsinan.pipelining.pipeline_steps.adaptation.activation_adaptation_step import ActivationAdaptationStep
from mimarsinan.pipelining.pipeline_steps.adaptation.clamp_adaptation_step import ClampAdaptationStep
from mimarsinan.pipelining.pipeline_steps.adaptation.activation_shift_step import ActivationShiftStep
from mimarsinan.pipelining.pipeline_steps.quantization.activation_quantization_step import ActivationQuantizationStep


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

    def test_lif_preconditioning_runs_before_lif_adaptation(self):
        """LIF mode: analytical preconditioning runs before the LIF tuning ramp."""
        config = {
            "configuration_mode": "user",
            "spiking_mode": "lif",
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        expected = [
            "Activation Analysis",
            "Activation Adaptation",
            "Clamp Adaptation",
            "Activation Shifting",
            "Activation Quantization",
            "LIF Adaptation",
        ]
        assert [name for name in names if name in expected] == expected

    def _ttfs_cycle_config(self, **overrides):
        config = {
            "configuration_mode": "user",
            "spiking_mode": "ttfs_cycle_based",
            "activation_quantization": True,
            "weight_quantization": True,
            "model_type": "mlp_mixer",
        }
        config.update(overrides)
        return config

    def test_ttfs_cycle_preconditioning_runs_before_cycle_finetuning(self):
        """TTFS-cycle tuning is preconditioned by the analytical chain first."""
        names = _step_names(self._ttfs_cycle_config(activation_quantization=False))
        expected = [
            "Activation Analysis",
            "Activation Adaptation",
            "Clamp Adaptation",
            "Activation Shifting",
            "Activation Quantization",
            "TTFS Cycle Fine-Tuning",
        ]
        assert [name for name in names if name in expected] == expected
        assert names.index("TTFS Cycle Fine-Tuning") < names.index("Weight Quantization")

    def test_ttfs_cycle_synchronized_disables_nevresim_simulation(self):
        # nevresim has no genuine synchronized-window backend yet; the "Simulation"
        # (nevresim) step is skipped for the synchronized schedule only.
        names = _step_names(self._ttfs_cycle_config(
            enable_nevresim_simulation=True, ttfs_cycle_schedule="synchronized",
        ))
        assert "Simulation" not in names

    def test_ttfs_cycle_cascaded_keeps_nevresim_simulation(self):
        # Cascaded greedy TTFS runs genuinely on nevresim (fire-once-latch policy).
        names = _step_names(self._ttfs_cycle_config(
            enable_nevresim_simulation=True, ttfs_cycle_schedule="cascaded",
        ))
        assert "Simulation" in names
        # default schedule is cascaded → nevresim stays enabled.
        default_names = _step_names(
            self._ttfs_cycle_config(enable_nevresim_simulation=True)
        )
        assert "Simulation" in default_names
        # other modes keep nevresim Simulation.
        lif_names = _step_names({
            "configuration_mode": "user", "spiking_mode": "lif",
            "activation_quantization": False, "weight_quantization": True,
            "model_type": "mlp_mixer", "enable_nevresim_simulation": True,
        })
        assert "Simulation" in lif_names

    @pytest.mark.parametrize("spiking", ["ttfs", "ttfs_quantized", "lif", "rate"])
    def test_ttfs_cycle_finetuning_absent_for_other_modes(self, spiking):
        config = {
            "configuration_mode": "user",
            "spiking_mode": spiking,
            "activation_quantization": True,
            "weight_quantization": True,
            "model_type": "mlp_mixer",
        }
        assert "TTFS Cycle Fine-Tuning" not in _step_names(config)

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
    """Default-off activation quantization still applies outside cycle-based modes."""

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

    @pytest.mark.parametrize("spiking_mode", ["lif", "ttfs_cycle_based"])
    def test_cycle_based_modes_force_quant_preconditioning_even_when_flag_is_off(self, spiking_mode):
        config = {
            "configuration_mode": "user",
            "spiking_mode": spiking_mode,
            "activation_quantization": False,
            "weight_quantization": False,
            "model_type": "mlp_mixer",
        }
        names = _step_names(config)
        assert "Activation Shifting" in names
        assert "Activation Quantization" in names


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


class TestSimulationStepToggles:
    """Optional simulation backends controlled by enable_* flags."""

    _BASE = {
        "configuration_mode": "user",
        "spiking_mode": "lif",
        "activation_quantization": False,
        "weight_quantization": False,
        "model_type": "mlp_mixer",
    }

    def test_simulation_present_by_default(self):
        names = _step_names(self._BASE)
        assert "Simulation" in names

    def test_nevresim_disabled_omits_simulation_step(self):
        config = {**self._BASE, "enable_nevresim_simulation": False}
        names = _step_names(config)
        assert "Simulation" not in names

    def test_nevresim_off_loihi_still_present_when_enabled(self):
        config = {
            **self._BASE,
            "enable_nevresim_simulation": False,
            "enable_loihi_simulation": True,
        }
        names = _step_names(config)
        assert "Simulation" not in names
        assert "Loihi Simulation" in names

    def test_nevresim_off_sanafe_still_present_when_enabled(self):
        config = {
            **self._BASE,
            "enable_nevresim_simulation": False,
            "enable_sanafe_simulation": True,
        }
        names = _step_names(config)
        assert "Simulation" not in names
        assert "SANA-FE Simulation" in names
