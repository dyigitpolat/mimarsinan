"""DeploymentPlan under core_semantics='mvm': domain-first dispatch, inert spiking axes."""

import pytest

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import (
    merge_pipeline_config,
)
from mimarsinan.pipelining.core.pipelines.deployment_specs import (
    get_pipeline_step_specs,
)


def _mvm_plan(**extra):
    cfg = {"core_semantics": "mvm", **extra}
    return DeploymentPlan.resolve(cfg)


class TestDomainAxis:
    def test_default_plan_is_spiking(self):
        p = DeploymentPlan.resolve({})
        assert p.core_semantics == "spiking"
        assert p.is_mvm is False
        assert p.spiking_mode == "lif"

    def test_mvm_plan_resolves(self):
        p = _mvm_plan()
        assert p.core_semantics == "mvm"
        assert p.is_mvm is True

    def test_spiking_axes_are_inert_under_mvm(self):
        # A merged config carries spiking defaults (e.g. spiking_mode='lif');
        # the plan neutralizes them so no spiking-gated step can arm.
        p = _mvm_plan(spiking_mode="lif", firing_mode="Default")
        assert p.spiking_mode not in ("lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based")
        assert p.requires_ttfs_firing is False
        assert p.is_ttfs_cycle_based is False
        assert p.is_synchronized_ttfs is False
        assert p.is_cascaded_ttfs is False
        assert p.uses_ttfs_floor_ceil_convention is False
        assert p.is_lif_style is False
        assert p.runs_cycle_accurate_activation_tuner is False
        assert p.requires_clamp_preconditioning is False
        assert p.requires_activation_quantization_preconditioning is False

    def test_novena_gate_is_skipped_under_mvm(self):
        # The chip-faithful-LIF-forward gate must not run for a value plan.
        p = _mvm_plan(firing_mode="Novena", cycle_accurate_lif_forward=False)
        assert p.is_mvm is True

    def test_native_builders_are_rejected(self):
        with pytest.raises(ValueError, match="torch-category"):
            _mvm_plan(model_type="simple_mlp")

    def test_unsafe_aq_override_cannot_arm_the_ladder(self):
        p = _mvm_plan(activation_quantization=True)
        assert p.requires_clamp_preconditioning is False
        assert p.requires_activation_quantization_preconditioning is False


class TestDomainDispatch:
    def test_mode_policy_is_mvm(self):
        from mimarsinan.chip_simulation.mvm_core_policy import MvmCorePolicy

        assert isinstance(_mvm_plan().mode_policy(), MvmCorePolicy)

    def test_conversion_recipe_is_the_mvm_recipe(self):
        recipe = _mvm_plan().conversion_recipe
        assert recipe.special_case == "mvm_value_domain"
        assert recipe.sim_enables == {
            "enable_nevresim_simulation": False,
            "enable_sanafe_simulation": False,
            "enable_loihi_simulation": False,
        }
        assert "lif_exact_qat" not in recipe.knobs
        assert recipe.knobs["wq_fast_rates"] == [0.5, 1.0]

    def test_spiking_contract_fails_loud(self):
        with pytest.raises(RuntimeError, match="mvm"):
            _mvm_plan().spiking_contract()


class TestMvmStepPlan:
    def test_step_list_for_an_mvm_config(self):
        document_dp = {
            "core_semantics": "mvm",
            "model_type": "lenet5",
            "model_config": {},
            "weight_quantization": True,
        }
        config = merge_pipeline_config(dict(document_dp), {"weight_bits": 8})
        names = [name for name, _cls in get_pipeline_step_specs(config)]
        assert names == [
            "Model Configuration",
            "Model Building",
            "Pretraining",
            "Torch Mapping",
            "Weight Quantization",
            "Quantization Verification",
            "Normalization Fusion",
            "Soft Core Mapping",
            "Core Quantization Verification",
            "Hard Core Mapping",
        ]

    def test_float_mvm_step_list_drops_quantization(self):
        document_dp = {
            "core_semantics": "mvm",
            "model_type": "lenet5",
            "model_config": {},
            "pipeline_mode": "vanilla",
        }
        config = merge_pipeline_config(dict(document_dp), {})
        names = [name for name, _cls in get_pipeline_step_specs(config)]
        assert names == [
            "Model Configuration",
            "Model Building",
            "Pretraining",
            "Torch Mapping",
            "Normalization Fusion",
            "Soft Core Mapping",
            "Hard Core Mapping",
        ]
