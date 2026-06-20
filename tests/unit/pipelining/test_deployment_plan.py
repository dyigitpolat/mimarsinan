"""DeploymentPlan: the single resolver for the deployment_parameters config.

Locks the precedence / derivation rules (defaults, pruning_enabled,
schedule-derived spiking booleans, the budget default) and asserts the plan is
byte-identical to the inline ``config.get(...)`` reads it replaces.
"""

import pytest

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


def _resolve(**cfg):
    return DeploymentPlan.resolve(dict(cfg))


class TestDefaults:
    def test_empty_config_defaults(self):
        p = _resolve()
        assert p.spiking_mode == "lif"
        assert p.ttfs_cycle_schedule == "cascaded"
        assert p.activation_quantization is False
        assert p.weight_quantization is False
        assert p.enable_training_noise is False
        assert p.cycle_accurate_lif_forward is False
        assert p.pruning is False
        assert p.pruning_fraction == 0.0
        assert p.pruning_enabled is False
        assert p.enable_nevresim_simulation is True
        assert p.enable_loihi_simulation is False
        assert p.enable_sanafe_simulation is False
        assert p.cuda_debug is False
        assert p.deployment_metric_full_eval is True
        assert p.max_simulation_samples == 0
        assert p.simulation_batch_count is None
        assert p.simulation_batch_size == 8
        assert p.seed == 0
        assert p.weight_source is None
        assert p.model_type == ""

    def test_does_not_require_simulation_steps(self):
        # The step planner resolves a plan from a config without sim length.
        p = _resolve(spiking_mode="ttfs_cycle_based")
        assert p.requires_ttfs_firing is True


class TestSpikingDerived:
    def test_requires_ttfs_firing(self):
        assert _resolve(spiking_mode="lif").requires_ttfs_firing is False
        assert _resolve(spiking_mode="rate").requires_ttfs_firing is False
        for m in ("ttfs", "ttfs_quantized", "ttfs_cycle_based"):
            assert _resolve(spiking_mode=m).requires_ttfs_firing is True

    def test_schedule_normalized_and_default(self):
        assert _resolve().ttfs_cycle_schedule == "cascaded"
        assert _resolve(ttfs_cycle_schedule="bogus").ttfs_cycle_schedule == "cascaded"
        assert (
            _resolve(ttfs_cycle_schedule="synchronized").ttfs_cycle_schedule
            == "synchronized"
        )

    def test_synchronized_only_for_ttfs_cycle(self):
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="synchronized"
        ).is_synchronized_ttfs is True
        assert _resolve(
            spiking_mode="ttfs_cycle_based", ttfs_cycle_schedule="cascaded"
        ).is_synchronized_ttfs is False
        # schedule is inert for non-cycle modes.
        assert _resolve(
            spiking_mode="ttfs", ttfs_cycle_schedule="synchronized"
        ).is_synchronized_ttfs is False

    def test_is_ttfs_cycle_based(self):
        assert _resolve(spiking_mode="ttfs_cycle_based").is_ttfs_cycle_based is True
        assert _resolve(spiking_mode="lif").is_ttfs_cycle_based is False


class TestPruningDerivation:
    def test_pruning_enabled_requires_positive_fraction(self):
        assert _resolve(pruning=True, pruning_fraction=0.0).pruning_enabled is False
        assert _resolve(pruning=True, pruning_fraction=0.3).pruning_enabled is True
        assert _resolve(pruning=False, pruning_fraction=0.3).pruning_enabled is False


class TestToleranceAndBudget:
    def test_budget_defaults_to_twice_tolerance(self):
        assert _resolve(degradation_tolerance=0.05).degradation_budget_total == 0.10
        assert _resolve(degradation_tolerance=0.03).degradation_budget_total == 0.06

    def test_explicit_budget_overrides_default(self):
        assert (
            _resolve(degradation_tolerance=0.05, degradation_budget_total=0.42)
            .degradation_budget_total == 0.42
        )

    def test_scm_tolerance_optional(self):
        assert _resolve().scm_degradation_tolerance is None
        assert _resolve(scm_degradation_tolerance=0.02).scm_degradation_tolerance == 0.02


class TestModelIdentity:
    def test_model_name_falls_back_to_model_type(self):
        assert _resolve(model_type="mlp_mixer").model_name == "mlp_mixer"
        assert (
            _resolve(model_type="mlp_mixer", model_name="my_run").model_name == "my_run"
        )


class TestByteIdentityWithInlineReads:
    """Each resolved field must equal the inline read it replaces."""

    _CFGS = [
        {},
        {"spiking_mode": "ttfs", "activation_quantization": True},
        {
            "spiking_mode": "ttfs_cycle_based",
            "ttfs_cycle_schedule": "synchronized",
            "weight_quantization": True,
            "pruning": True,
            "pruning_fraction": 0.5,
            "enable_loihi_simulation": False,
            "enable_sanafe_simulation": True,
            "enable_nevresim_simulation": False,
            "enable_training_noise": True,
        },
        {
            "degradation_tolerance": 0.07,
            "scm_degradation_tolerance": 0.02,
            "cuda_debug": True,
            "deployment_metric_full_eval": False,
            "max_simulation_samples": 256,
            "simulation_batch_count": 4,
            "simulation_batch_size": 16,
            "seed": 7,
            "weight_source": "/tmp/w.pt",
            "model_type": "torch_custom",
            "cycle_accurate_lif_forward": True,
        },
    ]

    @pytest.mark.parametrize("cfg", _CFGS)
    def test_matches_inline_reads(self, cfg):
        p = DeploymentPlan.resolve(cfg)
        g = cfg.get
        assert p.spiking_mode == g("spiking_mode", "lif")
        assert p.activation_quantization == bool(g("activation_quantization", False))
        assert p.weight_quantization == bool(g("weight_quantization", False))
        assert p.pruning == g("pruning", False)
        assert p.pruning_fraction == float(g("pruning_fraction", 0.0))
        assert p.weight_source == g("weight_source")
        assert p.model_type == g("model_type", "")
        assert p.enable_loihi_simulation == bool(g("enable_loihi_simulation", False))
        assert p.enable_sanafe_simulation == bool(g("enable_sanafe_simulation", False))
        assert p.enable_nevresim_simulation == bool(
            g("enable_nevresim_simulation", True)
        )
        assert p.enable_training_noise == bool(g("enable_training_noise", False))
        assert p.cuda_debug == bool(g("cuda_debug", False))
        assert p.simulation_batch_size == int(g("simulation_batch_size", 8))
        assert p.seed == int(g("seed", 0))


class TestPipelineAccessor:
    def test_of_reads_pipeline_config(self):
        class _Stub:
            config = {"spiking_mode": "ttfs", "weight_quantization": True}

        p = DeploymentPlan.of(_Stub())
        assert p.spiking_mode == "ttfs"
        assert p.weight_quantization is True

    def test_spiking_contract_is_the_sub_part(self):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        cfg = {"spiking_mode": "lif", "simulation_steps": 32}
        contract = DeploymentPlan.resolve(cfg).spiking_contract()
        assert isinstance(contract, SpikingDeploymentContract)
        assert contract.spiking_mode == "lif"
        assert contract.simulation_steps == 32
