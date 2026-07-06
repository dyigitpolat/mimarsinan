"""Unit tests for config_schema: defaults, validation, apply_preset."""

import pytest

from mimarsinan.config_schema import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
    get_pipeline_mode_presets,
    apply_preset,
    validate_deployment_config,
    validate_merged_config,
    get_config_keys_set,
)

class TestDefaults:
    """Default dicts match pipeline expectations and are complete."""

    def test_default_deployment_parameters_has_required_keys(self):
        d = get_default_deployment_parameters()
        assert "lr" in d
        assert "training_epochs" in d
        assert "tuning_budget_scale" in d
        assert "degradation_tolerance" in d
        assert "model_config_mode" in d
        assert "hw_config_mode" in d
        assert "spiking_mode" in d
        assert "allow_scheduling" in d
        assert d["allow_scheduling"] is False

    def test_default_platform_constraints_has_required_keys(self):
        d = get_default_platform_constraints()
        assert "cores" in d
        assert isinstance(d["cores"], list)
        assert len(d["cores"]) > 0
        assert "max_axons" in d["cores"][0]
        assert "max_neurons" in d["cores"][0]
        assert "target_tq" in d
        assert "simulation_steps" in d
        assert "weight_bits" in d
        assert "allow_coalescing" in d

    def test_pipeline_mode_presets_has_vanilla_and_phased(self):
        presets = get_pipeline_mode_presets()
        assert "vanilla" in presets
        assert "phased" in presets

    def test_presets_never_inject_quant_flags(self):
        # Derivation owns AQ/WQ; a preset-injected value would masquerade as an
        # explicit one under the quantization contract.
        for preset in get_pipeline_mode_presets().values():
            assert "activation_quantization" not in preset
            assert "weight_quantization" not in preset


class TestApplyPreset:
    """apply_preset merges preset with setdefault semantics."""

    def test_phased_leaves_quant_keys_to_derivation(self):
        params = {}
        apply_preset("phased", params)
        assert "activation_quantization" not in params
        assert "weight_quantization" not in params

    def test_vanilla_leaves_empty(self):
        params = {"lr": 0.01}
        apply_preset("vanilla", params)
        assert params["lr"] == 0.01
        assert "activation_quantization" not in params

    def test_explicit_user_value_wins(self):
        params = {"activation_quantization": False}
        apply_preset("phased", params)
        assert params["activation_quantization"] is False


class TestValidateDeploymentConfig:
    """validate_deployment_config checks JSON shape main.py expects."""

    def test_valid_minimal_config_passes(self):
        cfg = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {"max_axons": 256, "max_neurons": 256},
            "deployment_parameters": {"model_config_mode": "user", "model_type": "mlp_mixer", "model_config": {}},
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert errors == []

    def test_missing_data_provider_name_fails(self):
        cfg = {
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {},
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert any("data_provider_name" in str(e).lower() for e in errors)

    def test_missing_deployment_parameters_fails(self):
        cfg = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert any("deployment_parameters" in str(e).lower() for e in errors)

    def test_user_mode_requires_model_type_and_model_config(self):
        cfg = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {"model_config_mode": "user"},
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert any("model_type" in str(e).lower() or "model_config" in str(e).lower() for e in errors)

    def test_search_mode_requires_arch_search(self):
        cfg = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {"model_config_mode": "search"},
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert any("arch_search" in str(e).lower() for e in errors)

    def test_deprecated_coalescing_key_in_platform_constraints_fails(self):
        cfg = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {
                "allow_core_coalescing": True,
                "cores": [{"max_axons": 8, "max_neurons": 8, "count": 1}],
            },
            "deployment_parameters": {
                "model_config_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {},
            },
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert any("not supported" in e and "allow_coalescing" in e for e in errors)

    def test_ttfs_requires_ttfs_firing_and_spike_gen(self):
        cfg = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {
                "model_config_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {"patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 16, "fc_w_1": 32, "fc_w_2": 32},
                "spiking_mode": "ttfs",
                "firing_mode": "Default",
                "spike_generation_mode": "Deterministic",
            },
            "start_step": None,
        }
        errors = validate_deployment_config(cfg)
        assert any("ttfs" in str(e).lower() or "firing" in str(e).lower() for e in errors)


class TestSAllocationValidation:
    """EW2 — only ``uniform`` is wired; ``explicit``/``budget`` loud-reject (Q2 foot-gun).

    The reserved modes would silently no-op to uniform, so validation rejects them at
    config-validation time, BEFORE the silent-uniform resolver path is reachable.
    """

    @staticmethod
    def _cfg(dp_extra=None, pc_extra=None):
        dp = {"model_config_mode": "user", "model_type": "mlp_mixer", "model_config": {}}
        if dp_extra:
            dp.update(dp_extra)
        pc = {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 1}]}
        if pc_extra:
            pc.update(pc_extra)
        return {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "test",
            "generated_files_path": "./out",
            "platform_constraints": pc,
            "deployment_parameters": dp,
            "start_step": None,
        }

    def test_uniform_default_is_valid_and_ungated(self):
        # Default (no s_allocation key) => uniform => always valid, no capability needed.
        assert validate_deployment_config(self._cfg()) == []
        assert validate_deployment_config(self._cfg({"s_allocation": "uniform"})) == []

    def test_unknown_mode_fails(self):
        errs = validate_deployment_config(self._cfg({"s_allocation": "magic"}))
        assert any("s_allocation must be one of" in e for e in errs)

    def test_explicit_is_rejected_as_reserved(self):
        # Reserved/not-implemented => loud-reject regardless of capability/shape.
        errs = validate_deployment_config(self._cfg(
            {"s_allocation": "explicit", "s_allocation_explicit": [4, 4]},
        ))
        assert any(
            "s_allocation='explicit' is reserved/not implemented" in e
            and "only 'uniform' is supported" in e
            for e in errs
        ), errs

    def test_budget_is_rejected_as_reserved(self):
        errs = validate_deployment_config(self._cfg(
            {"s_allocation": "budget", "s_allocation_budget": {"target": 0.96}},
        ))
        assert any(
            "s_allocation='budget' is reserved/not implemented" in e
            and "only 'uniform' is supported" in e
            for e in errs
        ), errs

    def test_explicit_rejected_even_with_capability_and_valid_list(self):
        # The capability gate + a well-formed list do NOT unlock a reserved mode.
        errs = validate_deployment_config(self._cfg(
            {"s_allocation": "explicit", "s_allocation_explicit": [4, 4, 8]},
            {"allow_per_layer_s": True},
        ))
        assert any("s_allocation='explicit' is reserved/not implemented" in e
                   for e in errs), errs

    def test_budget_rejected_even_with_capability_and_objective(self):
        for body in (
            {"max_energy_proxy": 1.0},
            {"max_latency_steps": 64},
            {"target": 0.96},
        ):
            errs = validate_deployment_config(self._cfg(
                {"s_allocation": "budget", "s_allocation_budget": body},
                {"allow_per_layer_s": True},
            ))
            assert any("s_allocation='budget' is reserved/not implemented" in e
                       for e in errs), body

    def test_explicit_rejected_in_wrapped_user_platform(self):
        # Wrapped {mode:'user', user:{...}} platform shape still loud-rejects explicit.
        cfg = self._cfg({"s_allocation": "explicit", "s_allocation_explicit": [4]})
        cfg["platform_constraints"] = {
            "mode": "user",
            "user": {"cores": [{"max_axons": 8, "max_neurons": 8, "count": 1}],
                     "allow_per_layer_s": True},
        }
        errs = validate_deployment_config(cfg)
        assert any("s_allocation='explicit' is reserved/not implemented" in e
                   for e in errs), errs

    def test_s_allocation_config_errors_is_exported_and_rejects_reserved(self):
        from mimarsinan.config_schema import s_allocation_config_errors
        # uniform passes; explicit/budget loud-reject.
        assert s_allocation_config_errors({"s_allocation": "uniform"}, {}) == []
        for mode in ("explicit", "budget"):
            errs = s_allocation_config_errors(
                {"s_allocation": mode}, {"allow_per_layer_s": True},
            )
            assert any(f"s_allocation={mode!r} is reserved/not implemented" in e
                       for e in errs), mode


class TestValidateMergedConfig:
    """validate_merged_config checks flat dict (runtime config)."""

    def test_valid_merged_config_passes(self):
        flat = {
            "lr": 0.001,
            "training_epochs": 10,
            "max_axons": 256,
            "max_neurons": 256,
            "target_tq": 32,
            "spiking_mode": "lif",
            "firing_mode": "Default",
            "spike_generation_mode": "Deterministic",
            "model_config_mode": "user",
            "model_type": "mlp_mixer",
            "model_config": {"patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 16, "fc_w_1": 32, "fc_w_2": 32},
        }
        errors = validate_merged_config(flat)
        assert errors == []

    def test_merged_ttfs_inconsistency_fails(self):
        flat = {
            "spiking_mode": "ttfs",
            "firing_mode": "Default",
            "spike_generation_mode": "Deterministic",
        }
        errors = validate_merged_config(flat)
        assert len(errors) >= 1

    def test_merged_ttfs_cycle_based_requires_ttfs_firing(self):
        flat = {
            "spiking_mode": "ttfs_cycle_based",
            "firing_mode": "Default",
            "spike_generation_mode": "Deterministic",
        }
        errors = validate_merged_config(flat)
        assert len(errors) >= 1

    def test_merged_ttfs_cycle_based_valid_passes(self):
        flat = {
            "spiking_mode": "ttfs_cycle_based",
            "firing_mode": "TTFS",
            "spike_generation_mode": "TTFS",
        }
        errors = validate_merged_config(flat)
        assert errors == []


class TestConfigKeysSet:
    """Config keys set includes all keys read by pipeline consumers."""

    def test_has_deployment_and_platform_keys(self):
        keys = get_config_keys_set()
        assert "lr" in keys
        assert "cores" in keys
        assert "target_tq" in keys
        assert "spiking_mode" in keys
        assert "model_type" in keys
        assert "model_config" in keys
