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
        assert "tuner_epochs" in d
        assert "degradation_tolerance" in d
        assert "model_config_mode" in d
        assert "hw_config_mode" in d
        assert "spiking_mode" in d
        assert "allow_scheduling" in d
        assert d["allow_scheduling"] is False

    def test_default_platform_constraints_has_required_keys(self):
        d = get_default_platform_constraints()
        assert "max_axons" in d
        assert "max_neurons" in d
        assert "target_tq" in d
        assert "simulation_steps" in d
        assert "weight_bits" in d
        assert "allow_axon_tiling" in d

    def test_pipeline_mode_presets_has_vanilla_and_phased(self):
        presets = get_pipeline_mode_presets()
        assert "vanilla" in presets
        assert "phased" in presets
        assert presets["phased"].get("activation_quantization") is True
        assert presets["phased"].get("weight_quantization") is True


class TestApplyPreset:
    """apply_preset merges preset with setdefault semantics."""

    def test_phased_sets_quantization_flags(self):
        params = {}
        apply_preset("phased", params)
        assert params.get("activation_quantization") is True
        assert params.get("weight_quantization") is True

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


class TestValidateMergedConfig:
    """validate_merged_config checks flat dict (runtime config)."""

    def test_valid_merged_config_passes(self):
        flat = {
            "lr": 0.001,
            "training_epochs": 10,
            "max_axons": 256,
            "max_neurons": 256,
            "target_tq": 32,
            "spiking_mode": "rate",
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


class TestConfigKeysSet:
    """Config keys set includes all keys read by pipeline consumers."""

    def test_has_deployment_and_platform_keys(self):
        keys = get_config_keys_set()
        assert "lr" in keys
        assert "max_axons" in keys
        assert "target_tq" in keys
        assert "spiking_mode" in keys
        assert "model_type" in keys
        assert "model_config" in keys
