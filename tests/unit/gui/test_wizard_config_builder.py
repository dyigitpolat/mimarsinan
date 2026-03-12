"""Unit tests for wizard config builder and validation."""

import pytest

from mimarsinan.gui.wizard import build_deployment_config_from_state, validate_wizard_state
from mimarsinan.gui.wizard.flow import (
    WIZARD_STEP_IDS,
    get_step_index,
    get_next_step_id,
    get_previous_step_id,
)
from mimarsinan.gui.wizard.schema import (
    get_pipeline_step_names_for_state,
    get_wizard_model_types,
    get_wizard_nas_schema,
)


class TestBuildDeploymentConfigFromState:
    def test_empty_state_gets_defaults(self):
        out = build_deployment_config_from_state({})
        assert out["data_provider_name"] == "MNIST_DataProvider"
        assert out["experiment_name"] == "experiment"
        assert out["generated_files_path"] == "./generated"
        assert out["pipeline_mode"] == "phased"
        assert "deployment_parameters" in out
        assert "platform_constraints" in out
        assert out["deployment_parameters"]["configuration_mode"] == "user"
        assert out["deployment_parameters"]["spiking_mode"] == "rate"

    def test_phased_preset_applied(self):
        out = build_deployment_config_from_state({"pipeline_mode": "phased"})
        assert out["deployment_parameters"].get("activation_quantization") is True
        assert out["deployment_parameters"].get("weight_quantization") is True

    def test_vanilla_preset_no_quantization_flags(self):
        out = build_deployment_config_from_state({"pipeline_mode": "vanilla"})
        # setdefault only adds if missing; defaults don't have act/wt quant
        assert "activation_quantization" not in out["deployment_parameters"] or out["deployment_parameters"]["activation_quantization"] is not True

    def test_state_not_mutated(self):
        state = {"experiment_name": "my_run", "deployment_parameters": {"lr": 0.01}}
        out = build_deployment_config_from_state(state)
        assert state["experiment_name"] == "my_run"
        assert state["deployment_parameters"]["lr"] == 0.01
        assert out["experiment_name"] == "my_run"
        assert out["deployment_parameters"]["lr"] == 0.01

    def test_full_state_preserved(self):
        state = {
            "data_provider_name": "CIFAR10_DataProvider",
            "experiment_name": "cifar_run",
            "generated_files_path": "/out",
            "seed": 42,
            "pipeline_mode": "phased",
            "start_step": None,
            "stop_step": None,
            "target_metric_override": None,
            "deployment_parameters": {
                "configuration_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {"patch_n_1": 4, "patch_m_1": 4, "patch_c_1": 16, "fc_w_1": 32, "fc_w_2": 32},
            },
            "platform_constraints": {"max_axons": 512, "max_neurons": 512},
        }
        out = build_deployment_config_from_state(state)
        assert out["data_provider_name"] == "CIFAR10_DataProvider"
        assert out["experiment_name"] == "cifar_run"
        assert out["deployment_parameters"]["model_type"] == "mlp_mixer"
        assert out["platform_constraints"]["max_axons"] == 512

    def test_platform_constraints_mode_user_gets_user_key(self):
        state = {
            "platform_constraints": {"mode": "user", "max_axons": 128, "max_neurons": 128},
            "deployment_parameters": {},
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "x",
            "generated_files_path": "./out",
            "start_step": None,
        }
        out = build_deployment_config_from_state(state)
        assert out["platform_constraints"]["mode"] == "user"
        assert "user" in out["platform_constraints"]
        assert out["platform_constraints"]["user"]["max_axons"] == 128

    def test_platform_constraints_mode_auto_preserves_auto(self):
        state = {
            "platform_constraints": {
                "mode": "auto",
                "auto": {"fixed": {"target_tq": 16}, "search_space": {"num_core_types": 2}},
            },
            "deployment_parameters": {},
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "x",
            "generated_files_path": "./out",
            "start_step": None,
        }
        out = build_deployment_config_from_state(state)
        assert out["platform_constraints"]["mode"] == "auto"
        assert out["platform_constraints"]["auto"]["fixed"]["target_tq"] == 16
        assert out["platform_constraints"]["auto"]["search_space"]["num_core_types"] == 2


class TestValidateWizardState:
    def test_valid_state_passes(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {"configuration_mode": "user", "model_type": "mlp_mixer", "model_config": {}},
            "start_step": None,
        }
        assert validate_wizard_state(state) == []

    def test_empty_state_fails(self):
        errs = validate_wizard_state({})
        assert len(errs) >= 1
        assert "empty" in errs[0].lower() or "missing" in errs[0].lower()

    def test_missing_required_keys_fails(self):
        errs = validate_wizard_state({"experiment_name": "x"})
        assert any("data_provider_name" in e.lower() for e in errs)


class TestWizardFlow:
    def test_step_ids_ordered(self):
        assert len(WIZARD_STEP_IDS) >= 5
        assert WIZARD_STEP_IDS[0] == "experiment_basics"
        assert "review" in WIZARD_STEP_IDS

    def test_get_step_index(self):
        assert get_step_index("experiment_basics") == 0
        assert get_step_index("review") == len(WIZARD_STEP_IDS) - 1
        assert get_step_index("nonexistent") == -1

    def test_next_after_configuration_mode_user(self):
        state = {"deployment_parameters": {"configuration_mode": "user"}}
        assert get_next_step_id(state, "configuration_mode") == "user_model"

    def test_next_after_configuration_mode_nas(self):
        state = {"deployment_parameters": {"configuration_mode": "nas"}}
        assert get_next_step_id(state, "configuration_mode") == "nas_options"

    def test_previous_from_user_model_goes_to_configuration_mode(self):
        state = {"deployment_parameters": {"configuration_mode": "user"}}
        assert get_previous_step_id(state, "user_model") == "configuration_mode"

    def test_previous_from_nas_options_goes_to_configuration_mode(self):
        state = {"deployment_parameters": {"configuration_mode": "nas"}}
        assert get_previous_step_id(state, "nas_options") == "configuration_mode"


class TestWizardSchema:
    def test_get_pipeline_step_names_user_phased(self):
        state = {
            "deployment_parameters": {
                "configuration_mode": "user",
                "model_type": "mlp_mixer",
                "activation_quantization": True,
                "weight_quantization": True,
                "pruning": False,
                "spiking_mode": "rate",
            },
        }
        steps = get_pipeline_step_names_for_state(state)
        assert "Model Configuration" in steps
        assert "Model Building" in steps
        assert "Pretraining" in steps
        assert "Activation Analysis" in steps
        assert "Weight Quantization" in steps
        assert "Simulation" in steps

    def test_get_pipeline_step_names_nas(self):
        state = {"deployment_parameters": {"configuration_mode": "nas"}}
        steps = get_pipeline_step_names_for_state(state)
        assert steps[0] == "Architecture Search"
        assert "Simulation" in steps

    def test_get_wizard_model_types_has_mlp_mixer_and_vit(self):
        types = get_wizard_model_types()
        ids = [t["id"] for t in types]
        assert "mlp_mixer" in ids
        assert "vit" in ids
        assert "torch_sequential_linear" in ids

    def test_get_wizard_nas_schema_has_optimizers(self):
        nas = get_wizard_nas_schema()
        assert "optimizer_options" in nas
        assert any(o["id"] == "nsga2" for o in nas["optimizer_options"])
        assert "common_fields" in nas
