"""Unit tests for wizard config builder, validation, and run utilities."""

import pytest

from mimarsinan.gui.runs import suggest_resume_step
from mimarsinan.gui.wizard import build_deployment_config_from_state, validate_wizard_state
from mimarsinan.gui.wizard.flow import (
    WIZARD_STEP_IDS,
    get_step_index,
    get_next_step_id,
    get_previous_step_id,
)
from mimarsinan.gui.wizard.schema import (
    get_pipeline_step_names_for_config,
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
        assert out["deployment_parameters"]["model_config_mode"] == "user"
        assert out["deployment_parameters"]["hw_config_mode"] == "fixed"
        assert out["deployment_parameters"]["spiking_mode"] == "rate"
        assert out["deployment_parameters"].get("allow_scheduling") is False

    def test_phased_preset_applied(self):
        out = build_deployment_config_from_state({"pipeline_mode": "phased"})
        assert out["deployment_parameters"].get("activation_quantization") is True
        assert out["deployment_parameters"].get("weight_quantization") is True

    def test_vanilla_preset_no_quantization_flags(self):
        out = build_deployment_config_from_state({"pipeline_mode": "vanilla"})
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
                "model_config_mode": "user",
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

    def test_max_simulation_samples_preserved_when_set(self):
        state = {
            "deployment_parameters": {
                "model_config_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {},
                "max_simulation_samples": 200,
            },
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "x",
            "generated_files_path": "./out",
        }
        out = build_deployment_config_from_state(state)
        assert out["deployment_parameters"]["max_simulation_samples"] == 200

    def test_max_simulation_samples_absent_by_default_full_set(self):
        out = build_deployment_config_from_state({})
        assert "max_simulation_samples" not in out["deployment_parameters"]


class TestValidateWizardState:
    def test_valid_state_passes(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {"model_config_mode": "user", "model_type": "mlp_mixer", "model_config": {}},
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

    def test_next_after_search_toggles_goes_to_user_model(self):
        state = {"deployment_parameters": {"model_config_mode": "user"}}
        assert get_next_step_id(state, "search_toggles") == "user_model"

    def test_next_after_user_model_no_search(self):
        state = {"deployment_parameters": {"model_config_mode": "user", "hw_config_mode": "fixed"}}
        assert get_next_step_id(state, "user_model") == "platform_constraints"

    def test_next_after_user_model_with_model_search(self):
        state = {"deployment_parameters": {"model_config_mode": "search", "hw_config_mode": "fixed"}}
        assert get_next_step_id(state, "user_model") == "nas_options"

    def test_next_after_nas_options_hw_search_skips_platform(self):
        state = {"deployment_parameters": {"model_config_mode": "search", "hw_config_mode": "search"}}
        assert get_next_step_id(state, "nas_options") == "spiking_quantization"

    def test_next_after_nas_options_model_only_goes_to_platform(self):
        state = {"deployment_parameters": {"model_config_mode": "search", "hw_config_mode": "fixed"}}
        assert get_next_step_id(state, "nas_options") == "platform_constraints"

    def test_previous_from_user_model_goes_to_search_toggles(self):
        state = {"deployment_parameters": {"model_config_mode": "user"}}
        assert get_previous_step_id(state, "user_model") == "search_toggles"

    def test_previous_from_nas_options_goes_to_user_model(self):
        state = {"deployment_parameters": {"model_config_mode": "search"}}
        assert get_previous_step_id(state, "nas_options") == "user_model"


class TestWizardSchema:
    def test_get_pipeline_step_names_user_phased(self):
        config = {
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_type": "mlp_mixer",
            "activation_quantization": True,
            "weight_quantization": True,
            "pruning": False,
            "spiking_mode": "rate",
        }
        steps = get_pipeline_step_names_for_config(config)
        assert "Model Configuration" in steps
        assert "Model Building" in steps
        assert "Pretraining" in steps
        assert "Activation Analysis" in steps
        assert "Weight Quantization" in steps
        assert "Simulation" in steps

    def test_get_pipeline_step_names_model_search(self):
        config = {"model_config_mode": "search", "hw_config_mode": "fixed"}
        steps = get_pipeline_step_names_for_config(config)
        assert steps[0] == "Architecture Search"
        assert "Simulation" in steps

    def test_get_pipeline_step_names_hw_search(self):
        config = {"model_config_mode": "user", "hw_config_mode": "search"}
        steps = get_pipeline_step_names_for_config(config)
        assert steps[0] == "Architecture Search"

    def test_get_pipeline_step_names_joint_search(self):
        config = {"model_config_mode": "search", "hw_config_mode": "search"}
        steps = get_pipeline_step_names_for_config(config)
        assert steps[0] == "Architecture Search"

    def test_get_wizard_model_types_has_mlp_mixer_and_torch_vit(self):
        types = get_wizard_model_types()
        ids = [t["id"] for t in types]
        assert "mlp_mixer" in ids
        assert "torch_vit" in ids
        assert "torch_sequential_linear" in ids

    def test_get_wizard_nas_schema_has_optimizers(self):
        nas = get_wizard_nas_schema()
        assert "optimizer_options" in nas
        assert any(o["id"] == "nsga2" for o in nas["optimizer_options"])
        assert "common_fields" in nas

    def test_get_wizard_nas_schema_has_objective_options(self):
        nas = get_wizard_nas_schema()
        assert "objective_options" in nas
        assert len(nas["objective_options"]) == 7
        ids = [o["id"] for o in nas["objective_options"]]
        assert "estimated_accuracy" in ids
        assert "total_params" in ids
        assert "param_utilization_pct" in ids


class TestSuggestResumeStep:
    def test_returns_first_incomplete_step(self):
        steps = ["Training", "Quantization", "Simulation"]
        completed = {"Training"}
        assert suggest_resume_step(steps, completed) == "Quantization"

    def test_returns_first_step_when_none_completed(self):
        steps = ["Training", "Quantization"]
        assert suggest_resume_step(steps, set()) == "Training"

    def test_returns_none_when_all_completed(self):
        steps = ["Training", "Quantization"]
        assert suggest_resume_step(steps, {"Training", "Quantization"}) is None

    def test_returns_none_for_empty_steps(self):
        assert suggest_resume_step([], {"Training"}) is None

    def test_preserves_canonical_order(self):
        steps = ["A", "B", "C", "D"]
        completed = {"A", "C"}
        assert suggest_resume_step(steps, completed) == "B"

    def test_step_not_in_completed_set_is_suggested(self):
        steps = ["Training", "Quantization", "Simulation"]
        completed = {"Training", "Quantization"}
        assert suggest_resume_step(steps, completed) == "Simulation"
