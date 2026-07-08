"""Unit tests for wizard config builder, validation, and run utilities."""

import json
import os

import pytest

from mimarsinan.config_schema import build_flat_pipeline_config, to_flat, to_namespaced
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.gui.runs import suggest_resume_step
from mimarsinan.gui.wizard import build_deployment_config_from_state, validate_wizard_state
from mimarsinan.gui.wizard.schema import (
    get_pipeline_step_names_for_config,
    get_wizard_defaults,
    get_wizard_model_types,
    get_wizard_nas_schema,
    get_wizard_temporal_allocation_schema,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_PER_LAYER_S_TEMPLATE = os.path.join(
    _REPO_ROOT, "templates", "mnist_mmixcore_per_layer_s_uniform.json"
)


def _resolved_flat(doc):
    return build_flat_pipeline_config(
        doc.get("deployment_parameters") or {},
        doc.get("platform_constraints") or {},
        pipeline_mode=doc.get("pipeline_mode", "phased"),
    )


class TestBuildDeploymentConfigFromState:
    def test_empty_state_gets_defaults(self):
        out = build_deployment_config_from_state({})
        assert out["data_provider_name"] == "MNIST_DataProvider"
        assert out["experiment_name"] == "experiment"
        assert out["generated_files_path"] == "./generated"
        assert "pipeline_mode" not in out
        assert out["deployment_parameters"] == {}
        assert out["platform_constraints"] == {}

    def test_phased_preset_applied(self):
        out = build_deployment_config_from_state({"pipeline_mode": "phased"})
        assert "activation_quantization" not in out["deployment_parameters"]
        assert _resolved_flat(out)["weight_quantization"] is True
        # Default spiking_mode is lif; the pipeline preconditions LIF with AQ.
        assert _resolved_flat(out)["activation_quantization"] is True

    def test_ttfs_quantized_enables_activation_quantization(self):
        out = build_deployment_config_from_state({
            "pipeline_mode": "phased",
            "deployment_parameters": {"spiking_mode": "ttfs_quantized"},
        })
        assert "activation_quantization" not in out["deployment_parameters"]
        assert _resolved_flat(out)["activation_quantization"] is True

    def test_vanilla_preset_no_quantization_flags(self):
        out = build_deployment_config_from_state({
            "pipeline_mode": "vanilla",
            "deployment_parameters": {"weight_quantization": False},
        })
        assert out["pipeline_mode"] == "vanilla"
        assert "activation_quantization" not in out["deployment_parameters"]
        assert _resolved_flat(out)["weight_quantization"] is False
        assert _resolved_flat(out)["activation_quantization"] is False

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
            "platform_constraints": {
                "cores": [{"max_axons": 512, "max_neurons": 512, "count": 200}],
            },
        }
        out = build_deployment_config_from_state(state)
        assert out["data_provider_name"] == "CIFAR10_DataProvider"
        assert out["experiment_name"] == "cifar_run"
        assert out["deployment_parameters"]["model_type"] == "mlp_mixer"
        assert out["platform_constraints"]["cores"][0]["max_axons"] == 512

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

    def test_enable_nevresim_simulation_default_true(self):
        out = build_deployment_config_from_state({})
        assert "enable_nevresim_simulation" not in out["deployment_parameters"]
        assert _resolved_flat(out)["enable_nevresim_simulation"] is True

    def test_declared_simulator_off_survives_emission(self):
        """Round-3 defect 6: user-off on a supported vehicle is a legitimate
        declarable override — emission preserves it (the recipe only owns the
        unset default)."""
        out = build_deployment_config_from_state({
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "x",
            "generated_files_path": "./out",
            "deployment_parameters": {
                "model_config_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {},
                "enable_nevresim_simulation": False,
            },
        })
        assert out["deployment_parameters"]["enable_nevresim_simulation"] is False
        assert validate_wizard_state(out) == []

    def test_explicit_keys_survive_and_owned_derived_keys_do_not(self):
        # Explicit-keys-only emission: what the draft declares survives
        # VERBATIM (declarable derived pins included — dropping them is how
        # endpoint_floor_wall_s was historically lost); only the keys the
        # derivation exclusively owns (activation_quantization) are removed.
        dp = get_default_deployment_parameters()
        dp.update({
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_type": "mlp_mixer",
            "model_config": {},
            "spiking_mode": "ttfs_quantized",
            "weight_quantization": True,
            "activation_quantization": True,
            "firing_mode": "TTFS",
            "spike_generation_mode": "TTFS",
            "thresholding_mode": "<=",
        })
        out = build_deployment_config_from_state({
            "pipeline_mode": "phased",
            "deployment_parameters": dp,
            "platform_constraints": get_default_platform_constraints(),
        })

        persisted = out["deployment_parameters"]
        assert "activation_quantization" not in persisted
        for key in ("firing_mode", "spike_generation_mode", "thresholding_mode",
                    "kd_ce_alpha", "kd_temperature"):
            assert persisted[key] == dp[key], key
        assert out["pipeline_mode"] == "phased"

    def test_minimal_persistence_resolves_byte_identical_to_bloated_config(self):
        bloated_dp = get_default_deployment_parameters()
        bloated_dp.update({
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_type": "mlp_mixer",
            "model_config": {},
            "spiking_mode": "ttfs_quantized",
            "weight_quantization": True,
            "activation_quantization": True,
            "pipeline_mode": "phased",
            "firing_mode": "TTFS",
            "spike_generation_mode": "TTFS",
            "thresholding_mode": "<=",
        })
        bloated = {
            "pipeline_mode": "phased",
            "deployment_parameters": bloated_dp,
            "platform_constraints": get_default_platform_constraints(),
        }

        minimal = build_deployment_config_from_state(bloated)

        assert json.dumps(_resolved_flat(minimal), sort_keys=True) == json.dumps(
            _resolved_flat(bloated), sort_keys=True
        )

    def test_lif_declarable_spiking_fields_survive_with_identical_resolution(self):
        bloated = {
            "pipeline_mode": "phased",
            "deployment_parameters": {
                "model_config_mode": "user",
                "hw_config_mode": "fixed",
                "model_type": "mlp_mixer",
                "model_config": {},
                "spiking_mode": "lif",
                "weight_quantization": True,
                "activation_quantization": True,
                "pipeline_mode": "phased",
                "firing_mode": "Default",
                "spike_generation_mode": "Uniform",
                "thresholding_mode": "<=",
            },
            "platform_constraints": get_default_platform_constraints(),
        }

        minimal = build_deployment_config_from_state(bloated)

        # Declarable spiking pins are consistent redundant declarations: kept
        # verbatim, and the resolved runtime config is byte-identical.
        assert minimal["deployment_parameters"]["thresholding_mode"] == "<="
        assert _resolved_flat(minimal)["thresholding_mode"] == "<="
        assert json.dumps(_resolved_flat(minimal), sort_keys=True) == json.dumps(
            _resolved_flat(bloated), sort_keys=True
        )

    def test_continue_from_run_id_preserved_for_edit_and_continue(self):
        source = "mnist_hard_all_lif_phased_deployment_run_20260520_094327"
        out = build_deployment_config_from_state({
            "experiment_name": "mnist_hard_all_lif",
            "start_step": "Simulation",
            "_continue_from_run_id": source,
        })
        assert out["_continue_from_run_id"] == source


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

    def test_ttfs_minimal_state_may_omit_derived_spiking_modes(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {
                "model_config_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {},
                "spiking_mode": "ttfs_quantized",
            },
            "start_step": None,
        }
        assert validate_wizard_state(state) == []

    def test_ttfs_explicit_wrong_spiking_modes_still_fail(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {
                "model_config_mode": "user",
                "model_type": "mlp_mixer",
                "model_config": {},
                "spiking_mode": "ttfs_quantized",
                "firing_mode": "Default",
            },
            "start_step": None,
        }
        assert any("firing_mode" in e for e in validate_wizard_state(state))


class TestWizardSchema:
    def test_get_pipeline_step_names_user_phased(self):
        config = {
            "model_config_mode": "user",
            "hw_config_mode": "fixed",
            "model_type": "mlp_mixer",
            "activation_quantization": True,
            "weight_quantization": True,
            "pruning": False,
            "spiking_mode": "lif",
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
        ids = {o["id"] for o in nas["optimizer_options"]}
        assert {"nsga2", "agent_evolve", "compilagent"} <= ids
        assert "common_fields" in nas

    def test_get_wizard_nas_schema_has_compilagent_fields(self):
        nas = get_wizard_nas_schema()
        assert "compilagent_fields" in nas
        fields = nas["compilagent_fields"]
        # Same shape contract as agent_evolve_fields so the wizard's
        # generic field renderer works without a special case. There is
        # intentionally no `primary_objective` field: every objective is
        # equally weighted in compilagent's multi-objective leaderboard.
        for required_key in (
            "model", "harness", "max_candidates", "max_continuations",
            "system_prompt_extra",
        ):
            assert required_key in fields, f"missing compilagent field {required_key!r}"
        assert "primary_objective" not in fields
        # harness options enumerate at least the canonical pair
        assert "pydantic_ai" in fields["harness"]["options"]

    def test_get_wizard_nas_schema_accuracy_evaluator_matches_wizard_ui(self):
        nas = get_wizard_nas_schema()
        opts = nas["common_fields"]["accuracy_evaluator"]["options"]
        assert opts == ["extrapolating", "fast"]

    def test_get_wizard_nas_schema_has_objective_options(self):
        nas = get_wizard_nas_schema()
        assert "objective_options" in nas
        assert len(nas["objective_options"]) == 8
        ids = [o["id"] for o in nas["objective_options"]]
        assert "estimated_accuracy" in ids
        assert "total_params" in ids
        assert "param_utilization_pct" in ids
        assert "fragmentation_pct" in ids


class TestWizardTemporalAllocationSurface:
    """EW2 — the per-layer-S axis is wizard-constructible (form surface + gate)."""

    def test_defaults_expose_temporal_allocation(self):
        defaults = get_wizard_defaults()
        ta = defaults["temporal_allocation"]
        assert ta["field"] == "s_allocation"
        assert ta["options"] == ["uniform", "explicit", "budget"]
        assert ta["default"] == "uniform"
        assert ta["capability_gate"] == "allow_per_layer_s"

    def test_capability_gate_is_in_platform_defaults(self):
        defaults = get_wizard_defaults()
        assert "allow_per_layer_s" in defaults["platform_constraints"]
        assert defaults["platform_constraints"]["allow_per_layer_s"] is False

    def test_temporal_allocation_schema_declares_reserved_inputs(self):
        ta = get_wizard_temporal_allocation_schema()
        assert ta["explicit_field"] == "s_allocation_explicit"
        assert ta["budget_field"] == "s_allocation_budget"
        assert ta["requires_capability_modes"] == ["explicit", "budget"]
        assert set(ta["budget_objective_keys"]) == {
            "max_energy_proxy", "max_latency_steps", "target",
        }

    def test_uniform_wizard_state_validates_without_capability(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {
                "model_config_mode": "user", "model_type": "mlp_mixer",
                "model_config": {}, "s_allocation": "uniform",
            },
            "start_step": None,
        }
        assert validate_wizard_state(state) == []

    def test_explicit_state_without_capability_fails(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {},
            "deployment_parameters": {
                "model_config_mode": "user", "model_type": "mlp_mixer",
                "model_config": {},
                "s_allocation": "explicit", "s_allocation_explicit": [4, 4],
            },
            "start_step": None,
        }
        errs = validate_wizard_state(state)
        assert any("reserved/not implemented" in e for e in errs)

    def test_explicit_state_with_capability_still_reserved(self):
        state = {
            "data_provider_name": "MNIST_DataProvider",
            "experiment_name": "t",
            "generated_files_path": "./out",
            "platform_constraints": {"allow_per_layer_s": True},
            "deployment_parameters": {
                "model_config_mode": "user", "model_type": "mlp_mixer",
                "model_config": {},
                "s_allocation": "explicit", "s_allocation_explicit": [4, 4],
            },
            "start_step": None,
        }
        assert any("reserved/not implemented" in e for e in validate_wizard_state(state))


class TestPerLayerSTemplate:
    """EW2 — the demonstration template is wizard-valid + build/round-trips."""

    @staticmethod
    def _doc():
        with open(_PER_LAYER_S_TEMPLATE) as fh:
            return json.load(fh)

    def test_template_exists_and_exercises_the_axis(self):
        doc = self._doc()
        dp = doc["deployment_parameters"]
        pc = doc["platform_constraints"]
        # Exercises the axis: explicit mode + the capability gate ON.
        assert dp["s_allocation"] == "explicit"
        assert pc["allow_per_layer_s"] is True
        # Byte-identical-runnable: every explicit S equals the global uniform S.
        assert all(s == pc["simulation_steps"] for s in dp["s_allocation_explicit"])

    def test_template_documents_reserved_axis_but_is_not_runnable_yet(self):
        assert any("reserved/not implemented" in e for e in validate_wizard_state(self._doc()))

    def test_template_builds_flat_pipeline_config(self):
        doc = self._doc()
        cfg = build_flat_pipeline_config(
            doc["deployment_parameters"], doc["platform_constraints"],
            pipeline_mode=doc.get("pipeline_mode", "phased"),
        )
        assert cfg["s_allocation"] == "explicit"
        assert cfg["allow_per_layer_s"] is True
        assert cfg["s_allocation_explicit"] == [4, 4, 4]

    def test_template_namespaced_roundtrips_byte_identical(self):
        doc = self._doc()
        cfg = build_flat_pipeline_config(
            doc["deployment_parameters"], doc["platform_constraints"],
            pipeline_mode=doc.get("pipeline_mode", "phased"),
        )
        assert to_flat(to_namespaced(cfg)) == cfg


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
