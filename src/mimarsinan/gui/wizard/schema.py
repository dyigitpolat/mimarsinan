"""Wizard schema: model-type, NAS, temporal-allocation, and pipeline-step surfaces for the frontend."""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.models.builders.wizard_schema import get_all_model_type_schemas
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.search.results import ALL_OBJECTIVES, ACCURACY_OBJECTIVE_NAME
from mimarsinan.tuning.orchestration.temporal_allocation import S_ALLOCATION_MODES


def get_wizard_model_types() -> List[Dict[str, Any]]:
    """Return model type id, label, description, and config_schema from each builder class."""
    return get_all_model_type_schemas()


def get_wizard_defaults() -> Dict[str, Any]:
    """Platform + NAS defaults for the wizard (SSOT with config_schema.defaults)."""
    nas = get_wizard_nas_schema()
    return {
        "platform_constraints": dict(get_default_platform_constraints()),
        "nas_common_fields": nas.get("common_fields", {}),
        "firing_modes_by_spiking": {
            "lif": ["Default", "Novena"],
            "ttfs": ["TTFS"],
            "ttfs_quantized": ["TTFS"],
            "ttfs_cycle_based": ["TTFS"],
        },
        "temporal_allocation": get_wizard_temporal_allocation_schema(),
    }


def get_wizard_temporal_allocation_schema() -> Dict[str, Any]:
    """Per-layer-S declaration surface (EW2) for the wizard form.

    Declares the s_allocation modes and the allow_per_layer_s capability gate;
    the wizard only declares intent, the per-depth S map derivation is downstream.
    """
    dp_defaults = get_default_deployment_parameters()
    return {
        "field": "s_allocation",
        "options": list(S_ALLOCATION_MODES),
        "default": dp_defaults.get("s_allocation", "uniform"),
        "capability_gate": "allow_per_layer_s",
        "explicit_field": "s_allocation_explicit",
        "budget_field": "s_allocation_budget",
        "budget_objective_keys": ["max_energy_proxy", "max_latency_steps", "target"],
        "requires_capability_modes": ["explicit", "budget"],
    }


def get_wizard_nas_schema() -> Dict[str, Any]:
    """Return NAS optimizer options and field schemas for arch_search."""
    return {
        "optimizer_options": [
            {"id": "nsga2", "label": "NSGA-II"},
            {"id": "agent_evolve", "label": "Agentic Evolution (LLM-based)"},
            {"id": "compilagent", "label": "Compilagent (LLM session)"},
        ],
        "common_fields": {
            "pop_size": {"type": "int", "default": 12, "min": 2, "max": 64, "doc": "Population size"},
            "generations": {"type": "int", "default": 5, "min": 1, "max": 100, "doc": "Generations"},
            "seed": {"type": "int", "default": 42, "min": 0, "max": 999999, "doc": "Random seed"},
            "warmup_fraction": {"type": "float", "default": 0.1, "min": 0, "max": 1, "doc": "Warmup fraction"},
            "training_batch_size": {"type": "int", "default": 1024, "min": 1, "max": 8192, "doc": "Batch size"},
            "accuracy_evaluator": {"type": "str", "default": "extrapolating", "options": ["extrapolating", "fast"], "doc": "Evaluator"},
            "extrapolation_num_train_epochs": {"type": "int", "default": 1, "min": 1, "max": 10, "doc": "Extrapolation train epochs"},
            "extrapolation_num_checkpoints": {"type": "int", "default": 5, "min": 1, "max": 20, "doc": "Extrapolation checkpoints"},
            "extrapolation_target_epochs": {"type": "int", "default": 10, "min": 1, "max": 100, "doc": "Extrapolation target epochs"},
        },
        "agent_evolve_fields": {
            "agent_model": {"type": "str", "default": "deepseek:deepseek-chat", "doc": "LLM model (pydantic-ai format)"},
            "candidates_per_batch": {"type": "int", "default": 5, "min": 1, "max": 20, "doc": "Candidates per batch"},
            "max_regen_rounds": {"type": "int", "default": 10, "min": 1, "max": 50, "doc": "Max regen rounds"},
            "max_failed_examples": {"type": "int", "default": 5, "min": 0, "max": 20, "doc": "Max failed examples"},
            "constraints_description": {"type": "textarea", "default": "", "doc": "Constraints for LLM"},
        },
        "compilagent_fields": {
            "model": {"type": "str", "default": "openai:gpt-4o", "doc": "LLM model id (provider:model)"},
            "harness": {"type": "str", "default": "pydantic_ai", "options": ["pydantic_ai", "claude_agent_sdk"], "doc": "Compilagent harness"},
            "max_candidates": {"type": "int", "default": 8, "min": 1, "max": 64, "doc": "Max candidates per session"},
            "max_continuations": {"type": "int", "default": 4, "min": 0, "max": 32, "doc": "Max continuation rounds"},
            "system_prompt_extra": {"type": "textarea", "default": "", "doc": "Extra system-prompt text (appended)"},
        },
        "objective_options": [
            {"id": o.name, "label": _objective_label(o.name), "goal": o.goal,
             "requires_training": o.name == ACCURACY_OBJECTIVE_NAME}
            for o in ALL_OBJECTIVES
        ],
    }


def _objective_label(name: str) -> str:
    labels = {
        "estimated_accuracy": "Estimated Accuracy",
        "total_params": "Total Parameters",
        "total_param_capacity": "Chip Capacity",
        "total_sync_barriers": "Sync Barriers",
        "param_utilization_pct": "Param Utilization %",
        "neuron_wastage_pct": "Neuron Wastage %",
        "axon_wastage_pct": "Axon Wastage %",
        "fragmentation_pct": "Fragmentation %",
    }
    return labels.get(name, name)


def get_pipeline_step_names_for_config(config: dict) -> List[str]:
    """Return ordered pipeline step names for the given config.

    Delegates to the single source of truth in ``deployment_pipeline``.
    """
    return [name for name, _ in get_pipeline_step_specs(config)]
