"""
Wizard schema: delegates model types to builders; NAS schema and pipeline step names.

Used by GET /api/wizard/schema and POST /api/wizard/pipeline-steps so the
frontend can render full-fidelity forms. Model config schema comes from
model builders (WIZARD_SCHEMA on each builder class).
"""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.models.builders.wizard_schema import get_all_model_type_schemas
from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.pipelining.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.search.results import ALL_OBJECTIVES, ACCURACY_OBJECTIVE_NAME


def get_wizard_model_types() -> List[Dict[str, Any]]:
    """Return model type id, label, description, and config_schema from each builder class."""
    return get_all_model_type_schemas()


def get_wizard_nas_schema() -> Dict[str, Any]:
    """Return NAS optimizer options and field schemas for arch_search."""
    return {
        "optimizer_options": [
            {"id": "nsga2", "label": "NSGA-II"},
            {"id": "agent_evolve", "label": "Agentic Evolution (LLM-based)"},
        ],
        "common_fields": {
            "pop_size": {"type": "int", "default": 8, "min": 2, "max": 64, "doc": "Population size"},
            "generations": {"type": "int", "default": 15, "min": 1, "max": 100, "doc": "Generations"},
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
