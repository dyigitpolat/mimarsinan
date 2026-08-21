"""Wizard schema: model-type, NAS, temporal-allocation, and pipeline-step surfaces for the frontend."""

from __future__ import annotations

from typing import Any, Dict, List

from mimarsinan.models.builders.wizard_schema import get_all_model_type_schemas
from mimarsinan.config_schema.defaults import (
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.chip_simulation.spiking_semantics import legal_firing_modes
from mimarsinan.config_schema.registry.entries_semantics import SPIKING_MODES
from mimarsinan.deployment_record.objectives import OBJECTIVES, SEARCH_MODES
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import get_pipeline_step_specs
from mimarsinan.search.optimizers.catalog import optimizer_options
from mimarsinan.search.optimizers.sampling_optimizer import DEFAULT_GRID_CAP
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
        # The legality SSOT answers this, never a second table here.
        "firing_modes_by_spiking": {
            mode: list(legal_firing_modes(mode)) for mode in SPIKING_MODES
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
        # [TS2] The catalogue itself — one list of backends, read by the wizard,
        # the factory's builder table and the declared OptimizerType alike.
        "optimizer_options": optimizer_options(),
        "common_fields": {
            "pop_size": {"type": "int", "default": 12, "min": 2, "max": 64, "doc": "Population size"},
            "generations": {"type": "int", "default": 5, "min": 1, "max": 100, "doc": "Generations"},
            "seed": {"type": "int", "default": 42, "min": 0, "max": 999999, "doc": "Random seed"},
            # [TS1] No default: an unset budget is an UNMETERED run, and the
            # wizard's empty field is how a run declares that.
            "evaluation_budget": {"type": "int", "min": 1, "max": 100000, "doc": "Evaluation budget (distinct evaluations; empty = unmetered)"},
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
        # [TS2] The exhaustive backend's one knob: how large a grid a run is
        # willing to enumerate. Past it the encoding refuses by name rather
        # than serving a truncated "exhaustive" front.
        "exhaustive_fields": {
            "grid_cap": {"type": "int", "default": DEFAULT_GRID_CAP, "min": 1, "max": 1000000, "doc": "Max grid points to enumerate (refused above this)"},
        },
        "objective_options": [
            {"id": o.name, "label": _objective_label(o.name), "goal": o.goal,
             "requires_training": o.name == ACCURACY_OBJECTIVE_NAME,
             "requires_physics": OBJECTIVES.requires_physics(o.name),
             "requires_activity": OBJECTIVES.requires_activity(o.name)}
            for o in ALL_OBJECTIVES
        ],
        "objective_catalog": get_wizard_objective_catalog(),
    }


def get_wizard_objective_catalog() -> List[Dict[str, Any]]:
    """Every registered objective with its per-search-mode availability, honestly.

    Availability comes from the objectives registry (an axis is available where
    its backing datum exists), so an axis the frontend must not offer says so
    itself, with the requirement it is missing.
    """
    rows: List[Dict[str, Any]] = []
    for spec in OBJECTIVES.all():
        modes = OBJECTIVES.modes_available(spec.key)
        rows.append({
            "id": spec.key,
            "label": _objective_label(spec.key),
            "goal": spec.goal,
            "provenance": spec.provenance,
            "available_in_modes": list(modes),
            "requires_physics": OBJECTIVES.requires_physics(spec.key),
            "requires_activity": OBJECTIVES.requires_activity(spec.key),
            "unavailable_reason": (
                "" if len(modes) == len(SEARCH_MODES) else f"requires {spec.requires}"
            ),
        })
    return rows


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
        "deployed_accuracy": "Deployed Accuracy",
        "mj_per_sample": "Energy per Sample (mJ)",
        "latency_steps": "Latency (timesteps)",
        "host_op_wall_s": "Host ComputeOp Wall (s)",
        "total_spikes": "Total Spikes",
        "pass_count": "Schedule Passes",
        "reprogram_passes": "Reprogramming Passes",
        "reprogramming_bytes": "Reprogramming Bytes",
        "params_reloaded": "Parameters Reloaded",
        "noc_inter_tile_packets": "NoC Inter-Tile Packets",
        "noc_total_packets": "NoC Total Packets",
        "programming_energy_mj": "Programming Energy (mJ)",
        "sync_barrier_energy_mj": "Sync Barrier Energy (mJ)",
        # [C5] The vendor-priced axes: named for a chip designer, not a schema.
        "chip_area_mm2": "Chip Area (mm²)",
        "energy_per_inference_mj": "Energy per Inference (mJ)",
        "e2e_latency_s": "E2E Latency (s)",
        "throughput_inferences_s": "Throughput (inf/s)",
        # [N3] The traffic axis: measured on a sealed record, modeled on a
        # candidate (wire census x declared activity on the resolved floorplan).
        "noc_total_hops": "NoC Total Hops",
        # [H3] the chip-sizing axis; [H2] the pass-buffer metrics, searchable.
        "chip_occupancy_pct": "Chip Occupancy (%)",
        "carry_peak_live_bytes": "Peak Pass-Buffer Bytes",
        "carried_raster_bytes": "Carried Raster Bytes",
        "throughput_samples_per_s": "Throughput (samples/s)",
    }
    return labels.get(name, name)


def get_pipeline_step_names_for_config(config: dict) -> List[str]:
    """Return ordered pipeline step names for the given config.

    Delegates to the single source of truth in ``deployment_pipeline``.
    """
    return [name for name, _ in get_pipeline_step_specs(config)]
