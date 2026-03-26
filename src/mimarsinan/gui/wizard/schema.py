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


def get_wizard_model_types() -> List[Dict[str, Any]]:
    """Return model type id, label, description, and config_schema from each builder class."""
    return get_all_model_type_schemas()


def get_wizard_nas_schema() -> Dict[str, Any]:
    """Return NAS optimizer options and field schemas for arch_search."""
    return {
        "optimizer_options": [
            {"id": "nsga2", "label": "NSGA-II"},
            {"id": "kedi", "label": "Kedi (LLM-based)"},
        ],
        "common_fields": {
            "pop_size": {"type": "int", "default": 8, "min": 2, "max": 64, "doc": "Population size"},
            "generations": {"type": "int", "default": 15, "min": 1, "max": 100, "doc": "Generations"},
            "seed": {"type": "int", "default": 42, "min": 0, "max": 999999, "doc": "Random seed"},
            "warmup_fraction": {"type": "float", "default": 0.1, "min": 0, "max": 1, "doc": "Warmup fraction"},
            "training_batch_size": {"type": "int", "default": 1024, "min": 1, "max": 8192, "doc": "Batch size"},
            "accuracy_evaluator": {"type": "str", "default": "extrapolating", "options": ["extrapolating", "direct"], "doc": "Evaluator"},
            "extrapolation_num_train_epochs": {"type": "int", "default": 1, "min": 1, "max": 10, "doc": "Extrapolation train epochs"},
            "extrapolation_num_checkpoints": {"type": "int", "default": 5, "min": 1, "max": 20, "doc": "Extrapolation checkpoints"},
            "extrapolation_target_epochs": {"type": "int", "default": 10, "min": 1, "max": 100, "doc": "Extrapolation target epochs"},
        },
        "kedi_fields": {
            "kedi_model": {"type": "str", "default": "deepseek:deepseek-chat", "doc": "Kedi model"},
            "kedi_adapter": {"type": "str", "default": "pydantic", "doc": "Adapter"},
            "candidates_per_batch": {"type": "int", "default": 5, "min": 1, "max": 20, "doc": "Candidates per batch"},
            "max_regen_rounds": {"type": "int", "default": 10, "min": 1, "max": 50, "doc": "Max regen rounds"},
            "max_failed_examples": {"type": "int", "default": 5, "min": 0, "max": 20, "doc": "Max failed examples"},
            "constraints_description": {"type": "textarea", "default": "", "doc": "Constraints for LLM"},
        },
        "search_space_mlp_mixer": {
            "patch_rows_options": {"type": "array_int", "default": [1, 2, 4, 7, 14, 28], "doc": "Patch row options (e.g. divisors of 28)"},
            "patch_cols_options": {"type": "array_int", "default": [1, 2, 4, 7, 14, 28], "doc": "Patch col options"},
            "patch_channels_options": {"type": "array_int", "default": [16, 32, 48, 64, 96, 128], "doc": "Patch channel options"},
            "fc_w1_options": {"type": "array_int", "default": [32, 64, 96, 128], "doc": "FC w1 options"},
            "fc_w2_options": {"type": "array_int", "default": [32, 64, 96, 128], "doc": "FC w2 options"},
        },
        "deployment_nas_fields": {
            "nas_cycles": {"type": "int", "default": 5, "min": 1, "max": 50, "doc": "NAS cycles"},
            "nas_batch_size": {"type": "int", "default": 5, "min": 1, "max": 32, "doc": "NAS batch size"},
            "nas_workers": {"type": "int", "default": 1, "min": 0, "max": 16, "doc": "NAS workers"},
        },
    }


def get_pipeline_step_names_for_state(state: Dict[str, Any]) -> List[str]:
    """
    Return the ordered list of pipeline step names for the given wizard state.

    Mirrors DeploymentPipeline._assemble_steps() without instantiating the pipeline.
    Used for start_step / stop_step dropdowns.
    """
    dp = state.get("deployment_parameters") or {}
    config_mode = dp.get("configuration_mode", "user")
    model_type = dp.get("model_type", "")
    weight_source = dp.get("weight_source") or ""
    pruning = bool(dp.get("pruning", False))
    pruning_fraction = float(dp.get("pruning_fraction", 0) or 0)
    act_q = bool(dp.get("activation_quantization", False))
    wt_q = bool(dp.get("weight_quantization", False))
    spiking = dp.get("spiking_mode", "rate")

    steps: List[str] = []

    if config_mode == "nas":
        steps.append("Architecture Search")
    else:
        steps.append("Model Configuration")

    steps.append("Model Building")

    if weight_source:
        steps.append("Weight Preloading")
    else:
        steps.append("Pretraining")

    if ModelRegistry.get_category(model_type) == "torch":
        steps.append("Torch Mapping")

    if pruning and pruning_fraction > 0:
        steps.append("Pruning Adaptation")

    # Activation Analysis and Activation Adaptation always run (in that order).
    steps.append("Activation Analysis")
    steps.append("Activation Adaptation")
    if act_q or spiking in ("ttfs", "ttfs_quantized"):
        steps.append("Clamp Adaptation")
    if act_q:
        steps.extend([
            "Activation Shifting",
            "Activation Quantization",
        ])

    if wt_q:
        steps.extend([
            "Weight Quantization",
            "Quantization Verification",
        ])

    steps.append("Normalization Fusion")
    steps.append("Soft Core Mapping")
    if wt_q:
        steps.append("Core Quantization Verification")

    if spiking == "rate":
        steps.append("CoreFlow Tuning")

    steps.append("Hard Core Mapping")
    steps.append("Simulation")

    return steps


def get_data_provider_descriptions() -> Dict[str, str]:
    """Short description per data provider for the wizard."""
    return {
        "MNIST_DataProvider": "MNIST 28×28 grayscale, 10 classes",
        "CIFAR10_DataProvider": "CIFAR-10 32×32 RGB, 10 classes",
        "CIFAR100_DataProvider": "CIFAR-100 32×32 RGB, 100 classes",
        "ECG_DataProvider": "ECG dataset",
        "MNIST32_DataProvider": "MNIST resized to 32×32",
        "ImageNet_DataProvider": "ImageNet ILSVRC2012 224×224 RGB, 1000 classes",
    }
