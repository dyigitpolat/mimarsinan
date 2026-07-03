"""Deployment pipeline step ordering and semantic grouping."""

from __future__ import annotations

import sys

import torch

from mimarsinan.chip_simulation.backend import BACKEND_REGISTRY
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.step_plan import StepPlan, StepSpec
from mimarsinan.pipelining.pipeline_steps import *


def select_device() -> torch.device:
    """Pick the CUDA device with the most free memory, or CPU if none available."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if n <= 1:
        return torch.device("cuda:0")
    best_idx, best_free = 0, -1
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_idx = i
    device = torch.device(f"cuda:{best_idx}")
    print(f"[DeviceSelect] Using {device} ({best_free / (1 << 30):.1f} GiB free out of {n} GPUs)")
    return device


_STEP_PLAN = StepPlan([
    StepSpec("Architecture Search",            ArchitectureSearchStep),
    StepSpec("Model Configuration",            ModelConfigurationStep),
    StepSpec("Model Building",                 ModelBuildingStep),
    StepSpec("Weight Preloading",              WeightPreloadingStep),
    StepSpec("Pretraining",                    PretrainingStep),
    StepSpec("Torch Mapping",                  TorchMappingStep),
    StepSpec("Pruning Adaptation",             PruningAdaptationStep),
    StepSpec("Activation Analysis",            ActivationAnalysisStep),
    StepSpec("Activation Adaptation",          ActivationAdaptationStep),
    StepSpec("Clamp Adaptation",               ClampAdaptationStep),
    StepSpec("Activation Shifting",            ActivationShiftStep),
    StepSpec("Activation Quantization",        ActivationQuantizationStep),
    StepSpec("LIF Adaptation",                 LIFAdaptationStep),
    StepSpec("TTFS Cycle Fine-Tuning",         TTFSCycleAdaptationStep),
    StepSpec("Noise Adaptation",               NoiseAdaptationStep),
    StepSpec("Weight Quantization",            WeightQuantizationStep),
    StepSpec("Quantization Verification",      QuantizationVerificationStep),
    StepSpec("Normalization Fusion",           NormalizationFusionStep),
    StepSpec("Soft Core Mapping",              SoftCoreMappingStep),
    StepSpec("Core Quantization Verification", CoreQuantizationVerificationStep),
    StepSpec("Hard Core Mapping",              HardCoreMappingStep),
    lambda plan: BACKEND_REGISTRY.selected_step_specs(plan),
])


_SEMANTIC_GROUP_BY_STEP_CLASS: dict[type, str] = {
    ArchitectureSearchStep:             "configuration",
    ModelConfigurationStep:             "configuration",
    ModelBuildingStep:                  "model_building",
    PretrainingStep:                    "pretraining",
    WeightPreloadingStep:               "pretraining",
    TorchMappingStep:                   "torch_mapping",
    PruningAdaptationStep:              "pruning",
    ActivationAnalysisStep:             "activation",
    ActivationAdaptationStep:           "activation",
    ClampAdaptationStep:                "activation",
    LIFAdaptationStep:                  "activation",
    TTFSCycleAdaptationStep:            "activation",
    NoiseAdaptationStep:                "activation",
    ActivationShiftStep:                "activation_quantization",
    ActivationQuantizationStep:         "activation_quantization",
    WeightQuantizationStep:             "weight_quantization",
    QuantizationVerificationStep:       "weight_quantization",
    NormalizationFusionStep:            "normalization",
    SoftCoreMappingStep:                "soft_mapping",
    CoreQuantizationVerificationStep:   "core_verification",
    HardCoreMappingStep:                "hardware",
    SimulationStep:                     "simulation",
    LoihiSimulationStep:                "simulation",
    SanafeSimulationStep:               "simulation",
}


def get_pipeline_semantic_group_by_step_name(config: dict) -> dict[str, str]:
    """Return {step_name: semantic_group_id} for every step in the given config."""
    return {
        name: _SEMANTIC_GROUP_BY_STEP_CLASS.get(cls, "other")
        for name, cls in get_pipeline_step_specs(config)
    }


def get_pipeline_step_specs(config: dict) -> list[tuple[str, type]]:
    """Return the ordered ``(name, class)`` list for the config.

    The ``_STEP_PLAN`` registry filters itself by each step's ``applies_to``; the
    requires/promises DAG is validated up-front, failing loud with the missing producer named.
    """
    plan = DeploymentPlan.resolve(config)
    return _STEP_PLAN.validate_data_contract(plan)


def validate_deployment_config(config: dict, *, model_name: str, cuda_debug: bool) -> None:
    """Non-fatal sanity checks; warnings go to stderr."""
    plan = DeploymentPlan.resolve(config)
    clamp_in_play = plan.requires_clamp_preconditioning

    if "vit" in model_name.lower() and clamp_in_play and not cuda_debug:
        print(
            "[DeploymentPipeline] ViT + Clamp Adaptation detected without "
            "cuda_debug. If you hit a CUDA device-side assert, re-run with "
            "--debug (or set deployment_parameters.cuda_debug=true) to get "
            "a precise traceback.",
            file=sys.stderr,
        )
