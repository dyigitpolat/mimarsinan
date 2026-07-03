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


def _backend_tail(plan: DeploymentPlan) -> list[StepSpec]:
    return [
        StepSpec(name, cls, group="simulation")
        for name, cls in BACKEND_REGISTRY.selected_step_specs(plan)
    ]


_STEP_PLAN = StepPlan([
    StepSpec("Architecture Search",            ArchitectureSearchStep,           group="configuration"),
    StepSpec("Model Configuration",            ModelConfigurationStep,           group="configuration"),
    StepSpec("Model Building",                 ModelBuildingStep,                group="model_building"),
    StepSpec("Weight Preloading",              WeightPreloadingStep,             group="pretraining"),
    StepSpec("Pretraining",                    PretrainingStep,                  group="pretraining"),
    StepSpec("Torch Mapping",                  TorchMappingStep,                 group="torch_mapping"),
    StepSpec("Pruning Adaptation",             PruningAdaptationStep,            group="pruning"),
    StepSpec("Activation Analysis",            ActivationAnalysisStep,           group="activation"),
    StepSpec("Activation Adaptation",          ActivationAdaptationStep,         group="activation"),
    StepSpec("Clamp Adaptation",               ClampAdaptationStep,              group="activation"),
    StepSpec("Activation Shifting",            ActivationShiftStep,              group="activation_quantization"),
    StepSpec("Activation Quantization",        ActivationQuantizationStep,       group="activation_quantization"),
    StepSpec("LIF Adaptation",                 LIFAdaptationStep,                group="activation"),
    StepSpec("TTFS Cycle Fine-Tuning",         TTFSCycleAdaptationStep,          group="activation"),
    StepSpec("Noise Adaptation",               NoiseAdaptationStep,              group="activation"),
    StepSpec("Weight Quantization",            WeightQuantizationStep,           group="weight_quantization"),
    StepSpec("Quantization Verification",      QuantizationVerificationStep,     group="weight_quantization"),
    StepSpec("Normalization Fusion",           NormalizationFusionStep,          group="normalization"),
    StepSpec("Soft Core Mapping",              SoftCoreMappingStep,              group="soft_mapping"),
    StepSpec("Core Quantization Verification", CoreQuantizationVerificationStep, group="core_verification"),
    StepSpec("Hard Core Mapping",              HardCoreMappingStep,              group="hardware"),
    _backend_tail,
])


def get_pipeline_semantic_group_by_step_name(config: dict) -> dict[str, str]:
    """Return {step_name: semantic_group_id} for every step in the given config."""
    plan = DeploymentPlan.resolve(config)
    return {spec.name: spec.group for spec in _STEP_PLAN.resolve(plan)}


def get_pipeline_step_specs(config: dict) -> list[tuple[str, type]]:
    """Return the ordered ``(name, class)`` list for the config.

    The ``_STEP_PLAN`` registry filters itself by each step's ``applies_to``; the
    requires/promises DAG is validated up-front, failing loud with the missing producer named.
    """
    plan = DeploymentPlan.resolve(config)
    return [spec.to_pair() for spec in _STEP_PLAN.validate_data_contract(plan)]


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
