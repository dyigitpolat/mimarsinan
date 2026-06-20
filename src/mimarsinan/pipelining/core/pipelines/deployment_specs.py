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
    # ── configuration: search vs fixed (mutually exclusive on applies_to) ──
    StepSpec("Architecture Search",            ArchitectureSearchStep),
    StepSpec("Model Configuration",            ModelConfigurationStep),
    StepSpec("Model Building",                 ModelBuildingStep),
    # ── weights: preload vs pretrain (mutually exclusive on applies_to) ──
    StepSpec("Weight Preloading",              WeightPreloadingStep),
    StepSpec("Pretraining",                    PretrainingStep),
    StepSpec("Torch Mapping",                  TorchMappingStep),
    StepSpec("Pruning Adaptation",             PruningAdaptationStep),
    StepSpec("Activation Analysis",            ActivationAnalysisStep),
    # ── activation-adaptation family: V2 policy chooses LIF-style (one
    #    replacement step) vs the analytical clamp→shift→quantize chain ──
    StepSpec("LIF Adaptation",                 LIFAdaptationStep),
    StepSpec("TTFS Cycle Fine-Tuning",         TTFSCycleAdaptationStep),
    StepSpec("Noise Adaptation",               NoiseAdaptationStep),
    StepSpec("Activation Adaptation",          ActivationAdaptationStep),
    StepSpec("Clamp Adaptation",               ClampAdaptationStep),
    StepSpec("Activation Shifting",            ActivationShiftStep),
    StepSpec("Activation Quantization",        ActivationQuantizationStep),
    # ── weight quantization ──
    StepSpec("Weight Quantization",            WeightQuantizationStep),
    StepSpec("Quantization Verification",      QuantizationVerificationStep),
    # ── mapping ──
    StepSpec("Normalization Fusion",           NormalizationFusionStep),
    StepSpec("Soft Core Mapping",              SoftCoreMappingStep),
    StepSpec("Core Quantization Verification", CoreQuantizationVerificationStep),
    StepSpec("Hard Core Mapping",              HardCoreMappingStep),
    # ── deployment backends (V3): validate every enabled backend against the
    #    capability matrix UP-FRONT, then append the applicable backend steps ──
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
    """Return ordered (step_name, step_class) list for the given config.

    Vector V5: the ordered ``_STEP_PLAN`` registry filters itself by each step's
    ``applies_to(plan)`` (the per-flag conditions now live on the steps). The
    backend tail still routes through ``BACKEND_REGISTRY`` so an enabled but
    unsupported backend×mode raises at assembly (V3), not mid-run.
    """
    plan = DeploymentPlan.resolve(config)
    return _STEP_PLAN.resolve(plan)


def validate_deployment_config(config: dict, *, model_name: str, cuda_debug: bool) -> None:
    """Non-fatal sanity checks; warnings go to stderr."""
    plan = DeploymentPlan.resolve(config)
    act_q = plan.activation_quantization
    spiking = plan.spiking_mode
    clamp_in_play = (spiking != "lif") and (act_q or spiking in ("ttfs", "ttfs_quantized"))

    if "vit" in model_name.lower() and clamp_in_play and not cuda_debug:
        print(
            "[DeploymentPipeline] ViT + Clamp Adaptation detected without "
            "cuda_debug. If you hit a CUDA device-side assert, re-run with "
            "--debug (or set deployment_parameters.cuda_debug=true) to get "
            "a precise traceback.",
            file=sys.stderr,
        )
