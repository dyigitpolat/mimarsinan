"""Deployment pipeline step ordering and semantic grouping."""

from __future__ import annotations

import sys

import torch

from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
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


_ACTIVATION_ANALYSIS_STEP: tuple[str, type] = ("Activation Analysis", ActivationAnalysisStep)
_CLAMP_ADAPTATION_STEP: tuple[str, type] = ("Clamp Adaptation", ClampAdaptationStep)
_ACTIVATION_ADAPTATION_NO_QUANT_STEP: tuple[str, type] = (
    "Activation Adaptation",
    ActivationAdaptationStep,
)

_ACTIVATION_QUANTIZATION_STEPS: list[tuple[str, type]] = [
    ("Activation Shifting",       ActivationShiftStep),
    ("Activation Quantization",   ActivationQuantizationStep),
]

_PRUNING_STEPS: list[tuple[str, type]] = [
    ("Pruning Adaptation",        PruningAdaptationStep),
]

_WEIGHT_QUANTIZATION_STEPS: list[tuple[str, type]] = [
    ("Weight Quantization",       WeightQuantizationStep),
    ("Quantization Verification", QuantizationVerificationStep),
]

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
    """Return ordered (step_name, step_class) list for the given config."""
    search_mode = derive_search_mode(config)
    spiking = config.get("spiking_mode", "lif")
    act_q = config.get("activation_quantization", False)
    wt_q = config.get("weight_quantization", False)
    pruning = config.get("pruning", False)
    pruning_fraction = float(config.get("pruning_fraction", 0.0))
    weight_source = config.get("weight_source")
    model_type = config.get("model_type", "")
    loihi_sim = bool(config.get("enable_loihi_simulation", False))
    sanafe_sim = bool(config.get("enable_sanafe_simulation", False))
    nevresim_sim = config.get("enable_nevresim_simulation", True)

    specs: list[tuple[str, type]] = []

    if search_mode != "fixed":
        specs.append(("Architecture Search", ArchitectureSearchStep))
    else:
        specs.append(("Model Configuration", ModelConfigurationStep))

    specs.append(("Model Building", ModelBuildingStep))

    if weight_source:
        specs.append(("Weight Preloading", WeightPreloadingStep))
    else:
        specs.append(("Pretraining", PretrainingStep))

    if ModelRegistry.get_category(model_type) == "torch":
        specs.append(("Torch Mapping", TorchMappingStep))

    if pruning and pruning_fraction > 0:
        specs.extend(_PRUNING_STEPS)

    specs.append(_ACTIVATION_ANALYSIS_STEP)

    if spiking == "lif":
        specs.append(("LIF Adaptation", LIFAdaptationStep))
        if bool(config.get("enable_training_noise", False)):
            specs.append(("Noise Adaptation", NoiseAdaptationStep))
    else:
        specs.append(_ACTIVATION_ADAPTATION_NO_QUANT_STEP)
        if act_q or requires_ttfs_firing(spiking):
            specs.append(_CLAMP_ADAPTATION_STEP)
        if act_q:
            specs.extend(_ACTIVATION_QUANTIZATION_STEPS)

    if wt_q:
        specs.extend(_WEIGHT_QUANTIZATION_STEPS)

    specs.append(("Normalization Fusion", NormalizationFusionStep))
    specs.append(("Soft Core Mapping", SoftCoreMappingStep))
    if wt_q:
        specs.append(
            ("Core Quantization Verification", CoreQuantizationVerificationStep)
        )

    specs.append(("Hard Core Mapping", HardCoreMappingStep))
    if nevresim_sim:
        specs.append(("Simulation", SimulationStep))

    if loihi_sim:
        specs.append(("Loihi Simulation", LoihiSimulationStep))

    if sanafe_sim:
        specs.append(("SANA-FE Simulation", SanafeSimulationStep))

    if loihi_sim and requires_ttfs_firing(spiking):
        raise ValueError(
            f"enable_loihi_simulation is not supported for spiking_mode={spiking!r}; "
            "Loihi/Lava only implements LIF dynamics."
        )

    return specs


def validate_deployment_config(config: dict, *, model_name: str, cuda_debug: bool) -> None:
    """Non-fatal sanity checks; warnings go to stderr."""
    act_q = bool(config.get("activation_quantization", False))
    spiking = config.get("spiking_mode", "lif")
    clamp_in_play = (spiking != "lif") and (act_q or spiking in ("ttfs", "ttfs_quantized"))

    if "vit" in model_name.lower() and clamp_in_play and not cuda_debug:
        print(
            "[DeploymentPipeline] ViT + Clamp Adaptation detected without "
            "cuda_debug. If you hit a CUDA device-side assert, re-run with "
            "--debug (or set deployment_parameters.cuda_debug=true) to get "
            "a precise traceback.",
            file=sys.stderr,
        )
