"""Unified SNN deployment pipeline with configurable quantization."""

from __future__ import annotations

import os
import sys

from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory
from mimarsinan.pipelining.pipeline_steps import *

import numpy as np
import torch


from mimarsinan.pipelining.search_mode import derive_search_mode  # noqa: F401 — re-export


def _select_device() -> torch.device:
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


PIPELINE_MODE_PRESETS: dict[str, dict] = {
    "vanilla": {},
    "phased": {
        "activation_quantization": True,
        "weight_quantization": True,
    },
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
    specs.append(_ACTIVATION_ADAPTATION_NO_QUANT_STEP)

    if spiking == "lif":
        specs.append(("LIF Adaptation", LIFAdaptationStep))
    else:
        if act_q or spiking in ("ttfs", "ttfs_quantized"):
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
    specs.append(("Simulation", SimulationStep))

    if loihi_sim:
        specs.append(("Loihi Simulation", LoihiSimulationStep))

    if sanafe_sim:
        specs.append(("SANA-FE Simulation", SanafeSimulationStep))

    return specs


class DeploymentPipeline(Pipeline):
    """Unified deployment pipeline with configurable quantization."""

    default_deployment_parameters: dict = {
        "lr": 0.001,
        "lr_range_min": 1e-5,
        "lr_range_max": 1e-1,
        "training_epochs": 10,
        "tuning_budget_scale": 1.0,
        "degradation_tolerance": 0.05,
        "model_config_mode": "user",
        "hw_config_mode": "fixed",
        "spiking_mode": "lif",
        "enable_loihi_simulation": False,
        "enable_sanafe_simulation": False,
        "sanafe_sample_count": 1,
        "sanafe_arch_preset": "loihi",
        "sanafe_custom_arch_path": None,
        "sanafe_log_potential_trace": False,
        "sanafe_log_message_trace": True,
        "sanafe_parity_check": True,
        "allow_scheduling": False,
        "training_recipe": {
            "optimizer": "adamw",
            "weight_decay": 0.05,
            "betas": [0.9, 0.999],
            "scheduler": "cosine",
            "warmup_ratio": 0.1,
            "grad_clip_norm": 1.0,
            "layer_wise_lr_decay": 0.75,
            "label_smoothing": 0.1,
        },
        "tuning_recipe": {
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "scheduler": "cosine",
            "warmup_ratio": 0.0,
            "grad_clip_norm": 1.0,
            "layer_wise_lr_decay": 1.0,
            "label_smoothing": 0.0,
        },
    }

    default_platform_constraints: dict = {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}],
        "target_tq": 32,
        "simulation_steps": 32,
        "weight_bits": 8,
    }


    def __init__(
        self,
        data_provider_factory: DataProviderFactory,
        deployment_parameters: dict,
        platform_constraints: dict,
        reporter,
        working_directory: str,
    ):
        super().__init__(working_directory)

        self.data_provider_factory = data_provider_factory
        self.reporter = reporter

        self.config: dict = {}
        data_provider = self.data_provider_factory.create()
        self.loss = data_provider.create_loss()
        self._initialize_config(
            deployment_parameters, platform_constraints, data_provider=data_provider
        )
        self._display_config()
        self._assemble_steps()


    def _initialize_config(
        self,
        deployment_parameters: dict,
        platform_constraints: dict,
        *,
        data_provider=None,
    ):
        self.config.update(self.default_deployment_parameters)
        self.config.update(deployment_parameters)

        self.config.update(self.default_platform_constraints)
        self.config.update(platform_constraints)

        if data_provider is None:
            data_provider = self.data_provider_factory.create()
        self.config["input_shape"] = data_provider.get_input_shape()
        self.config["input_size"] = int(np.prod(self.config["input_shape"]))
        self.config["num_classes"] = data_provider.get_prediction_mode().num_classes
        self.config["device"] = _select_device()

        spiking = self.config.get("spiking_mode", "lif")
        if spiking in ("ttfs", "ttfs_quantized"):
            self.config.setdefault("firing_mode", "TTFS")
            self.config.setdefault("spike_generation_mode", "TTFS")
            self.config.setdefault("thresholding_mode", "<=")

            if self.config["firing_mode"] != "TTFS":
                raise ValueError(
                    f"spiking_mode='{spiking}' requires firing_mode='TTFS', "
                    f"got '{self.config['firing_mode']}'"
                )
            if self.config["spike_generation_mode"] != "TTFS":
                raise ValueError(
                    f"spiking_mode='{spiking}' requires spike_generation_mode='TTFS', "
                    f"got '{self.config['spike_generation_mode']}'"
                )
        else:
            self.config.setdefault("firing_mode", "Default")
            self.config.setdefault("spike_generation_mode", "Uniform")
            self.config.setdefault("thresholding_mode", "<=")
            self.config.setdefault("cycle_accurate_lif_forward", False)

        self.tolerance = 1.0 - float(self.config.get("degradation_tolerance", 0.05))

        default_budget = 2.0 * float(self.config.get("degradation_tolerance", 0.05))
        self.accuracy_budget.budget_total = float(
            self.config.get("degradation_budget_total", default_budget)
        )

        if os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1":
            self.config.setdefault("cuda_debug", True)
        self.cuda_debug = bool(self.config.get("cuda_debug", False))

        self._validate_config()

    def _validate_config(self):
        """Non-fatal sanity checks; warnings go to stderr."""
        model_name = self.config.get("model_name") or self.config.get("model_type", "")
        act_q = bool(self.config.get("activation_quantization", False))
        spiking = self.config.get("spiking_mode", "lif")
        clamp_in_play = (spiking != "lif") and (act_q or spiking in ("ttfs", "ttfs_quantized"))

        if "vit" in model_name.lower() and clamp_in_play and not self.cuda_debug:
            print(
                "[DeploymentPipeline] ViT + Clamp Adaptation detected without "
                "cuda_debug. If you hit a CUDA device-side assert, re-run with "
                "--debug (or set deployment_parameters.cuda_debug=true) to get "
                "a precise traceback.",
                file=sys.stderr,
            )

    def _display_config(self):
        spiking = self.config.get("spiking_mode", "lif")
        search_mode = derive_search_mode(self.config)
        act_q = self.config.get("activation_quantization", False)
        wt_q = self.config.get("weight_quantization", False)
        print(
            f"Deployment pipeline  "
            f"[search_mode={search_mode}, spiking={spiking}, "
            f"act_quant={act_q}, wt_quant={wt_q}]"
        )
        for key, value in self.config.items():
            print(f"  {key}: {value}")


    def _assemble_steps(self):
        for name, cls in get_pipeline_step_specs(self.config):
            self.add_pipeline_step(name, cls(self))
        pruning = self.config.get("pruning", False)
        pruning_fraction = float(self.config.get("pruning_fraction", 0.0))
        if pruning and pruning_fraction > 0:
            print(f"[DeploymentPipeline] Pruning enabled: pruning={pruning}, pruning_fraction={pruning_fraction}; PruningAdaptationStep added.")
        else:
            print(f"[DeploymentPipeline] Pruning not in pipeline: pruning={pruning}, pruning_fraction={pruning_fraction}")


    @staticmethod
    def apply_preset(pipeline_mode: str, deployment_parameters: dict) -> None:
        """Merge a pipeline_mode preset into deployment_parameters (setdefault)."""
        preset = PIPELINE_MODE_PRESETS.get(pipeline_mode, {})
        for key, value in preset.items():
            deployment_parameters.setdefault(key, value)
