"""
DeploymentPipeline — unified SNN deployment pipeline.

Configuration
=============

Pipeline mode  (``pipeline_mode`` in JSON — selects a preset)
  ``"vanilla"``  Pretraining + direct mapping.
  ``"phased"``   Pretraining + activation quantization + weight quantization.

Spiking mode  (``spiking_mode`` in ``deployment_parameters``)
  ``"rate"``     Rate-coded SNN.  CoreFlow tuning step is included.
  ``"ttfs"``     Time-to-first-spike SNN.  Analytical ReLU↔TTFS mapping.

Activation adaptation
  Activation Analysis always runs. Activation Adaptation always runs next:
  it replaces non-ReLU chip-targeted perceptrons (e.g. GELU, LeakyReLU) with
  ReLU when needed and applies activation_scales in all cases. When
  ``activation_quantization`` is True OR ``spiking_mode`` is ``"ttfs"`` or
  ``"ttfs_quantized"``, Clamp Adaptation then runs: trains with a
  ClampDecorator so the model operates within the TTFS saturation range
  (relu(V)/θ clamped to 1.0 in hardware).

Quantization flags  (booleans in ``deployment_parameters``)
  ``activation_quantization``
      Enables additional activation quantization: Activation Shifting →
      Activation Quantization (discrete levels for chip deployment).
      Configured via ``target_tq``.
  ``weight_quantization``
      Enables weight quantization + verification.
      Configured via ``weight_bits``.

These flags can be set explicitly in the JSON to override presets.
"""

from __future__ import annotations

from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.pipelining.model_registry import ModelRegistry
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory
from mimarsinan.pipelining.pipeline_steps import *

import numpy as np
import torch


from mimarsinan.pipelining.search_mode import derive_search_mode  # noqa: F401 — re-export

# ── Step groups ─────────────────────────────────────────────────────────────

# Activation Analysis always runs (produces activation_scales for IR).
# Activation Adaptation always runs next (ReLU replacement when needed, scales always).
# Clamp Adaptation runs only when act_q or TTFS, after Activation Adaptation.
_ACTIVATION_ANALYSIS_STEP: tuple[str, type] = ("Activation Analysis", ActivationAnalysisStep)
_CLAMP_ADAPTATION_STEP: tuple[str, type] = ("Clamp Adaptation", ClampAdaptationStep)
_ACTIVATION_ADAPTATION_NO_QUANT_STEP: tuple[str, type] = (
    "Activation Adaptation",
    ActivationAdaptationStep,
)

# Full activation quantization: Shifting + Quantization are only needed
# when discrete activation levels are required for chip deployment.
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

# ── Semantic groups (step class → UI group id) ──────────────────────────────
# Stable lowercase ids used by the GUI to color-code pipeline bars.
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
    ActivationShiftStep:                "activation_quantization",
    ActivationQuantizationStep:         "activation_quantization",
    WeightQuantizationStep:             "weight_quantization",
    QuantizationVerificationStep:       "weight_quantization",
    NormalizationFusionStep:            "normalization",
    SoftCoreMappingStep:                "soft_mapping",
    CoreQuantizationVerificationStep:   "core_verification",
    CoreFlowTuningStep:                 "coreflow_tuning",
    HardCoreMappingStep:                "hardware",
    SimulationStep:                     "simulation",
}


def get_pipeline_semantic_group_by_step_name(config: dict) -> dict[str, str]:
    """Return {step_name: semantic_group_id} for every step in the given config.

    Derived from ``get_pipeline_step_specs``, so the mapping is always
    consistent with actual pipeline composition.
    """
    return {
        name: _SEMANTIC_GROUP_BY_STEP_CLASS.get(cls, "other")
        for name, cls in get_pipeline_step_specs(config)
    }


# ── Pipeline-mode presets ───────────────────────────────────────────────────

PIPELINE_MODE_PRESETS: dict[str, dict] = {
    "vanilla": {},
    "phased": {
        "activation_quantization": True,
        "weight_quantization": True,
    },
}


def get_pipeline_step_specs(config: dict) -> list[tuple[str, type]]:
    """Return ordered list of (step_name, step_class) for the given config.

    Single source of truth for pipeline step order and presence. Uses only
    config keys; no pipeline or data provider required. Used by
    DeploymentPipeline._assemble_steps() and by the wizard API for preview.
    """
    search_mode = derive_search_mode(config)
    spiking = config.get("spiking_mode", "rate")
    act_q = config.get("activation_quantization", False)
    wt_q = config.get("weight_quantization", False)
    pruning = config.get("pruning", False)
    pruning_fraction = float(config.get("pruning_fraction", 0.0))
    weight_source = config.get("weight_source")
    model_type = config.get("model_type", "")

    specs: list[tuple[str, type]] = []

    # ── Configuration ───────────────────────────────────────────────
    if search_mode != "fixed":
        specs.append(("Architecture Search", ArchitectureSearchStep))
    else:
        specs.append(("Model Configuration", ModelConfigurationStep))

    # ── Model Building ──────────────────────────────────────────────
    specs.append(("Model Building", ModelBuildingStep))

    # ── Pretraining / Weight Preloading ────────────────────────────
    if weight_source:
        specs.append(("Weight Preloading", WeightPreloadingStep))
    else:
        specs.append(("Pretraining", PretrainingStep))

    # ── Torch Mapping (category "torch" models) ─────────────────────
    if ModelRegistry.get_category(model_type) == "torch":
        specs.append(("Torch Mapping", TorchMappingStep))

    # ── Pruning ─────────────────────────────────────────────────────
    if pruning and pruning_fraction > 0:
        specs.extend(_PRUNING_STEPS)

    # ── Activation Adaptation + Quantization ──────────────────────
    # Activation Analysis always runs (activation_scales for IR).
    # Activation Adaptation always runs next: ReLU replacement when any
    # chip-targeted perceptron is non-ReLU, scales applied in all cases.
    # When act_q or TTFS, Clamp Adaptation runs after (clamp for quant/TTFS).
    specs.append(_ACTIVATION_ANALYSIS_STEP)
    specs.append(_ACTIVATION_ADAPTATION_NO_QUANT_STEP)
    if act_q or spiking in ("ttfs", "ttfs_quantized"):
        specs.append(_CLAMP_ADAPTATION_STEP)

    # Full activation quantization (Shifting + Quantization) is only
    # needed when discrete activation levels are required.
    if act_q:
        specs.extend(_ACTIVATION_QUANTIZATION_STEPS)

    # ── Weight Quantization ─────────────────────────────────────────
    if wt_q:
        specs.extend(_WEIGHT_QUANTIZATION_STEPS)

    # ── Normalization Fusion & IR Mapping ───────────────────────────
    specs.append(("Normalization Fusion", NormalizationFusionStep))
    specs.append(("Soft Core Mapping", SoftCoreMappingStep))
    if wt_q:
        specs.append(
            ("Core Quantization Verification", CoreQuantizationVerificationStep)
        )

    # ── CoreFlow Tuning (rate-coded only) ───────────────────────────
    if spiking == "rate":
        specs.append(("CoreFlow Tuning", CoreFlowTuningStep))

    # ── Hardware Deployment ─────────────────────────────────────────
    specs.append(("Hard Core Mapping", HardCoreMappingStep))
    specs.append(("Simulation", SimulationStep))

    return specs


class DeploymentPipeline(Pipeline):
    """Unified deployment pipeline with configurable quantization."""

    default_deployment_parameters: dict = {
        "lr": 0.001,
        "lr_range_min": 1e-5,
        "lr_range_max": 1e-1,
        "training_epochs": 10,
        "tuning_budget_scale": 1.0,
        "tuner_calibrate_smooth_tolerance": True,
        "tuner_smooth_tolerance_residual_threshold": 1e-3,
        "tuner_smooth_tolerance_min": 0.01,
        "tuner_smooth_tolerance_max": 0.15,
        "tuner_smooth_tolerance_baseline_epsilon": 1e-9,
        "tuner_smooth_tolerance_lr_scale": 1.0,
        "degradation_tolerance": 0.95,
        "model_config_mode": "user",
        "hw_config_mode": "fixed",
        "spiking_mode": "rate",
        "allow_scheduling": False,
    }

    default_platform_constraints: dict = {
        "cores": [{"max_axons": 256, "max_neurons": 256, "count": 1000}],
        "target_tq": 32,
        "simulation_steps": 32,
        "weight_bits": 8,
    }

    # ------------------------------------------------------------------ init

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

    # -------------------------------------------------------- config helpers

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
        self.config["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.tolerance = self.config["degradation_tolerance"]

        # Spiking-mode defaults.
        # The user only needs to set ``spiking_mode``; firing_mode,
        # spike_generation_mode and thresholding_mode are derived automatically.
        #   "ttfs"           → analytical / continuous TTFS (Python + C++)
        #   "ttfs_quantized" → cycle-based quantised TTFS  (Python + C++)
        #   anything else    → rate-coded (Default)
        spiking = self.config.get("spiking_mode", "rate")
        if spiking in ("ttfs", "ttfs_quantized"):
            self.config.setdefault("firing_mode", "TTFS")
            self.config.setdefault("spike_generation_mode", "TTFS")
            self.config.setdefault("thresholding_mode", "<=")

            # Guard: TTFS spiking modes require TTFS firing / spike generation.
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
            self.config.setdefault("spike_generation_mode", "Deterministic")
            self.config.setdefault("thresholding_mode", "<")

    def _display_config(self):
        spiking = self.config.get("spiking_mode", "rate")
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

    # --------------------------------------------------- step assembly logic

    def _assemble_steps(self):
        for name, cls in get_pipeline_step_specs(self.config):
            self.add_pipeline_step(name, cls(self))
        pruning = self.config.get("pruning", False)
        pruning_fraction = float(self.config.get("pruning_fraction", 0.0))
        if pruning and pruning_fraction > 0:
            print(f"[DeploymentPipeline] Pruning enabled: pruning={pruning}, pruning_fraction={pruning_fraction}; PruningAdaptationStep added.")
        else:
            print(f"[DeploymentPipeline] Pruning not in pipeline: pruning={pruning}, pruning_fraction={pruning_fraction}")

    # -------------------------------------------------------- preset helper

    @staticmethod
    def apply_preset(pipeline_mode: str, deployment_parameters: dict) -> None:
        """Merge a ``pipeline_mode`` preset into *deployment_parameters*.

        Preset values are applied with ``setdefault`` — explicit user
        settings always win.
        """
        preset = PIPELINE_MODE_PRESETS.get(pipeline_mode, {})
        for key, value in preset.items():
            deployment_parameters.setdefault(key, value)
