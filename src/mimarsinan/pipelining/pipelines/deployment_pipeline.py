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

Quantization flags  (booleans in ``deployment_parameters``)
  ``activation_quantization``
      Enables the activation quantization chain: Activation Analysis →
      Clamp Adaptation → Input Activation Analysis → Activation Shifting
      → Activation Quantization.  Configured via ``target_tq``.
      Currently supported for rate-coded only.
  ``weight_quantization``
      Enables weight quantization + verification.
      Configured via ``weight_bits``.

These flags can be set explicitly in the JSON to override presets.
"""

from __future__ import annotations

from mimarsinan.pipelining.pipeline import Pipeline
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory
from mimarsinan.pipelining.pipeline_steps import *

import numpy as np
import torch


# ── Step groups ─────────────────────────────────────────────────────────────

_ACTIVATION_QUANTIZATION_STEPS: list[tuple[str, type]] = [
    ("Activation Analysis",       ActivationAnalysisStep),
    ("Clamp Adaptation",          ClampAdaptationStep),
    ("Input Activation Analysis", InputActivationAnalysisStep),
    ("Activation Shifting",       ActivationShiftStep),
    ("Activation Quantization",   ActivationQuantizationStep),
]

_WEIGHT_QUANTIZATION_STEPS: list[tuple[str, type]] = [
    ("Weight Quantization",       WeightQuantizationStep),
    ("Quantization Verification", QuantizationVerificationStep),
]

# ── Pipeline-mode presets ───────────────────────────────────────────────────

PIPELINE_MODE_PRESETS: dict[str, dict] = {
    "vanilla": {},
    "phased": {
        "activation_quantization": True,
        "weight_quantization": True,
    },
}


class DeploymentPipeline(Pipeline):
    """Unified deployment pipeline with configurable quantization."""

    default_deployment_parameters: dict = {
        "lr": 0.001,
        "training_epochs": 10,
        "tuner_epochs": 3,
        "degradation_tolerance": 0.95,
        "configuration_mode": "user",
        "spiking_mode": "rate",
    }

    default_platform_constraints: dict = {
        "max_axons": 256,
        "max_neurons": 256,
        "target_tq": 32,
        "simulation_steps": 32,
        "weight_bits": 8,
        "allow_axon_tiling": False,
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
        if self.config.get("spiking_mode") == "ttfs":
            self.config.setdefault("firing_mode", "TTFS")
            self.config.setdefault("spike_generation_mode", "TTFS")
            self.config.setdefault("thresholding_mode", "<=")
        else:
            self.config.setdefault("firing_mode", "Default")
            self.config.setdefault("spike_generation_mode", "Deterministic")
            self.config.setdefault("thresholding_mode", "<")

    def _display_config(self):
        spiking = self.config.get("spiking_mode", "rate")
        config_mode = self.config.get("configuration_mode", "user")
        act_q = self.config.get("activation_quantization", False)
        wt_q = self.config.get("weight_quantization", False)
        print(
            f"Deployment pipeline  "
            f"[config={config_mode}, spiking={spiking}, "
            f"act_quant={act_q}, wt_quant={wt_q}]"
        )
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    # --------------------------------------------------- step assembly logic

    def _assemble_steps(self):
        config_mode = self.config.get("configuration_mode", "user")
        spiking = self.config.get("spiking_mode", "rate")
        act_q = self.config.get("activation_quantization", False)
        wt_q = self.config.get("weight_quantization", False)

        # ── Configuration ───────────────────────────────────────────────
        if config_mode == "nas":
            self.add_pipeline_step(
                "Architecture Search", ArchitectureSearchStep(self)
            )
        else:
            self.add_pipeline_step(
                "Model Configuration", ModelConfigurationStep(self)
            )

        # ── Model Building ──────────────────────────────────────────────
        self.add_pipeline_step("Model Building", ModelBuildingStep(self))

        # ── Pretraining ─────────────────────────────────────────────────
        self.add_pipeline_step("Pretraining", PretrainingStep(self))

        # ── Activation Quantization ─────────────────────────────────────
        if act_q:
            for name, cls in _ACTIVATION_QUANTIZATION_STEPS:
                self.add_pipeline_step(name, cls(self))

        # ── Weight Quantization ─────────────────────────────────────────
        if wt_q:
            for name, cls in _WEIGHT_QUANTIZATION_STEPS:
                self.add_pipeline_step(name, cls(self))

        # ── Normalization Fusion & IR Mapping ───────────────────────────
        self.add_pipeline_step(
            "Normalization Fusion", NormalizationFusionStep(self)
        )
        self.add_pipeline_step("Soft Core Mapping", SoftCoreMappingStep(self))
        self.add_pipeline_step(
            "Core Quantization Verification",
            CoreQuantizationVerificationStep(self),
        )

        # ── CoreFlow Tuning (rate-coded only) ───────────────────────────
        if spiking == "rate":
            self.add_pipeline_step(
                "CoreFlow Tuning", CoreFlowTuningStep(self)
            )

        # ── Hardware Deployment ─────────────────────────────────────────
        self.add_pipeline_step("Hard Core Mapping", HardCoreMappingStep(self))
        self.add_pipeline_step("Simulation", SimulationStep(self))

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
