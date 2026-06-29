"""Unified SNN deployment pipeline with configurable quantization."""

from __future__ import annotations

import os
import sys

from mimarsinan.pipelining.core.engine.pipeline import Pipeline
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory
from mimarsinan.pipelining.core.pipelines.deployment_specs import (
    get_pipeline_semantic_group_by_step_name,
    get_pipeline_step_specs,
    select_device,
    validate_deployment_config,
)

import numpy as np

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
    apply_preset as _apply_preset,
    get_default_deployment_parameters,
    get_default_platform_constraints,
)
from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
    derive_pipeline_runtime_parameters,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.search_mode import derive_search_mode  # noqa: F401 — re-export


class DeploymentPipeline(Pipeline):
    """Unified deployment pipeline with configurable quantization."""

    default_deployment_parameters = DEFAULT_DEPLOYMENT_PARAMETERS
    default_platform_constraints = DEFAULT_PLATFORM_CONSTRAINTS

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
        self.config.update(get_default_deployment_parameters())
        self.config.update(deployment_parameters)

        self.config.update(get_default_platform_constraints())
        self.config.update(platform_constraints)
        derive_deployment_parameters(self.config)
        derive_pipeline_runtime_parameters(self.config)

        if data_provider is None:
            data_provider = self.data_provider_factory.create()
        self.config["input_shape"] = data_provider.get_input_shape()
        self.config["input_size"] = int(np.prod(self.config["input_shape"]))
        self.config["num_classes"] = data_provider.get_prediction_mode().num_classes
        self.config["device"] = select_device()

        plan = DeploymentPlan.resolve(self.config)
        spiking = plan.spiking_mode
        if plan.requires_ttfs_firing:
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
            self.config.setdefault("cycle_accurate_lif_forward", True)

        self.tolerance = 1.0 - plan.degradation_tolerance
        # Opt-in rung-2 budget: with NF and SCM sharing semantics, the honest
        # residual is the mapping-level wire effect, so ~0.02 is a sane value.
        if plan.scm_degradation_tolerance is not None:
            self.step_tolerances["Soft Core Mapping"] = (
                1.0 - plan.scm_degradation_tolerance
            )

        self.accuracy_budget.budget_total = plan.degradation_budget_total

        if os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1":
            self.config.setdefault("cuda_debug", True)
        # Re-resolve AFTER the env-driven setdefault so the plan observes it.
        self.cuda_debug = DeploymentPlan.resolve(self.config).cuda_debug

        self._validate_config()

    def _validate_config(self):
        model_name = self.config.get("model_name") or self.config.get("model_type", "")
        validate_deployment_config(
            self.config,
            model_name=model_name,
            cuda_debug=self.cuda_debug,
        )

    def _display_config(self):
        plan = DeploymentPlan.resolve(self.config)
        print(
            f"Deployment pipeline  "
            f"[search_mode={plan.search_mode}, spiking={plan.spiking_mode}, "
            f"act_quant={plan.activation_quantization}, "
            f"wt_quant={plan.weight_quantization}]"
        )
        for key, value in self.config.items():
            print(f"  {key}: {value}")

    def _assemble_steps(self):
        for name, cls in get_pipeline_step_specs(self.config):
            self.add_pipeline_step(name, cls(self))
        plan = DeploymentPlan.resolve(self.config)
        pruning = plan.pruning
        pruning_fraction = plan.pruning_fraction
        if plan.pruning_enabled:
            print(f"[DeploymentPipeline] Pruning enabled: pruning={pruning}, pruning_fraction={pruning_fraction}; PruningAdaptationStep added.")
        else:
            print(f"[DeploymentPipeline] Pruning not in pipeline: pruning={pruning}, pruning_fraction={pruning_fraction}")

    @staticmethod
    def apply_preset(pipeline_mode: str, deployment_parameters: dict) -> None:
        """Merge a pipeline_mode preset into deployment_parameters (setdefault)."""
        _apply_preset(pipeline_mode, deployment_parameters)
