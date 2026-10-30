"""Unified SNN deployment pipeline with configurable quantization."""

from __future__ import annotations

import os

from mimarsinan.pipelining.core.engine.pipeline import Pipeline
from mimarsinan.data_handling.data_provider_factory import DataProviderFactory
from mimarsinan.pipelining.core.pipelines.deployment_specs import (
    get_pipeline_semantic_group_by_step_name as get_pipeline_semantic_group_by_step_name,
    get_pipeline_step_specs,
    select_device,
)

import numpy as np

from mimarsinan.config_schema.defaults import (
    DEFAULT_DEPLOYMENT_PARAMETERS,
    DEFAULT_PLATFORM_CONSTRAINTS,
    apply_preset as _apply_preset,
)
from mimarsinan.config_schema.registry import declaration_judge
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
from mimarsinan.config_schema.runtime import (
    DerivationRefusal, model_registration, registration_errors, registration_updates, run_config,
)
from mimarsinan.common.workload_profile import fold_workload_profiles
from mimarsinan.mapping.platform.packaging_contract import packaging_contract_for
import mimarsinan.mapping.platform.target_expansion as target_expansion
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.pipelining.cache.legacy_placement import (
    resolve_cached_flow_placements,
)
from mimarsinan.pipelining.cache.legacy_soma_law import resolve_cached_forward_soma_laws
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode as derive_search_mode  # noqa: F401 — re-export


def merge_pipeline_config(
    deployment_parameters: dict, platform_constraints: dict
) -> dict:
    """The exact ``_initialize_config`` merge + derivation, sans provider facts
    and device I/O (shared with the golden-resolution harness).

    A declared target expands first (``expand_target``), so the one judge
    (``declared_sections_errors``, G4 interface note 1) judges every declaration
    as written, the bundle's included, under the core semantics the run takes
    and the model builder's registration (``model_registration``: its weight
    sets, which the pretrained regime's legal sets read), before anything else
    reads one; then ``run_config``, the one derivation every reader takes: the
    expansion writes the two labels a flat config carries, and
    ``derive_run_config`` lays the sections as the run lays them (a null
    declaration unset, as both doors read it), reads them as their consumers
    read them, and derives (its refusal a ``DerivationRefusal``, a declared
    derived key it would replace among them)."""
    deployment_parameters, platform_constraints = target_expansion.expand_target(
        deployment_parameters, platform_constraints, write_derived=False
    )
    layout = declaration_judge.sections_run_layout(deployment_parameters, platform_constraints)
    errors = declaration_judge.declared_sections_errors(
        deployment_parameters, platform_constraints,
        registration=registration_updates(model_registration(layout), None))
    if errors:
        raise DeclarationValueError("; ".join(errors))
    return run_config(deployment_parameters, platform_constraints)


def apply_provider_facts(config: dict, data_provider) -> None:
    """Copy the measured dataset facts from the provider contract into config."""
    config["input_shape"] = data_provider.get_input_shape()
    config["input_size"] = int(np.prod(config["input_shape"]))
    config["num_classes"] = data_provider.get_prediction_mode().num_classes


def apply_workload_profiles(config: dict, data_provider) -> None:
    """Fold the registered data/model workload profiles beneath explicit config
    keys (explicit > model registration > data registration > framework default)."""
    fold_workload_profiles(config, **_workload_profiles(config, data_provider))


def _workload_profiles(config: dict, data_provider) -> dict:
    return {"model_profile": ModelRegistry.get_workload_profile(str(config.get("model_type") or "")),
            "data_profile": data_provider.workload_profile()}


def apply_judged_workload_profiles(config: dict, data_provider, deployment_parameters: dict,
                                   platform_constraints: dict) -> None:
    """``apply_workload_profiles`` with the run's refusals of what the registrations meet, by name
    (``registration_errors``: a registration-owned key declared otherwise, a declared value the registrations
    make illegal, a declaration the workload reading refuses), over the declarations as the run lays them; the
    document door states the same refusals from the builder registration (``run_derivation_errors``)."""
    profiles = _workload_profiles(config, data_provider)
    declared = declaration_judge.sections_run_layout(deployment_parameters, platform_constraints)
    errors = registration_errors(config, declared, (profiles["model_profile"], profiles["data_profile"]))
    if errors:
        raise DerivationRefusal("; ".join(errors))


class DeploymentPipeline(Pipeline):
    """Unified deployment pipeline with configurable quantization."""

    #: [U-H] The run's name, as the session reads it from the document (empty for a pipeline built without one);
    #: the records that name the run that made them read it here (a deployment bundle's provenance).
    deployment_name: str = ""

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
        self.config.update(
            merge_pipeline_config(deployment_parameters, platform_constraints)
        )

        if data_provider is None:
            data_provider = self.data_provider_factory.create()
        apply_provider_facts(self.config, data_provider)
        apply_judged_workload_profiles(self.config, data_provider, deployment_parameters, platform_constraints)
        self.config["device"] = select_device()

        if os.environ.get("MIMARSINAN_CUDA_DEBUG") == "1":
            self.config.setdefault("cuda_debug", True)

        self.plan = DeploymentPlan.resolve(self.config)
        self.cuda_debug = self.plan.cuda_debug

        self.tolerance = 1.0 - self.plan.degradation_tolerance
        if self.plan.scm_degradation_tolerance is not None:
            self.step_tolerances["Soft Core Mapping"] = (
                1.0 - self.plan.scm_degradation_tolerance
            )
        self.accuracy_budget.budget_total = self.plan.degradation_budget_total
        self._resolve_legacy_cached_placements()
        self._resolve_legacy_cached_soma_laws()

    def _resolve_legacy_cached_soma_laws(self) -> None:
        """[U-H] Give a pre-axes cached NF forward the run's point at the same safe moment (refused by name when the
        run declares a non-default point); a no-op for a fresh run and for every forward written with a point."""
        bound = resolve_cached_forward_soma_laws(self.cache, soma_law=SomaLaw.resolve(self.config))
        if bound:
            print(
                f"[DeploymentPipeline] bound this run's default soma point to {len(bound)} cached NF "
                f"forward(s) written before the soma axes existed: {bound}"
            )

    def _resolve_legacy_cached_placements(self) -> None:
        """Give a pre-stamp cached flow its placement, at the one safe moment.

        ``Pipeline.__init__`` has already loaded the cache and no step has run,
        so this is the earliest point where the configured placement is known
        AND no negative-boundary host placement of THIS run exists to erase.
        A no-op for a fresh run and for any artifact written with a stamp.
        """
        stamped = resolve_cached_flow_placements(
            self.cache,
            placement=str(self.config.get("encoding_layer_placement", "subsume")),
            packaging=packaging_contract_for(self.plan),
        )
        if stamped:
            # The STAMP, not the request: a value-domain resume is stamped
            # not-applicable, and saying "subsume" there would be a lie in the
            # one message a resuming reader has.
            print(
                f"[DeploymentPipeline] resolved encoding placement "
                f"{sorted(set(stamped.values()))} on {len(stamped)} cached "
                f"flow(s) written before the placement stamp existed: "
                f"{sorted(stamped)}"
            )

    def _display_config(self):
        plan = self.plan
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
        plan = self.plan
        if plan.pruning_enabled:
            print(f"[DeploymentPipeline] Pruning enabled: pruning={plan.pruning}, pruning_fraction={plan.pruning_fraction}; PruningAdaptationStep added.")
        else:
            print(f"[DeploymentPipeline] Pruning not in pipeline: pruning={plan.pruning}, pruning_fraction={plan.pruning_fraction}")

    @staticmethod
    def apply_preset(pipeline_mode: str, deployment_parameters: dict) -> None:
        """Merge a pipeline_mode preset into deployment_parameters (setdefault)."""
        _apply_preset(pipeline_mode, deployment_parameters)
