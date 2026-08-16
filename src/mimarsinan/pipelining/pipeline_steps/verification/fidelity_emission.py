"""``fidelity.json`` emission — the candidate view of the run's OWN config (C4).

The contract the C4 machinery states but nothing wired into the pipeline
until now: per sealed run of a SEARCHED deployment, rebuild the candidate
view of the deployed configuration through the search problem's own path
(same platform resolver, same layout hook, same fragments) and zip it
axis-by-axis against the measurement. A run that never searched writes
nothing — no prediction, no correlation.
"""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from mimarsinan.deployment_record.fidelity_build import emit_fidelity_report
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.registry.model_registry import ModelRegistry
from mimarsinan.pipelining.core.search_mode import derive_search_mode
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    make_platform_resolver,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_structured_pruning import (
    resolve_prune_criterion,
)
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME


def _fidelity_axis_names(config: Any) -> Optional[List[str]]:
    """The run's SEARCHED axes, minus the training proxy (the rebuild never
    trains — hardware completeness is what a deployed config can answer)."""
    arch = config.get("arch_search") or {}
    names = arch.get("objectives")
    if not names:
        return None
    kept = [n for n in names if n != ACCURACY_OBJECTIVE_NAME]
    return kept or None


def _fidelity_problem(pipeline: Any, names: List[str]) -> JointArchHwProblem:
    """The deployed configuration as a hardware-mode candidate problem.

    Everything a VIEW needs comes from the same SSOTs the search step reads:
    the deployment's own platform resolver, the registry's builder, and the
    plan's declared pruning. Training-side knobs are irrelevant — the
    fidelity view is never trained.
    """
    config = pipeline.config
    plan = DeploymentPlan.of(pipeline)
    return JointArchHwProblem(
        data_provider_factory=None,
        device=torch.device("cpu"),
        input_shape=tuple(config["input_shape"]),
        num_classes=int(config["num_classes"]),
        target_tq=int(config["target_tq"]),
        lr=float(config.get("lr", 1e-3)),
        search_mode="hardware",
        builder_factory=ModelRegistry.get_builder_cls(str(config["model_type"])),
        arch_options=(),
        model_config_assembler=lambda raw: dict(raw),
        fixed_model_config=dict(config.get("model_config") or {}),
        platform_resolver=make_platform_resolver(config),
        active_objective_names=names,
        num_core_types=1,
        core_axons_bounds=(1, 4096),
        core_neurons_bounds=(1, 4096),
        core_count_bounds=(1, 4096),
        accuracy_seed=int(config.get("seed", 0) or 0),
        pruning_fraction=float(plan.pruning_fraction),
        pruning=bool(plan.pruning),
        prune_sparsity=float(plan.prune_sparsity),
        prune_criterion=resolve_prune_criterion(config),
        firing_mode=str(config.get("firing_mode", "Default")),
        encoding_placement=str(
            config.get("encoding_layer_placement", "subsume")
        ),
    )


def emit_run_fidelity(pipeline: Any, record: Any) -> Optional[str]:
    """Write ``fidelity.json`` beside the sealed record, when the run searched.

    Fail loud: a candidate rebuild that cannot resolve the run's own deployed
    configuration is a twin defect, which is exactly what fidelity exists to
    surface.
    """
    config = pipeline.config
    if derive_search_mode(config) == "fixed":
        return None
    names = _fidelity_axis_names(config)
    if names is None:
        return None
    problem = _fidelity_problem(pipeline, names)
    # The EMPTY overlay: the fixed model config and the run's own resolved
    # platform — i.e. exactly the configuration that deployed.
    view = problem.candidate_layout(
        {"model_config": {}, "platform_constraints": {}}
    ).view
    return emit_fidelity_report(record, view, pipeline.working_directory)
