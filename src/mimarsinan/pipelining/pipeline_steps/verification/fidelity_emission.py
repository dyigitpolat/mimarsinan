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
from mimarsinan.deployment_record.objectives import (
    OBJECTIVES,
    full_candidate_probe,
)
from mimarsinan.deployment_record.platform_physics.resolve import (
    resolve_platform_physics,
)
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    firing_semantics_kwargs,
    make_platform_resolver,
)
from mimarsinan.search.results import resolve_active_specs
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_structured_pruning import (
    resolve_prune_criterion,
)
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.results import ACCURACY_OBJECTIVE_NAME


def _fidelity_axis_names(config: Any) -> Optional[List[str]]:
    """[H3] The FULL candidate-answerable surface this run can back.

    The searched axes only tell us the run searched at all; the twin predicts
    every axis a candidate can answer under the run's own declarations, so the
    report's prediction column stops depending on what the run happened to
    optimize. Per-axis admission reuses the step's own gate (physics +
    activity), one axis at a time — an axis the gate refuses is simply not
    predicted, and the report's basis says why.
    """
    arch = config.get("arch_search") or {}
    if not arch.get("objectives"):
        return None  # the run never searched: nothing was predicted
    probe = full_candidate_probe()
    physics = resolve_platform_physics(
        str(config.get("platform_physics_profile", "") or ""),
        config.get("platform_physics_overrides") or {},
    )
    activity = float(config.get("activity_factor", 0.0) or 0.0)
    names: List[str] = []
    for spec in OBJECTIVES.all():
        if spec.key == ACCURACY_OBJECTIVE_NAME or not spec.available(probe):
            continue
        try:
            resolve_active_specs(
                "hardware", [spec.key], physics=physics, activity_factor=activity,
            )
        except ValueError:
            continue
        names.append(spec.key)
    return names or None


def _fidelity_problem(
    pipeline: Any, names: List[str], deployed_platform: Any,
) -> JointArchHwProblem:
    """The DEPLOYED configuration as a hardware-mode candidate problem.

    Everything a VIEW needs comes from the same SSOTs the search step reads:
    the deployment's own platform resolver, the registry's builder, and the
    plan's declared pruning. The chip is the SEALED winner's — the record's
    own identity — because the raw config still holds the pre-search
    declaration; a twin of the declaration would compare the measurement
    against a chip the run never deployed. Training-side knobs are
    irrelevant — the fidelity view is never trained.
    """
    config = dict(pipeline.config)
    config["cores"] = [dict(ct) for ct in deployed_platform["cores"]]
    for key in ("weight_bits", "target_tq"):
        if deployed_platform.get(key) is not None:
            config[key] = deployed_platform[key]
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
        # [H3] The SAME semantics the search step hands its problem — a twin
        # armed differently prices a program the run never executed (H0: a
        # re-timed run's twin priced the fused wall).
        **firing_semantics_kwargs(plan, config),
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
    problem = _fidelity_problem(pipeline, names, record.identity.platform)
    # The EMPTY overlay: the fixed model config and the run's own resolved
    # platform — i.e. exactly the configuration that deployed.
    view = problem.candidate_layout(
        {"model_config": {}, "platform_constraints": {}}
    ).view
    return emit_fidelity_report(record, view, pipeline.working_directory)
