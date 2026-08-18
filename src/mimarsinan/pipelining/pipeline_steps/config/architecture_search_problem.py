"""The problem :class:`ArchitectureSearchStep` searches, and how it reads back.

One place assembles the candidate contract from THIS run's declaration, so the
step is left saying what a search step does: build the problem, run a driver
over it, and deploy the winner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from mimarsinan.config_schema.registry import effective_value as _effective
from mimarsinan.deployment_record.platform_physics.resolve import (
    resolve_platform_physics,
)
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.pipeline_steps.config.architecture_search_helpers import (
    declared_int_pair,
    firing_semantics_kwargs,
    make_platform_resolver,
    resolve_evaluation_budget,
)
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_structured_pruning import (
    resolve_prune_criterion,
)
from mimarsinan.search.option_axes import build_option_axes
from mimarsinan.search.problems.joint import JointArchHwProblem
from mimarsinan.search.results import resolve_active_objectives


def build_search_problem(
    pipeline,
    *,
    search_mode: str,
    builder_cls: type,
    arch_options: List[Tuple[str, List[Any]]],
    model_config_assembler,
    seed: int,
) -> Tuple[JointArchHwProblem, List[str]]:
    """This run's candidate contract, and the objective names it will be scored on."""
    config = pipeline.config
    arch_cfg = config.get("arch_search", {})
    plan = DeploymentPlan.of(pipeline)
    validate_config_fn = getattr(builder_cls, "validate_config", None)

    def validate_fn(model_config, platform_constraints, inp_shape):
        if validate_config_fn is not None:
            return bool(validate_config_fn(model_config, platform_constraints, inp_shape))
        return True

    def constraint_fn(model_config, platform_constraints, inp_shape):
        if validate_config_fn is not None:
            if not validate_config_fn(model_config, platform_constraints, inp_shape):
                return 1.0
        return 0.0

    # THIS run's physics, from the same SSOT the deployment resolver uses: a
    # vendor-priced axis is refused by name here, before any candidate is built.
    run_physics = resolve_platform_physics(
        str(config.get("platform_physics_profile", "") or ""),
        config.get("platform_physics_overrides") or {},
    )
    active_objective_names = [
        o.name for o in resolve_active_objectives(
            search_mode, arch_cfg.get("objectives"), physics=run_physics,
            activity_factor=float(config.get("activity_factor", 0.0) or 0.0),
        )
    ]

    training_batch_size = arch_cfg.get("training_batch_size") or None
    problem = JointArchHwProblem(
        data_provider_factory=pipeline.data_provider_factory,
        device=config["device"],
        input_shape=tuple(config["input_shape"]),
        num_classes=int(config["num_classes"]),
        target_tq=int(config["target_tq"]),
        lr=float(config["lr"]),
        search_mode=search_mode,
        builder_factory=builder_cls,
        arch_options=arch_options,
        model_config_assembler=model_config_assembler,
        validate_fn=validate_fn,
        constraint_fn=constraint_fn,
        # A hardware-only search declares its model; every other mode searches it.
        fixed_model_config=(
            dict(config.get("model_config", {})) if search_mode == "hardware" else None
        ),
        # Every candidate platform is this run's DEPLOYMENT resolution with the
        # searched dimensions overlaid — searched chip and deployed chip are the
        # same chip by construction, in every mode.
        platform_resolver=make_platform_resolver(config),
        active_objective_names=active_objective_names,
        num_core_types=int(arch_cfg.get(
            "num_core_types", len(config.get("cores", [])) or 1
        )),
        core_axons_bounds=declared_int_pair(arch_cfg, "core_axons_bounds", (64, 2048)),
        core_neurons_bounds=declared_int_pair(arch_cfg, "core_neurons_bounds", (64, 2048)),
        core_count_bounds=declared_int_pair(arch_cfg, "core_count_bounds", (50, 500)),
        accuracy_seed=seed,
        warmup_fraction=float(arch_cfg.get("warmup_fraction", 0.10)),
        training_batch_size=(
            int(training_batch_size) if training_batch_size is not None else None
        ),
        accuracy_evaluator=str(arch_cfg.get("accuracy_evaluator", "extrapolating")),
        extrapolation_num_train_epochs=int(
            arch_cfg.get("extrapolation_num_train_epochs", 1)
        ),
        extrapolation_num_checkpoints=int(
            arch_cfg.get("extrapolation_num_checkpoints", 5)
        ),
        extrapolation_target_epochs=int(
            arch_cfg.get("extrapolation_target_epochs", 10)
        ),
        pruning_fraction=plan.pruning_fraction, pruning=plan.pruning,
        prune_sparsity=plan.prune_sparsity,
        prune_criterion=resolve_prune_criterion(config),
        firing_mode=str(config.get("firing_mode", "Default")),
        **firing_semantics_kwargs(plan, config),
        encoding_placement=str(config.get("encoding_layer_placement", "subsume")),
        # Deployment options the run promoted to search axes, and the floor
        # that shapes the feasible region they move through.
        option_axes=build_option_axes(arch_cfg.get("option_axes")),
        onchip_min_fraction=(
            float(_effective(config, "onchip_min_fraction"))
            if bool(_effective(config, "onchip_majority_gate"))
            else 0.0
        ),
        # [TS1] What this run declared it may spend; absent = unmetered.
        evaluation_budget=resolve_evaluation_budget(arch_cfg),
    )
    return problem, active_objective_names


def no_candidate_failure(problem: JointArchHwProblem) -> str:
    """Why a search produced nothing, in the problem's OWN recorded words.

    A bare "no candidates" sent the operator hunting bounds when the cause
    could be a single semantic refusal shared by every offspring.
    """
    reasons: Dict[str, int] = {}
    for verdict in getattr(problem, "_validation_errors", {}).values():
        key = f"{verdict.failure_phase}: {(verdict.error_message or '')[:160]}"
        reasons[key] = reasons.get(key, 0) + 1
    census = problem.constraint_census()
    detail = "; ".join(
        f"{count}x {reason}" for reason, count in
        sorted(reasons.items(), key=lambda kv: -kv[1])[:3]
    ) or "no recorded validation errors"
    return (
        "[ArchitectureSearchStep] Architecture search produced no candidates. "
        f"Failure census: {detail}. Constraint census: {census or 'none'}. "
        "Consider relaxing the named cause before touching "
        "pop_size/generations/bounds."
    )
