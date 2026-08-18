"""Shared helpers for :class:`ArchitectureSearchStep`."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.json_util import to_json_safe
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)
from mimarsinan.search.optimizers.budget import EvaluationBudget
from mimarsinan.search.problems.joint import PlatformResolver
from mimarsinan.search.search_space_description import SearchSpaceDescription
from mimarsinan.visualization.search_viz import (
    create_interactive_search_report,
    write_final_population_json,
)


OptimizerType = Literal["nsga2", "agent_evolve", "compilagent"]


def declared_int_pair(
    arch_cfg: Mapping[str, Any], key: str, default: Tuple[int, int],
) -> Tuple[int, int]:
    """A declared ``[low, high]`` search bound as the int pair the problem takes."""
    low, high = tuple(arch_cfg.get(key, default))
    return int(low), int(high)


def resolve_evaluation_budget(
    arch_cfg: Mapping[str, Any],
) -> Optional[EvaluationBudget]:
    """[TS1] The run's evaluation accountant, or None when none was declared.

    The declaration is a COUNT of distinct evaluations, the currency a campaign
    compares optimizers in. An undeclared budget meters nothing and the run is
    byte-identical to one from before the accountant existed; a declared
    non-count is refused by name rather than silently meaning "unlimited".
    """
    declared = arch_cfg.get("evaluation_budget")
    if declared is None:
        return None
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
        raise ValueError(
            f"arch_search.evaluation_budget must be a positive number of distinct "
            f"evaluations, or absent for an unmetered run; got {declared!r}"
        )
    return EvaluationBudget(limit=int(declared))


def create_optimizer(
    optimizer_type: OptimizerType,
    arch_cfg: Dict[str, Any],
    search_mode: str,
    arch_options: List[Tuple[str, List[Any]]],
    seed: int,
    pop_size: int,
    generations: int,
    target_tq: int,
    active_objective_names: Sequence[str] = (),
):
    description = SearchSpaceDescription.from_arch_search(
        search_mode=search_mode,
        arch_options=arch_options,
        arch_cfg=arch_cfg,
        target_tq=target_tq,
    )

    if optimizer_type == "agent_evolve":
        try:
            from mimarsinan.search.optimizers.agent_evolve import AgentEvolveOptimizer

            agent_model = arch_cfg.get("agent_model", "openai:gpt-4o")
            candidates_per_batch = arch_cfg.get("candidates_per_batch", 5)
            max_regen_rounds = arch_cfg.get("max_regen_rounds", 10)
            max_failed_examples = arch_cfg.get("max_failed_examples", 5)
            llm_retries = arch_cfg.get("llm_retries", 3)

            config_schema = description.to_agent_evolve_schema()
            example_config = description.to_agent_evolve_example()
            constraints_desc = (
                arch_cfg.get("constraints_description")
                or description.to_agent_evolve_constraints()
            )

            return AgentEvolveOptimizer(
                pop_size=pop_size,
                generations=generations,
                candidates_per_batch=candidates_per_batch,
                max_regen_rounds=max_regen_rounds,
                max_failed_examples=max_failed_examples,
                model=agent_model,
                llm_retries=llm_retries,
                config_schema=config_schema,
                example_config=example_config,
                constraints_description=constraints_desc,
                verbose=True,
            )
        except ImportError as e:
            print(f"[ArchitectureSearchStep] Agentic Evolution optimizer not available: {e}")
            print("[ArchitectureSearchStep] Falling back to NSGA2")

    if optimizer_type == "compilagent":
        from mimarsinan.search.optimizers.compilagent import CompilagentOptimizer

        return CompilagentOptimizer(
            pop_size=int(pop_size),
            description=description,
            model=str(arch_cfg.get("model", "openai:gpt-4o")),
            harness_id=str(arch_cfg.get("harness", "pydantic_ai")),
            max_candidates=int(arch_cfg.get("max_candidates", max(pop_size, 8))),
            max_continuations=int(arch_cfg.get("max_continuations", 4)),
            system_prompt_extra=str(arch_cfg.get("system_prompt_extra", "")),
            active_objective_names=tuple(active_objective_names),
            verbose=True,
        )

    from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer

    return NSGA2Optimizer(
        pop_size=pop_size,
        generations=generations,
        seed=seed,
        eliminate_duplicates=True,
        verbose=True,
    )


def search_result_to_jsonable(result) -> Dict[str, Any]:
    def cand_to_dict(c):
        return {
            "configuration": c.configuration,
            "objectives": c.objectives,
            "metadata": c.metadata,
        }

    payload = {
        "objectives": [{"name": o.name, "goal": o.goal} for o in result.objectives],
        "best": cand_to_dict(result.best),
        "pareto_front": [cand_to_dict(c) for c in result.pareto_front],
        "all_candidates": [cand_to_dict(c) for c in result.all_candidates],
        "history": result.history,
    }
    # [TS1] An unmetered run seals no ledger, and its artifact stays exactly
    # what it was before the accountant existed.
    if result.ledger is not None:
        payload["ledger"] = result.ledger.to_dict()
    return to_json_safe(payload)


def derive_arch_options(
    builder_cls: type,
    arch_cfg: Dict[str, Any],
    input_shape: tuple,
) -> Tuple[List[Tuple[str, List[Any]]], Dict[str, Any]]:
    schema = getattr(builder_cls, "get_config_schema", lambda: [])()
    schema_map = {f["key"]: f for f in schema}

    nas_opts_fn = getattr(builder_cls, "get_nas_search_options", None)
    builder_options: Dict[str, List[Any]] = (
        nas_opts_fn(input_shape=input_shape) if nas_opts_fn else {}
    )

    arch_options: List[Tuple[str, List[Any]]] = []
    for field_desc in schema:
        key = field_desc["key"]
        field_type = field_desc.get("type")
        if field_type == "select" and "options" in field_desc:
            values = arch_cfg.get(f"{key}_options", field_desc["options"])
            if len(values) > 1:
                arch_options.append((key, list(values)))
        elif key in builder_options:
            values = arch_cfg.get(f"{key}_options", builder_options[key])
            if len(values) > 1:
                arch_options.append((key, list(values)))

    schema_keys = {f["key"] for f in schema}
    for key, default_values in builder_options.items():
        if key not in schema_keys:
            values = arch_cfg.get(f"{key}_options", default_values)
            if len(values) > 1:
                arch_options.append((key, list(values)))

    return arch_options, schema_map


def make_assembler(schema: List[Dict[str, Any]], schema_map: Dict[str, Any]):
    def assembler(raw: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for field_desc in schema:
            if "default" in field_desc:
                result[field_desc["key"]] = field_desc["default"]
        for k, v in raw.items():
            field_info = schema_map.get(k, {})
            if field_info.get("type") == "number":
                default = field_info.get("default", 0)
                result[k] = float(v) if isinstance(default, float) else int(v)
            else:
                result[k] = v
        return result
    return assembler


def make_platform_resolver(pipeline_config: Mapping[str, Any]) -> PlatformResolver:
    """The search's platform resolution: this run's DEPLOYMENT resolver, curried.

    A candidate declares only what it searches (core dimensions, ``target_tq``);
    every other property of the chip comes from re-running the deployment's own
    ``build_platform_constraints_resolved`` over the declared platform with that
    overlay applied. Candidate and deployed twin are therefore the same chip by
    construction — there is no carried key list between them to drift, and no
    resolution mode that could serve the search a reduced surface.
    """
    declared = dict(pipeline_config)

    def resolve(overlay: Mapping[str, Any]) -> Dict[str, Any]:
        return build_platform_constraints_resolved({**declared, **overlay})

    return resolve


def build_fixed_platform_constraints(pipeline_config: Mapping[str, Any]) -> Dict[str, Any]:
    """The resolved platform with NO candidate overlay — the declared chip itself."""
    return make_platform_resolver(pipeline_config)({})


def write_search_visualizations(result_json: Dict[str, Any], out_dir: str) -> None:
    with best_effort("architecture-search visualization report"):
        write_final_population_json(result_json, os.path.join(out_dir, "final_population.json"))
        report_html = os.path.join(out_dir, "search_report.html")
        create_interactive_search_report(result_json, report_html)

        for legacy in ["search_report.pdf", "search_report.png"]:
            legacy_path = os.path.join(out_dir, legacy)
            if os.path.exists(legacy_path):
                with best_effort(f"remove legacy report {legacy}"):
                    os.remove(legacy_path)


def resolve_arch_options(
    builder_cls, arch_cfg: Dict, input_shape: Tuple, *,
    searches_model: bool, model_type: str,
):
    """The model-side search space and its raw->config assembler for one run.

    A hardware-only search has neither: its model config is fixed, so the
    options list is empty and the assembler is the identity.
    """
    if not searches_model:
        return [], (lambda raw: dict(raw))

    arch_options, schema_map = derive_arch_options(builder_cls, arch_cfg, input_shape)
    schema = getattr(builder_cls, "get_config_schema", lambda: [])()
    if not arch_options:
        raise NotImplementedError(
            f"No NAS search space defined for model_type='{model_type}'. "
            f"Add get_nas_search_options() or 'select' fields with multiple options "
            f"to {builder_cls.__name__}.get_config_schema(). "
            f"Current schema keys: {[f['key'] for f in schema]}"
        )
    return arch_options, make_assembler(schema, schema_map)


def firing_semantics_kwargs(plan, config: Mapping[str, Any]) -> Dict[str, Any]:
    """[H3] ONE resolution of the run's firing semantics, for every consumer.

    The search step and the fidelity twin must construct the SAME problem
    parameterization or the twin prices a program the run never executed —
    the exact drift H0 measured (a re-timed run's twin priced the fused wall).
    """
    from mimarsinan.chip_simulation.spiking_semantics import (
        lif_per_hop_retiming_enabled,
    )
    from mimarsinan.models.spiking.hybrid.carry import run_pass_transfer

    return {
        "spiking_mode": str(plan.spiking_mode),
        "ttfs_cycle_schedule": str(plan.ttfs_cycle_schedule),
        "per_hop_retiming": bool(lif_per_hop_retiming_enabled(config)),
        "pass_transfer": str(run_pass_transfer(config)),
    }
