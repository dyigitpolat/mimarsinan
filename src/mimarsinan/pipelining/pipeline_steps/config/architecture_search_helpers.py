"""Shared helpers for :class:`ArchitectureSearchStep`."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import (
    Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple,
)

from mimarsinan.common.best_effort import best_effort
from mimarsinan.gui.json_util import to_json_safe
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)
from mimarsinan.search.optimizers.budget import EvaluationBudget
from mimarsinan.search.optimizers.catalog import OPTIMIZER_IDS
from mimarsinan.search.optimizers.sampling_optimizer import (
    DEFAULT_GRID_CAP, GridStrategy, RandomStrategy, SamplingOptimizer,
    SamplingStrategy, SobolStrategy,
)
from mimarsinan.search.problems.joint import PlatformResolver
from mimarsinan.search.search_space_description import SearchSpaceDescription
from mimarsinan.visualization.search_viz import (
    create_interactive_search_report,
    write_final_population_json,
)


#: The declared names, kept in step with :data:`OPTIMIZER_IDS` by test.
OptimizerType = Literal[
    "nsga2", "agent_evolve", "compilagent", "random", "sobol", "exhaustive",
]


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


@dataclass(frozen=True)
class OptimizerRequest:
    """[TS2] What every backend is built from — one shape, one builder each."""

    arch_cfg: Mapping[str, Any]
    description: SearchSpaceDescription
    seed: int
    pop_size: int
    generations: int
    active_objective_names: Sequence[str] = ()

    def declared(self, key: str, default: Any) -> Any:
        return self.arch_cfg.get(key, default)

    @property
    def planned_samples(self) -> int:
        """The draw a sampling backend plans: the same budget NSGA-II spends."""
        return int(self.pop_size) * int(self.generations)


def _build_nsga2(request: OptimizerRequest):
    from mimarsinan.search.optimizers.nsga2_optimizer import NSGA2Optimizer

    return NSGA2Optimizer(
        pop_size=request.pop_size,
        generations=request.generations,
        seed=request.seed,
        eliminate_duplicates=True,
        verbose=True,
    )


def _build_agent_evolve(request: OptimizerRequest):
    try:
        from mimarsinan.search.optimizers.agent_evolve import AgentEvolveOptimizer
    except ImportError as e:
        print(f"[ArchitectureSearchStep] Agentic Evolution optimizer not available: {e}")
        print("[ArchitectureSearchStep] Falling back to NSGA2")
        return _build_nsga2(request)

    description = request.description
    return AgentEvolveOptimizer(
        pop_size=request.pop_size,
        generations=request.generations,
        candidates_per_batch=request.declared("candidates_per_batch", 5),
        max_regen_rounds=request.declared("max_regen_rounds", 10),
        max_failed_examples=request.declared("max_failed_examples", 5),
        model=request.declared("agent_model", "openai:gpt-4o"),
        llm_retries=request.declared("llm_retries", 3),
        config_schema=description.to_agent_evolve_schema(),
        example_config=description.to_agent_evolve_example(),
        constraints_description=(
            request.declared("constraints_description", None)
            or description.to_agent_evolve_constraints()
        ),
        verbose=True,
    )


def _build_compilagent(request: OptimizerRequest):
    from mimarsinan.search.optimizers.compilagent import CompilagentOptimizer

    return CompilagentOptimizer(
        pop_size=int(request.pop_size),
        description=request.description,
        model=str(request.declared("model", "openai:gpt-4o")),
        harness_id=str(request.declared("harness", "pydantic_ai")),
        max_candidates=int(request.declared("max_candidates", max(request.pop_size, 8))),
        max_continuations=int(request.declared("max_continuations", 4)),
        system_prompt_extra=str(request.declared("system_prompt_extra", "")),
        active_objective_names=tuple(request.active_objective_names),
        verbose=True,
    )


def _sampling(request: OptimizerRequest, strategy: SamplingStrategy) -> SamplingOptimizer:
    return SamplingOptimizer(
        strategy=strategy, pop_size=int(request.pop_size), seed=int(request.seed),
    )


def _build_random(request: OptimizerRequest):
    return _sampling(request, RandomStrategy(samples=request.planned_samples))


def _build_sobol(request: OptimizerRequest):
    return _sampling(request, SobolStrategy(samples=request.planned_samples))


def _build_exhaustive(request: OptimizerRequest):
    return _sampling(request, GridStrategy(
        cap=int(request.declared("grid_cap", DEFAULT_GRID_CAP)),
    ))


#: [TS2] Name -> builder. THE dispatch: an if/elif ladder let a name reach the
#: pipeline while the wizard, the declared type and the ladder itself disagreed
#: about which names exist.
OPTIMIZER_BUILDERS: Dict[str, Callable[[OptimizerRequest], Any]] = {
    "nsga2": _build_nsga2,
    "agent_evolve": _build_agent_evolve,
    "compilagent": _build_compilagent,
    "random": _build_random,
    "sobol": _build_sobol,
    "exhaustive": _build_exhaustive,
}


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
    """The declared backend, built — or a refusal naming every choice there is."""
    builder = OPTIMIZER_BUILDERS.get(str(optimizer_type))
    if builder is None:
        raise ValueError(
            f"unknown arch_search.optimizer {optimizer_type!r}; declare one of "
            f"{', '.join(OPTIMIZER_IDS)}"
        )
    return builder(OptimizerRequest(
        arch_cfg=arch_cfg,
        description=SearchSpaceDescription.from_arch_search(
            search_mode=search_mode,
            arch_options=arch_options,
            arch_cfg=arch_cfg,
            target_tq=target_tq,
        ),
        seed=seed,
        pop_size=pop_size,
        generations=generations,
        active_objective_names=active_objective_names,
    ))


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
