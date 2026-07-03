"""AgentEvolveOptimizer — LLM-based multi-objective search."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mimarsinan.common.best_effort import best_effort
from mimarsinan.search.optimizers.agent_evolve.batch_eval import BatchEvalMixin
from mimarsinan.search.optimizers.agent_evolve.llm_trace import LLMTraceMixin
from mimarsinan.search.optimizers.agent_evolve.prompting import PromptingMixin
from mimarsinan.search.optimizers.agent_evolve.codec import (
    compute_pareto_front,
    result_to_candidate,
    select_best_candidate_minimax,
    sort_pareto_results_minimax_first,
)
from mimarsinan.search.optimizers.agent_evolve.schema import (
    CandidateResult,
    format_search_space_description,
    prettify_configuration,
)
from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult

ConfigT = Dict[str, Any]


@dataclass
class AgentEvolveOptimizer(
    LLMTraceMixin,
    PromptingMixin,
    BatchEvalMixin,
    SearchOptimizer[ConfigT],
):
    """LLM-based multi-objective optimizer using pydantic-ai for agentic evolution."""

    pop_size: int = 8
    generations: int = 5
    candidates_per_batch: int = 5
    max_regen_rounds: int = 10
    max_failed_examples: int = 5

    model: str = "openai:gpt-4o"
    llm_retries: int = 3

    config_schema: Optional[Dict[str, Any]] = None
    example_config: Optional[ConfigT] = None
    constraints_description: Optional[str] = None

    verbose: bool = True
    invalid_penalty: float = 1e18

    _trace_reporter: Any = field(default=None, init=False, repr=False)
    _trace_gen: int = field(default=0, init=False, repr=False)
    _trace_seq: int = field(default=0, init=False, repr=False)

    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)

    def optimize(self, problem: SearchProblem[ConfigT], reporter=None) -> SearchResult[ConfigT]:
        """Run the LLM-based multi-objective optimization."""
        objectives = list(problem.objectives)
        if not objectives:
            raise ValueError("SearchProblem.objectives must not be empty")

        self._trace_reporter = reporter
        try:
            return asyncio.run(self._optimize_inner(problem, objectives))
        finally:
            self._trace_reporter = None

    def _append_generation_candidates(
        self,
        all_candidates: List[Candidate[ConfigT]],
        valid: List[CandidateResult],
        failed: List[CandidateResult],
        gen: int,
    ) -> None:
        for r in valid:
            all_candidates.append(result_to_candidate(r, {"generation": gen, "is_pareto": False}))
        for r in failed:
            all_candidates.append(
                result_to_candidate(r, {"generation": gen, "is_pareto": False, "valid": False})
            )

    def _emit_generation_complete(
        self,
        reporter,
        gen: int,
        valid_count: int,
        failed_count: int,
        pareto: List[CandidateResult],
        objectives: List[ObjectiveSpec],
        constraint_instruction: str,
        performance_insights: str,
    ) -> None:
        pareto_sorted = sort_pareto_results_minimax_first(pareto, objectives)
        pareto_front_summary = [r.objectives for r in pareto_sorted[:5]]
        self._report_search_event(reporter, {
            "type": "generation_complete",
            "gen": gen,
            "valid_count": valid_count,
            "failed_count": failed_count,
            "pareto_size": len(pareto),
            "pareto_front": pareto_front_summary,
            "constraint_instruction": (
                constraint_instruction[: self._TRACE_MAX_GEN_COMPLETE_STR]
                if constraint_instruction
                else ""
            ),
            "performance_insights": (
                performance_insights[: self._TRACE_MAX_GEN_COMPLETE_STR]
                if performance_insights
                else ""
            ),
        })

    def _report_generation_metrics(
        self,
        reporter,
        gen: int,
        valid_count: int,
        pareto_size: int,
    ) -> None:
        if not reporter:
            return
        with best_effort("report generation metrics"):
            reporter("Search generation", gen)
            reporter("Search valid count", valid_count)
            reporter("Search Pareto size", pareto_size)

    async def _optimize_inner(
        self,
        problem: SearchProblem[ConfigT],
        objectives: List[ObjectiveSpec],
    ) -> SearchResult[ConfigT]:
        reporter = self._trace_reporter

        search_space_desc = format_search_space_description(
            objectives=objectives,
            config_schema=self.config_schema,
            example_config=self.example_config,
            constraints=self.constraints_description,
        )

        all_valid_results: List[CandidateResult] = []
        all_failed_results: List[CandidateResult] = []
        all_candidates: List[Candidate[ConfigT]] = []
        history: List[Dict[str, Any]] = []

        constraint_instruction = ""
        performance_insights = ""

        self._trace_gen = 1
        self._trace_seq = 0

        self._log(f"=== Generation 1 / {self.generations} (initial sampling) ===")
        self._report_search_event(reporter, {
            "type": "generation_start",
            "gen": 1, "total_gens": self.generations, "phase": "initial",
            "objectives": [{"name": s.name, "goal": s.goal} for s in objectives],
        })

        gen1_valid, gen1_failed, constraint_instruction = await self._run_initial_generation(
            problem=problem,
            objectives=objectives,
            search_space_desc=search_space_desc,
            all_failed_results=all_failed_results,
            constraint_instruction=constraint_instruction,
            performance_insights=performance_insights,
            reporter=reporter,
        )

        all_valid_results.extend(gen1_valid)
        all_failed_results.extend(gen1_failed)
        self._append_generation_candidates(all_candidates, gen1_valid, gen1_failed, 1)

        pareto = compute_pareto_front(all_valid_results, objectives)
        self._log(f"Generation 1: {len(gen1_valid)} valid, Pareto size={len(pareto)}")

        if all_valid_results:
            performance_insights = await self._generate_performance_insights(
                valid_results=all_valid_results,
                objectives=objectives,
                search_space_desc=search_space_desc,
            )

        history.append({
            "gen": 1,
            "valid_count": len(gen1_valid),
            "failed_count": len(gen1_failed),
            "pareto_size": len(pareto),
        })
        self._report_generation_metrics(reporter, 1, len(gen1_valid), len(pareto))
        self._emit_generation_complete(
            reporter, 1, len(gen1_valid), len(gen1_failed), pareto, objectives,
            constraint_instruction, performance_insights,
        )

        prev_pareto = pareto

        for gen in range(2, self.generations + 1):
            self._log(f"\n=== Generation {gen} / {self.generations} ===")

            self._trace_gen = gen
            self._trace_seq = 0
            self._report_search_event(reporter, {
                "type": "generation_start",
                "gen": gen, "total_gens": self.generations, "phase": "evolution",
                "objectives": [{"name": s.name, "goal": s.goal} for s in objectives],
            })

            gen_valid, gen_failed, constraint_instruction = await self._run_evolution_generation(
                problem=problem,
                objectives=objectives,
                search_space_desc=search_space_desc,
                prev_pareto=prev_pareto,
                all_failed_results=all_failed_results,
                constraint_instruction=constraint_instruction,
                performance_insights=performance_insights,
                gen=gen,
                reporter=reporter,
            )

            all_valid_results.extend(gen_valid)
            all_failed_results.extend(gen_failed)
            self._append_generation_candidates(all_candidates, gen_valid, gen_failed, gen)

            pareto = compute_pareto_front(all_valid_results, objectives)

            if all_valid_results:
                performance_insights = await self._update_performance_insights(
                    previous_insights=performance_insights,
                    valid_results=all_valid_results,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )

            self._log(f"Generation {gen}: {len(gen_valid)} valid, Pareto size={len(pareto)}")

            history.append({
                "gen": gen,
                "valid_count": len(gen_valid),
                "failed_count": len(gen_failed),
                "pareto_size": len(pareto),
            })
            self._report_generation_metrics(reporter, gen, len(gen_valid), len(pareto))
            self._emit_generation_complete(
                reporter, gen, len(gen_valid), len(gen_failed), pareto, objectives,
                constraint_instruction, performance_insights,
            )

            prev_pareto = pareto

        final_pareto = compute_pareto_front(all_valid_results, objectives)

        pareto_configs = {prettify_configuration(r.configuration) for r in final_pareto}
        pareto_candidates: List[Candidate[ConfigT]] = []
        for r in final_pareto:
            pareto_candidates.append(result_to_candidate(r, {"is_pareto": True}))

        for c in all_candidates:
            c_key = prettify_configuration(c.configuration)
            if c_key in pareto_configs:
                c.metadata["is_pareto"] = True

        best_result = select_best_candidate_minimax(final_pareto, objectives)
        if best_result:
            best = result_to_candidate(best_result, {"is_pareto": True})
        else:
            best = Candidate(configuration={}, objectives={}, metadata={})

        self._log(f"\n=== Final Results ===")
        self._log(f"Total valid: {len(all_valid_results)}, Pareto size: {len(final_pareto)}")
        if best_result:
            self._log(f"Best: {best_result.objectives}")

        self._report_search_event(reporter, {
            "type": "search_complete",
            "total_valid": len(all_valid_results),
            "total_failed": len(all_failed_results),
            "final_pareto_size": len(final_pareto),
        })

        return SearchResult(
            objectives=objectives,
            best=best,
            pareto_front=pareto_candidates,
            all_candidates=all_candidates,
            history=history,
        )
