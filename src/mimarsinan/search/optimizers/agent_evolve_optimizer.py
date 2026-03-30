"""
Agentic Evolution: LLM-based multi-objective optimizer using pydantic-ai.

This optimizer uses agentic reasoning to explore the search space,
learning from failures and performance patterns to guide the search.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, get_args, get_origin

from mimarsinan.search.optimizers.base import SearchOptimizer
from mimarsinan.search.optimizers.agent_evolve_support import (
    CandidateResult,
    compute_pareto_front,
    compute_performance_stats,
    dominates,
    format_search_space_description,
    prettify_configuration,
    prettify_objectives,
    prettify_results,
    result_to_candidate,
    sample_failed_for_constraint,
    select_best_candidate,
)
from mimarsinan.search.optimizers.agent_evolve_prompts import (
    build_constraint_instruction_prompt,
    build_failure_insights_prompt,
    build_initial_candidates_prompt,
    build_offspring_prompt,
    build_performance_insights_prompt,
    build_regenerate_candidates_prompt,
    build_regenerate_offspring_prompt,
    build_update_constraint_prompt,
    build_update_performance_insights_prompt,
    parse_candidates,
)
from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import Candidate, ObjectiveSpec, SearchResult


# Type alias for configuration
ConfigT = Dict[str, Any]


@dataclass
class AgentEvolveOptimizer(SearchOptimizer[ConfigT]):
    """
    LLM-based multi-objective optimizer using pydantic-ai for agentic evolution.

    This optimizer uses an agentic approach where the LLM:
    1. Generates initial candidate configurations
    2. Learns from validation/evaluation failures via insights
    3. Consolidates constraint knowledge over generations
    4. Uses performance analysis to guide search direction
    5. Generates offspring from Pareto front patterns
    """

    # Search hyperparameters
    pop_size: int = 8
    generations: int = 5
    candidates_per_batch: int = 5
    max_regen_rounds: int = 10
    max_failed_examples: int = 5

    # LLM configuration
    model: str = "openai:gpt-4o"
    llm_retries: int = 3

    # Problem description (optional, for better LLM context)
    config_schema: Optional[Dict[str, Any]] = None
    example_config: Optional[ConfigT] = None
    constraints_description: Optional[str] = None

    # Verbosity
    verbose: bool = True

    # Penalty for invalid candidates
    invalid_penalty: float = 1e18

    # Internal state (not part of dataclass init)
    _agent: Any = field(default=None, init=False, repr=False)

    def _get_agent(self) -> Any:
        """Lazily initialize the pydantic-ai Agent."""
        if self._agent is None:
            from pydantic_ai import Agent
            self._agent = Agent(model=self.model, retries=self.llm_retries)
        return self._agent

    def _log(self, message: str) -> None:
        """Print a message if verbose mode is enabled."""
        if self.verbose:
            print(message, flush=True)

    @staticmethod
    def _schema_has_dict_type(output_schema: Dict[str, type]) -> bool:
        """Return True if any field type contains an open dict (additionalProperties issue)."""
        for field_type in output_schema.values():
            origin = get_origin(field_type)
            if origin is dict:
                return True
            if origin is list:
                args = get_args(field_type)
                if args and get_origin(args[0]) is dict:
                    return True
        return False

    def _llm_call(
        self,
        template: str,
        output_schema: Dict[str, type],
    ) -> Any:
        """
        Make an LLM call with the given template and output schema.

        When the schema contains Dict-valued fields (e.g. List[Dict[str, Any]]),
        some providers (Gemini) strip `additionalProperties` from the JSON schema
        and return empty objects. In that case we fall back to plain-text output
        and parse the JSON ourselves, which works universally.

        For simple field types (str, List[str], etc.) we use pydantic-ai structured
        output as normal.
        """
        agent = self._get_agent()

        if self._schema_has_dict_type(output_schema):
            # Plain-text mode: ask the LLM to return a JSON object, parse manually.
            keys = list(output_schema.keys())
            augmented = (
                template
                + f"\n\nRespond with a single valid JSON object containing exactly "
                f"these keys: {keys}. Output only the JSON — no markdown, no explanation."
            )
            result = asyncio.run(agent.run(augmented, output_type=str))
            raw = getattr(result, "output", "") or ""

            # Extract JSON from the raw text response
            data: Dict[str, Any] = {}
            try:
                data = json.loads(raw.strip())
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                    except Exception:
                        data = {}

            # Build a namespace with proper defaults for missing keys
            ns: Dict[str, Any] = {}
            for k, v in output_schema.items():
                val = data.get(k)
                if val is None:
                    origin = get_origin(v)
                    ns[k] = [] if (origin is list or origin is dict) else ""
                else:
                    ns[k] = val
            return SimpleNamespace(**ns)

        else:
            # Structured pydantic-ai output for simple field types
            from pydantic import BaseModel, create_model

            output_model = create_model(
                "_OutputModel",
                __base__=BaseModel,
                **{k: (v, ...) for k, v in output_schema.items()},
            )
            result = asyncio.run(agent.run(template, output_type=output_model))
            return getattr(result, "output", result)

    def optimize(self, problem: SearchProblem[ConfigT], reporter=None) -> SearchResult[ConfigT]:
        """
        Run the LLM-based multi-objective optimization.

        Args:
            problem: The search problem to optimize
            reporter: Optional callable(name, value) for per-generation reporting

        Returns:
            SearchResult containing Pareto front and all evaluated candidates
        """
        objectives = list(problem.objectives)
        if not objectives:
            raise ValueError("SearchProblem.objectives must not be empty")

        # Build search space description for LLM
        search_space_desc = format_search_space_description(
            objectives=objectives,
            config_schema=self.config_schema,
            example_config=self.example_config,
            constraints=self.constraints_description,
        )

        # Track all results
        all_valid_results: List[CandidateResult] = []
        all_failed_results: List[CandidateResult] = []
        all_candidates: List[Candidate[ConfigT]] = []
        history: List[Dict[str, Any]] = []

        # Accumulated instructions from learning
        constraint_instruction = ""
        performance_insights = ""

        # ===== Generation 1: Initial Sampling =====
        self._log(f"=== Generation 1 / {self.generations} (initial sampling) ===")

        self._report_search_event(reporter, {
            "type": "generation_start",
            "gen": 1, "total_gens": self.generations, "phase": "initial",
        })

        gen1_valid, gen1_failed, constraint_instruction = self._run_initial_generation(
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

        for r in gen1_valid:
            all_candidates.append(result_to_candidate(r, {"generation": 1, "is_pareto": False}))
        for r in gen1_failed:
            all_candidates.append(result_to_candidate(r, {"generation": 1, "is_pareto": False, "valid": False}))

        # Compute initial Pareto front
        pareto = compute_pareto_front(all_valid_results, objectives)
        self._log(f"Generation 1: {len(gen1_valid)} valid, Pareto size={len(pareto)}")

        # Generate initial performance insights
        if all_valid_results:
            performance_insights = self._generate_performance_insights(
                valid_results=all_valid_results,
                objectives=objectives,
                search_space_desc=search_space_desc,
            )

        pareto_front_summary = [
            r.objectives for r in pareto[:5]
        ]

        history.append({
            "gen": 1,
            "valid_count": len(gen1_valid),
            "failed_count": len(gen1_failed),
            "pareto_size": len(pareto),
        })
        if reporter:
            try:
                reporter("Search generation", 1)
                reporter("Search valid count", len(gen1_valid))
                reporter("Search Pareto size", len(pareto))
            except Exception:
                pass

        self._report_search_event(reporter, {
            "type": "generation_complete",
            "gen": 1,
            "valid_count": len(gen1_valid),
            "failed_count": len(gen1_failed),
            "pareto_size": len(pareto),
            "pareto_front": pareto_front_summary,
            "constraint_instruction": constraint_instruction[:500] if constraint_instruction else "",
            "performance_insights": performance_insights[:500] if performance_insights else "",
        })

        prev_pareto = pareto

        # ===== Generations 2..N: Evolution =====
        for gen in range(2, self.generations + 1):
            self._log(f"\n=== Generation {gen} / {self.generations} ===")

            self._report_search_event(reporter, {
                "type": "generation_start",
                "gen": gen, "total_gens": self.generations, "phase": "evolution",
            })

            gen_valid, gen_failed, constraint_instruction = self._run_evolution_generation(
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

            for r in gen_valid:
                all_candidates.append(result_to_candidate(r, {"generation": gen, "is_pareto": False}))
            for r in gen_failed:
                all_candidates.append(result_to_candidate(r, {"generation": gen, "is_pareto": False, "valid": False}))

            # Update Pareto front
            pareto = compute_pareto_front(all_valid_results, objectives)

            # Update performance insights
            if all_valid_results:
                performance_insights = self._update_performance_insights(
                    previous_insights=performance_insights,
                    valid_results=all_valid_results,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )

            self._log(f"Generation {gen}: {len(gen_valid)} valid, Pareto size={len(pareto)}")

            pareto_front_summary = [
                r.objectives for r in pareto[:5]
            ]

            history.append({
                "gen": gen,
                "valid_count": len(gen_valid),
                "failed_count": len(gen_failed),
                "pareto_size": len(pareto),
            })
            if reporter:
                try:
                    reporter("Search generation", gen)
                    reporter("Search valid count", len(gen_valid))
                    reporter("Search Pareto size", len(pareto))
                except Exception:
                    pass

            self._report_search_event(reporter, {
                "type": "generation_complete",
                "gen": gen,
                "valid_count": len(gen_valid),
                "failed_count": len(gen_failed),
                "pareto_size": len(pareto),
                "pareto_front": pareto_front_summary,
                "constraint_instruction": constraint_instruction[:500] if constraint_instruction else "",
                "performance_insights": performance_insights[:500] if performance_insights else "",
            })

            prev_pareto = pareto

        # ===== Build Final Results =====
        final_pareto = compute_pareto_front(all_valid_results, objectives)

        pareto_configs = {prettify_configuration(r.configuration) for r in final_pareto}
        pareto_candidates: List[Candidate[ConfigT]] = []
        for r in final_pareto:
            pareto_candidates.append(result_to_candidate(r, {"is_pareto": True}))

        for c in all_candidates:
            c_key = prettify_configuration(c.configuration)
            if c_key in pareto_configs:
                c.metadata["is_pareto"] = True

        best_result = select_best_candidate(final_pareto, objectives)
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

    def _run_initial_generation(
        self,
        problem: SearchProblem[ConfigT],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        all_failed_results: List[CandidateResult],
        constraint_instruction: str,
        performance_insights: str,
        reporter=None,
    ) -> Tuple[List[CandidateResult], List[CandidateResult], str]:
        """
        Run the initial generation with regeneration loops.

        Returns:
            (valid_results, failed_results, updated_constraint_instruction)
        """
        population_valid: List[CandidateResult] = []
        gen_failed: List[CandidateResult] = []
        last_round_failed: List[CandidateResult] = []

        regen_round = 0
        while len(population_valid) < self.pop_size and regen_round < self.max_regen_rounds:
            if regen_round == 0:
                candidates, reasoning = self._generate_initial_candidates(
                    n_candidates=self.candidates_per_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
            else:
                candidates, reasoning = self._regenerate_candidates(
                    failed_results=last_round_failed,
                    n_candidates=self.candidates_per_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                    constraint_instruction=constraint_instruction,
                    performance_insights=performance_insights,
                )

            self._report_search_event(reporter, {
                "type": "candidates_generated",
                "gen": 1, "count": len(candidates), "reasoning": reasoning,
            })

            valid_batch, failed_batch = self._evaluate_batch(
                problem, candidates, objectives,
                gen=1, batch_idx=regen_round, reporter=reporter,
            )

            self._log(f"  Batch {regen_round + 1}: {len(valid_batch)} valid, {len(failed_batch)} failed")

            if failed_batch:
                insights = self._generate_failure_insights(
                    failed_results=failed_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
                for r, insight in zip(failed_batch, insights):
                    r.insight = insight
                gen_failed.extend(failed_batch)

            population_valid.extend(valid_batch)
            last_round_failed = failed_batch

            # Always update constraint instruction when there are failures so the
            # LLM can learn from each round — especially critical when all candidates
            # fail, which is exactly when the instruction needs to improve most.
            if gen_failed:
                sampled_failures = sample_failed_for_constraint(
                    last_round_failed, gen_failed, self.max_failed_examples
                )
                if constraint_instruction:
                    constraint_instruction = self._update_constraint_instruction(
                        previous_instruction=constraint_instruction,
                        failed_results=sampled_failures,
                        objectives=objectives,
                        search_space_desc=search_space_desc,
                    )
                else:
                    constraint_instruction = self._generate_constraint_instruction(
                        failed_results=sampled_failures,
                        objectives=objectives,
                        search_space_desc=search_space_desc,
                    )

            regen_round += 1
            self._log(f"  Collected {len(population_valid)}/{self.pop_size} valid")

            if len(population_valid) >= self.pop_size:
                break

        population_valid = population_valid[:self.pop_size]

        return population_valid, gen_failed, constraint_instruction

    def _run_evolution_generation(
        self,
        problem: SearchProblem[ConfigT],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        prev_pareto: List[CandidateResult],
        all_failed_results: List[CandidateResult],
        constraint_instruction: str,
        performance_insights: str,
        gen: int = 0,
        reporter=None,
    ) -> Tuple[List[CandidateResult], List[CandidateResult], str]:
        """
        Run an evolution generation using Pareto front.

        Returns:
            (valid_results, failed_results, updated_constraint_instruction)
        """
        population_valid: List[CandidateResult] = []
        gen_failed: List[CandidateResult] = []
        last_round_failed: List[CandidateResult] = []

        if not prev_pareto:
            return self._run_initial_generation(
                problem, objectives, search_space_desc,
                all_failed_results, constraint_instruction, performance_insights,
                reporter=reporter,
            )

        candidates, reasoning = self._generate_offspring(
            pareto_results=prev_pareto,
            n_candidates=self.pop_size,
            objectives=objectives,
            search_space_desc=search_space_desc,
            constraint_instruction=constraint_instruction,
            performance_insights=performance_insights,
        )

        self._report_search_event(reporter, {
            "type": "candidates_generated",
            "gen": gen, "count": len(candidates), "reasoning": reasoning,
        })

        valid_batch, failed_batch = self._evaluate_batch(
            problem, candidates, objectives,
            gen=gen, batch_idx=0, reporter=reporter,
        )

        self._log(f"  Offspring: {len(valid_batch)} valid, {len(failed_batch)} failed")

        if failed_batch:
            insights = self._generate_failure_insights(
                failed_results=failed_batch,
                objectives=objectives,
                search_space_desc=search_space_desc,
            )
            for r, insight in zip(failed_batch, insights):
                r.insight = insight
            gen_failed.extend(failed_batch)

        population_valid.extend(valid_batch)
        last_round_failed = failed_batch

        regen_round = 0
        while len(population_valid) < self.pop_size and regen_round < self.max_regen_rounds:
            if not last_round_failed:
                break

            candidates, reasoning = self._regenerate_offspring(
                failed_results=last_round_failed,
                pareto_results=prev_pareto,
                n_candidates=self.candidates_per_batch,
                objectives=objectives,
                search_space_desc=search_space_desc,
                constraint_instruction=constraint_instruction,
                performance_insights=performance_insights,
            )

            self._report_search_event(reporter, {
                "type": "candidates_generated",
                "gen": gen, "count": len(candidates), "reasoning": reasoning,
            })

            valid_batch, failed_batch = self._evaluate_batch(
                problem, candidates, objectives,
                gen=gen, batch_idx=regen_round + 1, reporter=reporter,
            )

            self._log(f"  Regen {regen_round + 1}: {len(valid_batch)} valid, {len(failed_batch)} failed")

            if failed_batch:
                insights = self._generate_failure_insights(
                    failed_results=failed_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
                for r, insight in zip(failed_batch, insights):
                    r.insight = insight
                gen_failed.extend(failed_batch)

                sampled_failures = sample_failed_for_constraint(
                    failed_batch, gen_failed, self.max_failed_examples
                )
                constraint_instruction = self._update_constraint_instruction(
                    previous_instruction=constraint_instruction,
                    failed_results=sampled_failures,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )

            population_valid.extend(valid_batch)
            last_round_failed = failed_batch
            regen_round += 1

            self._log(f"  Collected {len(population_valid)}/{self.pop_size} valid")

        population_valid = population_valid[:self.pop_size]

        return population_valid, gen_failed, constraint_instruction

    @staticmethod
    def _report_search_event(reporter, event: Dict[str, Any]) -> None:
        """Emit a structured search event via the reporter."""
        if reporter is None:
            return
        try:
            reporter("search_event", json.dumps(event, default=str))
        except Exception:
            pass

    def _evaluate_batch(
        self,
        problem: SearchProblem[ConfigT],
        candidates: List[ConfigT],
        objectives: Sequence[ObjectiveSpec],
        gen: int = 0,
        batch_idx: int = 0,
        reporter=None,
    ) -> Tuple[List[CandidateResult], List[CandidateResult]]:
        """
        Evaluate a batch of candidates.

        Uses ``validate_detailed()`` when available to get rich error
        information (failure phase + message) for the constraint-learning loop.

        Returns:
            (valid_results, failed_results)
        """
        valid_results: List[CandidateResult] = []
        failed_results: List[CandidateResult] = []
        has_detailed = hasattr(problem, "validate_detailed")
        penalty_obj = {s.name: (0.0 if s.goal == "max" else self.invalid_penalty) for s in objectives}

        for idx, config in enumerate(candidates):
            try:
                if self.verbose:
                    self._log(f"    Candidate {idx+1}: {prettify_configuration(config)[:200]}...")

                if has_detailed:
                    vr = problem.validate_detailed(config)
                    is_valid = vr.is_valid
                    error_msg = vr.error_message
                    failure_phase = vr.failure_phase
                else:
                    is_valid = problem.validate(config)
                    error_msg = None
                    failure_phase = None

                if not is_valid:
                    if error_msg is None:
                        if not config:
                            error_msg = "Empty configuration — no keys provided"
                        else:
                            error_msg = f"Validation failed; provided keys: {list(config.keys())}"
                    if self.verbose:
                        phase_str = f" [{failure_phase}]" if failure_phase else ""
                        self._log(f"    -> FAILED{phase_str}: {error_msg}")
                    cr = CandidateResult(
                        configuration=config,
                        objectives=dict(penalty_obj),
                        is_valid=False,
                        error_message=error_msg,
                        failure_phase=failure_phase,
                    )
                    failed_results.append(cr)
                    self._report_search_event(reporter, {
                        "type": "candidate_result",
                        "gen": gen, "idx": idx,
                        "config_summary": str(config)[:200],
                        "is_valid": False,
                        "objectives": dict(penalty_obj),
                        "error_message": error_msg,
                        "failure_phase": failure_phase,
                    })
                    continue

                obj = problem.evaluate(config)
                if self.verbose:
                    self._log(f"    -> VALID: {obj}")
                valid_results.append(CandidateResult(
                    configuration=config,
                    objectives=obj,
                    is_valid=True,
                ))
                self._report_search_event(reporter, {
                    "type": "candidate_result",
                    "gen": gen, "idx": idx,
                    "config_summary": str(config)[:200],
                    "is_valid": True,
                    "objectives": obj,
                    "error_message": None,
                    "failure_phase": None,
                })
            except Exception as e:
                if self.verbose:
                    self._log(f"    -> EXCEPTION: {e}")
                failed_results.append(CandidateResult(
                    configuration=config,
                    objectives=dict(penalty_obj),
                    is_valid=False,
                    error_message=str(e),
                ))
                self._report_search_event(reporter, {
                    "type": "candidate_result",
                    "gen": gen, "idx": idx,
                    "config_summary": str(config)[:200],
                    "is_valid": False,
                    "objectives": dict(penalty_obj),
                    "error_message": str(e),
                    "failure_phase": "exception",
                })

        self._report_search_event(reporter, {
            "type": "batch_summary",
            "gen": gen, "batch_idx": batch_idx,
            "valid_count": len(valid_results),
            "failed_count": len(failed_results),
        })

        return valid_results, failed_results

    # ===== LLM Interaction Methods =====

    def _generate_initial_candidates(
        self,
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> Tuple[List[ConfigT], str]:
        """Generate initial candidate configurations using LLM.

        Returns (candidates, reasoning).
        """
        template = build_initial_candidates_prompt(n_candidates, search_space_desc)
        result = self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
        )
        candidates = getattr(result, "candidates", [])
        reasoning = getattr(result, "reasoning", "")
        if self.verbose:
            self._log(f"  LLM generated {len(candidates)} candidates")
            if reasoning:
                self._log(f"  Reasoning: {reasoning[:300]}...")
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    def _regenerate_candidates(
        self,
        failed_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[ConfigT], str]:
        """Regenerate candidates using failure insights.

        Returns (candidates, reasoning).
        """
        failed_str = prettify_results(failed_results, objectives)
        template = build_regenerate_candidates_prompt(
            failed_str, search_space_desc, constraint_instruction, performance_insights, n_candidates
        )
        result = self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
        )
        candidates = getattr(result, "candidates", [])
        reasoning = getattr(result, "reasoning", "")
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    def _generate_offspring(
        self,
        pareto_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[ConfigT], str]:
        """Generate offspring from Pareto front configurations.

        Returns (candidates, reasoning).
        """
        pareto_str = prettify_results(pareto_results[:5], objectives)
        template = build_offspring_prompt(
            pareto_str, search_space_desc, constraint_instruction, performance_insights, n_candidates
        )
        result = self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
        )
        candidates = getattr(result, "candidates", [])
        reasoning = getattr(result, "reasoning", "")
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    def _regenerate_offspring(
        self,
        failed_results: List[CandidateResult],
        pareto_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[ConfigT], str]:
        """Regenerate offspring using both Pareto front and failure insights.

        Returns (candidates, reasoning).
        """
        failed_str = prettify_results(failed_results, objectives)
        pareto_str = prettify_results(pareto_results[:3], objectives)
        template = build_regenerate_offspring_prompt(
            failed_str, pareto_str, search_space_desc, constraint_instruction, performance_insights, n_candidates
        )
        result = self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
        )
        candidates = getattr(result, "candidates", [])
        reasoning = getattr(result, "reasoning", "")
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    def _generate_failure_insights(
        self,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> List[str]:
        """Generate insights for why candidates failed."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_failure_insights_prompt(failed_str, search_space_desc, len(failed_results))
        result = self._llm_call(
            template=template,
            output_schema={"insights": List[str]},
        )
        insights = getattr(result, "insights", [])
        while len(insights) < len(failed_results):
            insights.append("Unknown failure reason")
        return insights[:len(failed_results)]

    def _generate_constraint_instruction(
        self,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Generate consolidated constraint instructions from failures."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_constraint_instruction_prompt(failed_str, search_space_desc)
        result = self._llm_call(
            template=template,
            output_schema={"constraint_instruction": str},
        )
        return getattr(result, "constraint_instruction", "")

    def _update_constraint_instruction(
        self,
        previous_instruction: str,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Update constraint instructions with new failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_update_constraint_prompt(previous_instruction, failed_str, search_space_desc)
        result = self._llm_call(
            template=template,
            output_schema={"updated_instruction": str},
        )
        return getattr(result, "updated_instruction", previous_instruction)

    def _generate_performance_insights(
        self,
        valid_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Generate performance insights from valid results."""
        stats = compute_performance_stats(valid_results, objectives)
        if not stats:
            return ""
        stats_lines = []
        for spec in objectives:
            best = stats.get(f'best_{spec.name}')
            worst = stats.get(f'worst_{spec.name}')
            if best:
                stats_lines.append(f"Best {spec.name}: {best.objectives.get(spec.name, 'N/A')}")
                stats_lines.append(f"  Config: {prettify_configuration(best.configuration)}")
            if worst:
                stats_lines.append(f"Worst {spec.name}: {worst.objectives.get(spec.name, 'N/A')}")
        top_pareto = stats.get('top_3_pareto', [])
        if top_pareto:
            stats_lines.append("\nTop Pareto configurations:")
            stats_lines.append(prettify_results(top_pareto, objectives))
        stats_str = "\n".join(stats_lines)
        template = build_performance_insights_prompt(stats_str, search_space_desc)
        result = self._llm_call(
            template=template,
            output_schema={"performance_insights": str},
        )
        return getattr(result, "performance_insights", "")

    def _update_performance_insights(
        self,
        previous_insights: str,
        valid_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Update performance insights with new results."""
        stats = compute_performance_stats(valid_results, objectives)
        if not stats:
            return previous_insights
        top_pareto = stats.get('top_3_pareto', [])
        pareto_str = prettify_results(top_pareto, objectives) if top_pareto else "None"
        template = build_update_performance_insights_prompt(
            previous_insights,
            pareto_str,
            len(valid_results),
            stats.get('pareto_size', 0),
            search_space_desc,
        )
        result = self._llm_call(
            template=template,
            output_schema={"updated_insights": str},
        )
        return getattr(result, "updated_insights", previous_insights)
