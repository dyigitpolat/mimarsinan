"""Candidate evaluation and per-generation batch orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from mimarsinan.search.optimizers.agent_evolve.schema import (
    CandidateResult,
    prettify_configuration,
)
from mimarsinan.search.optimizers.llm.trace import emit_search_event
from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import ObjectiveSpec


class BatchEvalMixin:
    """Evaluate candidate batches and run generation regen loops."""

    @staticmethod
    def _report_search_event(reporter, event: Dict[str, Any]) -> None:
        emit_search_event(reporter, event)

    def _evaluate_batch(
        self,
        problem: SearchProblem[Any],
        candidates: List[Any],
        objectives: Sequence[ObjectiveSpec],
        gen: int = 0,
        batch_idx: int = 0,
        reporter=None,
    ) -> Tuple[List[CandidateResult], List[CandidateResult]]:
        """Evaluate a batch of candidates; return (valid_results, failed_results)."""
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

    async def _run_initial_generation(
        self,
        problem: SearchProblem[Any],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        all_failed_results: List[CandidateResult],
        constraint_instruction: str,
        performance_insights: str,
        reporter=None,
    ) -> Tuple[List[CandidateResult], List[CandidateResult], str]:
        """Run the initial generation with regeneration loops."""
        population_valid: List[CandidateResult] = []
        gen_failed: List[CandidateResult] = []
        last_round_failed: List[CandidateResult] = []

        regen_round = 0
        while len(population_valid) < self.pop_size and regen_round < self.max_regen_rounds:
            if regen_round == 0:
                candidates, reasoning = await self._generate_initial_candidates(
                    n_candidates=self.candidates_per_batch,
                    objectives=objectives,
                    search_space_desc=search_space_desc,
                )
            else:
                candidates, reasoning = await self._regenerate_candidates(
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

            await self._apply_failure_insights(
                failed_batch, gen_failed, objectives, search_space_desc
            )

            population_valid.extend(valid_batch)
            last_round_failed = failed_batch

            if gen_failed:
                constraint_instruction = await self._refresh_constraint_instruction(
                    constraint_instruction, last_round_failed, gen_failed,
                    objectives, search_space_desc,
                )

            regen_round += 1
            self._log(f"  Collected {len(population_valid)}/{self.pop_size} valid")

            if len(population_valid) >= self.pop_size:
                break

        return population_valid[: self.pop_size], gen_failed, constraint_instruction

    async def _run_evolution_generation(
        self,
        problem: SearchProblem[Any],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        prev_pareto: List[CandidateResult],
        all_failed_results: List[CandidateResult],
        constraint_instruction: str,
        performance_insights: str,
        gen: int = 0,
        reporter=None,
    ) -> Tuple[List[CandidateResult], List[CandidateResult], str]:
        """Run an evolution generation using Pareto front."""
        if not prev_pareto:
            return await self._run_initial_generation(
                problem, objectives, search_space_desc,
                all_failed_results, constraint_instruction, performance_insights,
                reporter=reporter,
            )

        population_valid: List[CandidateResult] = []
        gen_failed: List[CandidateResult] = []
        last_round_failed: List[CandidateResult] = []

        candidates, reasoning = await self._generate_offspring(
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

        await self._apply_failure_insights(
            failed_batch, gen_failed, objectives, search_space_desc
        )

        population_valid.extend(valid_batch)
        last_round_failed = failed_batch

        regen_round = 0
        while len(population_valid) < self.pop_size and regen_round < self.max_regen_rounds:
            if not last_round_failed:
                break

            candidates, reasoning = await self._regenerate_offspring(
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

            await self._apply_failure_insights(
                failed_batch, gen_failed, objectives, search_space_desc
            )

            if failed_batch:
                constraint_instruction = await self._refresh_constraint_instruction(
                    constraint_instruction, last_round_failed, gen_failed,
                    objectives, search_space_desc, always_update=True,
                )

            population_valid.extend(valid_batch)
            last_round_failed = failed_batch
            regen_round += 1

            self._log(f"  Collected {len(population_valid)}/{self.pop_size} valid")

        return population_valid[: self.pop_size], gen_failed, constraint_instruction
