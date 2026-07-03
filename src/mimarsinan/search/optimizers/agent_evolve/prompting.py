"""LLM prompt orchestration for AgentEvolveOptimizer."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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
from mimarsinan.search.optimizers.agent_evolve.codec import (
    compute_performance_stats,
    sample_failed_for_constraint,
)
from mimarsinan.search.optimizers.agent_evolve.host_contract import EvolveHostContract
from mimarsinan.search.optimizers.agent_evolve.schema import (
    CandidateResult,
    format_performance_stats,
    prettify_results,
)
from mimarsinan.search.results import ObjectiveSpec


class PromptingMixin(EvolveHostContract):
    """High-level LLM interactions (candidates, insights, constraints)."""

    async def _generate_initial_candidates(
        self,
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Generate initial candidate configurations using LLM."""
        template = build_initial_candidates_prompt(n_candidates, search_space_desc)
        result = await self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
            call_kind="initial_candidates",
        )
        candidates = getattr(result, "candidates", [])
        reasoning = self._coerce_llm_text(getattr(result, "reasoning", ""))
        if self.verbose:
            self._log(f"  LLM generated {len(candidates)} candidates")
            if reasoning:
                self._log(f"  Reasoning: {reasoning[:300]}...")
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    async def _regenerate_candidates(
        self,
        failed_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Regenerate candidates using failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_regenerate_candidates_prompt(
            failed_str, search_space_desc, constraint_instruction, performance_insights, n_candidates
        )
        result = await self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
            call_kind="regenerate_candidates",
        )
        candidates = getattr(result, "candidates", [])
        reasoning = self._coerce_llm_text(getattr(result, "reasoning", ""))
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    async def _generate_offspring(
        self,
        pareto_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Generate offspring from Pareto front configurations."""
        pareto_str = prettify_results(pareto_results[:5], objectives)
        template = build_offspring_prompt(
            pareto_str, search_space_desc, constraint_instruction, performance_insights, n_candidates
        )
        result = await self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
            call_kind="offspring",
        )
        candidates = getattr(result, "candidates", [])
        reasoning = self._coerce_llm_text(getattr(result, "reasoning", ""))
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    async def _regenerate_offspring(
        self,
        failed_results: List[CandidateResult],
        pareto_results: List[CandidateResult],
        n_candidates: int,
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        constraint_instruction: str,
        performance_insights: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Regenerate offspring using both Pareto front and failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        pareto_str = prettify_results(pareto_results[:3], objectives)
        template = build_regenerate_offspring_prompt(
            failed_str, pareto_str, search_space_desc, constraint_instruction, performance_insights, n_candidates
        )
        result = await self._llm_call(
            template=template,
            output_schema={"reasoning": str, "candidates": List[Dict[str, Any]]},
            call_kind="regenerate_offspring",
        )
        candidates = getattr(result, "candidates", [])
        reasoning = self._coerce_llm_text(getattr(result, "reasoning", ""))
        return parse_candidates(candidates, n_candidates, self._log), reasoning

    async def _generate_failure_insights(
        self,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> List[str]:
        """Generate insights for why candidates failed."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_failure_insights_prompt(failed_str, search_space_desc, len(failed_results))
        result = await self._llm_call(
            template=template,
            output_schema={"insights": List[str]},
            call_kind="failure_insights",
        )
        insights = getattr(result, "insights", [])
        while len(insights) < len(failed_results):
            insights.append("Unknown failure reason")
        return insights[:len(failed_results)]

    async def _generate_constraint_instruction(
        self,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Generate consolidated constraint instructions from failures."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_constraint_instruction_prompt(failed_str, search_space_desc)
        result = await self._llm_call(
            template=template,
            output_schema={"constraint_instruction": str},
            call_kind="constraint_instruction",
        )
        return getattr(result, "constraint_instruction", "")

    async def _update_constraint_instruction(
        self,
        previous_instruction: str,
        failed_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Update constraint instructions with new failure insights."""
        failed_str = prettify_results(failed_results, objectives)
        template = build_update_constraint_prompt(previous_instruction, failed_str, search_space_desc)
        result = await self._llm_call(
            template=template,
            output_schema={"updated_instruction": str},
            call_kind="update_constraint",
        )
        return getattr(result, "updated_instruction", previous_instruction)

    async def _generate_performance_insights(
        self,
        valid_results: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> str:
        """Generate performance insights from valid results."""
        stats = compute_performance_stats(valid_results, objectives)
        if not stats:
            return ""
        stats_str = format_performance_stats(stats, objectives, len(valid_results))
        template = build_performance_insights_prompt(stats_str, search_space_desc)
        result = await self._llm_call(
            template=template,
            output_schema={"performance_insights": str},
            call_kind="performance_insights",
        )
        return getattr(result, "performance_insights", "")

    async def _update_performance_insights(
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
        stats_str = format_performance_stats(stats, objectives, len(valid_results))
        template = build_update_performance_insights_prompt(
            previous_insights,
            stats_str,
            search_space_desc,
        )
        result = await self._llm_call(
            template=template,
            output_schema={"updated_insights": str},
            call_kind="update_performance_insights",
        )
        return getattr(result, "updated_insights", previous_insights)

    async def _apply_failure_insights(
        self,
        failed_batch: List[CandidateResult],
        gen_failed: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
    ) -> None:
        """Attach LLM failure insights to failed candidates and extend gen_failed."""
        if not failed_batch:
            return
        insights = await self._generate_failure_insights(
            failed_results=failed_batch,
            objectives=objectives,
            search_space_desc=search_space_desc,
        )
        for r, insight in zip(failed_batch, insights):
            r.insight = insight
        gen_failed.extend(failed_batch)

    async def _refresh_constraint_instruction(
        self,
        constraint_instruction: str,
        last_round_failed: List[CandidateResult],
        gen_failed: List[CandidateResult],
        objectives: Sequence[ObjectiveSpec],
        search_space_desc: str,
        *,
        always_update: bool = False,
    ) -> str:
        """Update constraint instruction from sampled failures."""
        if not gen_failed and not always_update:
            return constraint_instruction
        sampled_failures = sample_failed_for_constraint(
            last_round_failed, gen_failed, self.max_failed_examples
        )
        if constraint_instruction:
            return await self._update_constraint_instruction(
                previous_instruction=constraint_instruction,
                failed_results=sampled_failures,
                objectives=objectives,
                search_space_desc=search_space_desc,
            )
        return await self._generate_constraint_instruction(
            failed_results=sampled_failures,
            objectives=objectives,
            search_space_desc=search_space_desc,
        )
