"""Annotation-only contract the AgentEvolve mixins require from their composed host."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

from mimarsinan.search.optimizers.agent_evolve.schema import CandidateResult
from mimarsinan.search.results import ObjectiveSpec


class EvolveHostContract:
    """Declares host members used across the AgentEvolve mixins; empty at runtime."""

    pop_size: int
    candidates_per_batch: int
    max_regen_rounds: int
    max_failed_examples: int
    verbose: bool
    invalid_penalty: float

    if TYPE_CHECKING:

        def _log(self, message: str) -> None: ...

        @staticmethod
        def _report_search_event(reporter: Any, event: Dict[str, Any]) -> None: ...

        async def _llm_call(
            self,
            template: str,
            output_schema: Dict[str, type],
            call_kind: str = "unknown",
        ) -> Any: ...

        @staticmethod
        def _coerce_llm_text(val: Any) -> str: ...

        async def _generate_initial_candidates(
            self,
            n_candidates: int,
            objectives: Sequence[ObjectiveSpec],
            search_space_desc: str,
        ) -> Tuple[List[Dict[str, Any]], str]: ...

        async def _regenerate_candidates(
            self,
            failed_results: List[CandidateResult],
            n_candidates: int,
            objectives: Sequence[ObjectiveSpec],
            search_space_desc: str,
            constraint_instruction: str,
            performance_insights: str,
        ) -> Tuple[List[Dict[str, Any]], str]: ...

        async def _generate_offspring(
            self,
            pareto_results: List[CandidateResult],
            n_candidates: int,
            objectives: Sequence[ObjectiveSpec],
            search_space_desc: str,
            constraint_instruction: str,
            performance_insights: str,
        ) -> Tuple[List[Dict[str, Any]], str]: ...

        async def _regenerate_offspring(
            self,
            failed_results: List[CandidateResult],
            pareto_results: List[CandidateResult],
            n_candidates: int,
            objectives: Sequence[ObjectiveSpec],
            search_space_desc: str,
            constraint_instruction: str,
            performance_insights: str,
        ) -> Tuple[List[Dict[str, Any]], str]: ...

        async def _apply_failure_insights(
            self,
            failed_batch: List[CandidateResult],
            gen_failed: List[CandidateResult],
            objectives: Sequence[ObjectiveSpec],
            search_space_desc: str,
        ) -> None: ...

        async def _refresh_constraint_instruction(
            self,
            constraint_instruction: str,
            last_round_failed: List[CandidateResult],
            gen_failed: List[CandidateResult],
            objectives: Sequence[ObjectiveSpec],
            search_space_desc: str,
            *,
            always_update: bool = False,
        ) -> str: ...
