"""Pareto analysis, candidate conversion, and selection helpers for AgentEvolve."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mimarsinan.search.results import (
    Candidate,
    ObjectiveSpec,
    _rank_candidates,
    select_minimax_rank,
)

from .schema import CandidateResult, prettify_configuration


def dominates(
    a: Dict[str, float],
    b: Dict[str, float],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    """Check if ``a`` Pareto-dominates ``b``."""
    dominated_in_all = True
    better_in_one = False

    for spec in objectives:
        val_a = a.get(spec.name, 0.0)
        val_b = b.get(spec.name, 0.0)

        if spec.goal == "max":
            if val_a < val_b:
                dominated_in_all = False
            elif val_a > val_b:
                better_in_one = True
        else:
            if val_a > val_b:
                dominated_in_all = False
            elif val_a < val_b:
                better_in_one = True

    return dominated_in_all and better_in_one


def compute_pareto_front(
    results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> List[CandidateResult]:
    """Compute the Pareto front from a list of CandidateResults."""
    valid = [r for r in results if r.is_valid]
    if not valid:
        return []

    pareto: List[CandidateResult] = []
    for i, candidate in enumerate(valid):
        dominated = False
        for j, other in enumerate(valid):
            if i == j:
                continue
            if dominates(other.objectives, candidate.objectives, objectives):
                dominated = True
                break
        if not dominated:
            pareto.append(candidate)
    return pareto


def compute_performance_stats(
    valid_results: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Dict[str, Any]]:
    """Compute comprehensive performance statistics for all valid results."""
    if not valid_results:
        return None

    stats: Dict[str, Any] = {}

    for spec in objectives:
        key = spec.name
        if spec.goal == "max":
            best = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))
            worst = min(valid_results, key=lambda r: r.objectives.get(key, 0.0))
        else:
            best = min(valid_results, key=lambda r: r.objectives.get(key, float("inf")))
            worst = max(valid_results, key=lambda r: r.objectives.get(key, 0.0))

        stats[f"best_{key}"] = best
        stats[f"worst_{key}"] = worst

    pareto_front = compute_pareto_front(valid_results, objectives)

    dominated_candidates: List[Tuple[CandidateResult, int]] = []
    for i, candidate in enumerate(valid_results):
        if candidate in pareto_front:
            continue
        domination_count = sum(
            1 for other in valid_results
            if dominates(other.objectives, candidate.objectives, objectives)
        )
        dominated_candidates.append((candidate, domination_count))

    pareto_sorted = sort_pareto_results_minimax_first(pareto_front, objectives)
    stats["top_3_pareto"] = pareto_sorted[:3]

    dominated_sorted = sorted(dominated_candidates, key=lambda x: -x[1])
    stats["bottom_3_dominated"] = [c for c, _ in dominated_sorted[:3]]

    stats["pareto_front"] = pareto_front
    stats["pareto_size"] = len(pareto_front)

    return stats


def sample_failed_for_constraint(
    latest_failed: List[CandidateResult],
    all_previous_failed: List[CandidateResult],
    max_examples: int,
) -> List[CandidateResult]:
    """Sample failed examples for constraint instruction generation."""
    sampled = list(latest_failed)

    if len(sampled) >= max_examples:
        return sampled[:max_examples]

    remaining_slots = max_examples - len(sampled)

    latest_ids = {id(r) for r in latest_failed}
    previous = [r for r in all_previous_failed if id(r) not in latest_ids]

    if previous and remaining_slots > 0:
        sample_size = min(remaining_slots, len(previous))
        sampled.extend(random.sample(previous, sample_size))

    return sampled


def candidate_to_result(candidate: Candidate[Dict[str, Any]]) -> CandidateResult:
    """Convert a Candidate to CandidateResult."""
    return CandidateResult(
        configuration=candidate.configuration,
        objectives=candidate.objectives,
        is_valid=True,
        error_message=None,
        insight="",
    )


def result_to_candidate(
    result: CandidateResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> Candidate[Dict[str, Any]]:
    """Convert a CandidateResult to Candidate."""
    return Candidate(
        configuration=result.configuration,
        objectives=result.objectives,
        metadata=metadata or {"is_pareto": False},
    )


def sort_pareto_results_minimax_first(
    pareto: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> List[CandidateResult]:
    """Order Pareto members by minimax rank (best-balanced first)."""
    if len(pareto) <= 1:
        return list(pareto)
    cands = [result_to_candidate(r) for r in pareto]
    ranks = _rank_candidates(cands, objectives)
    worst = [max(row) for row in ranks]
    rank_sums = [sum(row) for row in ranks]
    order = sorted(range(len(pareto)), key=lambda i: (worst[i], rank_sums[i]))
    return [pareto[i] for i in order]


def select_best_candidate_minimax(
    pareto: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[CandidateResult]:
    """Pick the reported best from the Pareto front using minimax-rank selection."""
    if not pareto:
        return None
    cands = [result_to_candidate(r) for r in pareto]
    best_cand = select_minimax_rank(cands, objectives)
    if best_cand is None:
        return None
    key = prettify_configuration(best_cand.configuration)
    for r in pareto:
        if prettify_configuration(r.configuration) == key:
            return r
    return pareto[0]


def select_best_candidate(
    pareto: List[CandidateResult],
    objectives: Sequence[ObjectiveSpec],
    priority_order: Optional[List[str]] = None,
) -> Optional[CandidateResult]:
    """Select the best candidate from the Pareto front via lexicographic ordering."""
    if not pareto:
        return None

    if priority_order is None:
        max_objs = [s for s in objectives if s.goal == "max"]
        min_objs = [s for s in objectives if s.goal == "min"]
        priority_order = [s.name for s in max_objs] + [s.name for s in min_objs]

    obj_map = {s.name: s for s in objectives}

    def sort_key(candidate: CandidateResult) -> Tuple:
        key = []
        for name in priority_order:
            spec = obj_map.get(name)
            val = candidate.objectives.get(name, 0.0)
            if spec and spec.goal == "max":
                key.append(-val)
            else:
                key.append(val)
        return tuple(key)

    return min(pareto, key=sort_key)


__all__ = [
    "candidate_to_result",
    "compute_pareto_front",
    "compute_performance_stats",
    "dominates",
    "result_to_candidate",
    "sample_failed_for_constraint",
    "select_best_candidate",
    "select_best_candidate_minimax",
    "sort_pareto_results_minimax_first",
]
