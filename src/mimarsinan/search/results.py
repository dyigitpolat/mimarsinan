from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, TypeVar


Goal = Literal["min", "max"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    goal: Goal


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class Candidate(Generic[ConfigT]):
    configuration: ConfigT
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult(Generic[ConfigT]):
    """
    Generic result container for any search backend.

    - For single-objective search, pareto_front may be empty and best is defined.
    - For multi-objective search, pareto_front contains the nondominated set and best is the
      selected tradeoff point.
    - all_candidates contains every evaluated candidate across all generations.
    """

    objectives: Sequence[ObjectiveSpec]
    best: Candidate[ConfigT]
    pareto_front: List[Candidate[ConfigT]] = field(default_factory=list)
    all_candidates: List[Candidate[ConfigT]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimax-rank selection
# ---------------------------------------------------------------------------

def _rank_candidates(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[List[int]]:
    """Return ``ranks[i][j]`` — the 1-based rank of candidate *i* on objective *j*.

    Ties receive the same rank (dense ranking).  Rank 1 is *best* for the
    objective's goal direction.
    """
    n = len(candidates)
    ranks: List[List[int]] = [[0] * len(objectives) for _ in range(n)]

    for j, spec in enumerate(objectives):
        values = [float(c.objectives.get(spec.name, 0.0)) for c in candidates]
        reverse = spec.goal == "max"
        order = sorted(range(n), key=lambda i: values[i], reverse=reverse)

        current_rank = 1
        for pos, idx in enumerate(order):
            if pos > 0 and values[order[pos]] != values[order[pos - 1]]:
                current_rank = pos + 1
            ranks[idx][j] = current_rank

    return ranks


def select_minimax_rank(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> Optional[Candidate[ConfigT]]:
    """Pick the best-balanced candidate from a Pareto front.

    Algorithm (increasing *N* until a candidate qualifies):
      N = 1 → is any candidate ranked #1 on **every** objective?
      N = 2 → is any candidate in the top 2 on every objective?
      …
      Stop at the smallest *N* that yields at least one qualifying candidate.

    This is equivalent to selecting the candidate whose *worst rank across
    all objectives* is minimal (minimax-rank).  Ties are broken by the sum
    of ranks (prefer the most uniformly strong candidate).

    Returns ``None`` when *candidates* is empty.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ranks = _rank_candidates(candidates, objectives)

    worst_ranks = [max(r) for r in ranks]
    min_worst = min(worst_ranks)

    tied = [i for i in range(len(candidates)) if worst_ranks[i] == min_worst]

    if len(tied) == 1:
        return candidates[tied[0]]

    # Tiebreaker: smallest sum of ranks (most uniformly strong).
    best_idx = min(tied, key=lambda i: sum(ranks[i]))
    return candidates[best_idx]


