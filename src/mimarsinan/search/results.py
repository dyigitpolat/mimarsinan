from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Tuple, TypeVar


Goal = Literal["min", "max"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    goal: Goal


# ---------------------------------------------------------------------------
# Canonical objective catalogue
# ---------------------------------------------------------------------------

ALL_OBJECTIVES: Tuple[ObjectiveSpec, ...] = (
    ObjectiveSpec("estimated_accuracy", "max"),
    ObjectiveSpec("total_params", "min"),
    ObjectiveSpec("total_param_capacity", "min"),
    ObjectiveSpec("total_sync_barriers", "min"),
    ObjectiveSpec("param_utilization_pct", "max"),
    ObjectiveSpec("neuron_wastage_pct", "min"),
    ObjectiveSpec("axon_wastage_pct", "min"),
)

ACCURACY_OBJECTIVE_NAME = "estimated_accuracy"

_OBJECTIVES_BY_NAME: Dict[str, ObjectiveSpec] = {o.name: o for o in ALL_OBJECTIVES}


def objectives_for_mode(search_mode: str) -> Tuple[ObjectiveSpec, ...]:
    """All objectives *available* for a given search mode.

    Accuracy is excluded for hardware-only search (no training is performed).
    """
    if search_mode == "hardware":
        return tuple(o for o in ALL_OBJECTIVES if o.name != ACCURACY_OBJECTIVE_NAME)
    return ALL_OBJECTIVES


def default_objectives_for_mode(search_mode: str) -> Tuple[str, ...]:
    """Default active objective *names* when the user does not specify."""
    if search_mode == "hardware":
        return ("total_param_capacity", "param_utilization_pct", "neuron_wastage_pct", "axon_wastage_pct")
    if search_mode == "model":
        return ("estimated_accuracy", "total_params")
    # joint
    return ("estimated_accuracy", "total_params", "param_utilization_pct", "neuron_wastage_pct")


def resolve_active_objectives(
    search_mode: str,
    user_selection: Optional[Sequence[str]] = None,
) -> Tuple[ObjectiveSpec, ...]:
    """Resolve user selection (or defaults) into validated ObjectiveSpec tuple."""
    available = {o.name for o in objectives_for_mode(search_mode)}
    names = tuple(user_selection) if user_selection else default_objectives_for_mode(search_mode)
    resolved = []
    for n in names:
        if n in available and n in _OBJECTIVES_BY_NAME:
            resolved.append(_OBJECTIVES_BY_NAME[n])
    if not resolved:
        # Fallback: use all defaults for the mode
        for n in default_objectives_for_mode(search_mode):
            resolved.append(_OBJECTIVES_BY_NAME[n])
    return tuple(resolved)


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


