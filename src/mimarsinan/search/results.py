"""Search result containers, and the legacy projection of the objectives registry.

The objective catalogue itself lives in ``deployment_record.objectives`` — one
registry over the deployment artifact, asked which axes a view can carry. This
module PROJECTS the searchable part of it onto the frozen ``ObjectiveSpec``
tuple every optimizer reads (``spec.name``/``spec.goal``), and keeps the
per-search-mode DEFAULTS, which are a search policy rather than a record fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, Tuple, TypeVar

from mimarsinan.deployment_record.objectives import (
    ACCURACY_OBJECTIVE_KEY,
    OBJECTIVES,
    ObjectiveSpecV2,
)

Goal = Literal["min", "max"]


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    goal: Goal


def _project(spec: ObjectiveSpecV2) -> ObjectiveSpec:
    """A registry axis as the legacy optimizer-facing pair."""
    return ObjectiveSpec(spec.name, spec.goal)


ALL_OBJECTIVES: Tuple[ObjectiveSpec, ...] = tuple(
    _project(spec) for spec in OBJECTIVES.search_catalog()
)

ACCURACY_OBJECTIVE_NAME = ACCURACY_OBJECTIVE_KEY


def objectives_for_mode(search_mode: str) -> Tuple[ObjectiveSpec, ...]:
    """All objectives *available* for a given search mode (registry availability)."""
    return tuple(_project(spec) for spec in OBJECTIVES.for_search_mode(search_mode))


def default_objectives_for_mode(search_mode: str) -> Tuple[str, ...]:
    """Default active objective *names* when the user does not specify."""
    if search_mode == "hardware":
        return (
            "total_param_capacity",
            "param_utilization_pct",
            "neuron_wastage_pct",
            "axon_wastage_pct",
            "fragmentation_pct",
        )
    if search_mode == "model":
        return ("estimated_accuracy", "total_params")
    return (
        "estimated_accuracy",
        "total_params",
        "param_utilization_pct",
        "neuron_wastage_pct",
        "fragmentation_pct",
    )


def resolve_active_objectives(
    search_mode: str,
    user_selection: Optional[Sequence[str]] = None,
) -> Tuple[ObjectiveSpec, ...]:
    """Resolve the selection (or the mode defaults) through the registry, loudly.

    An unknown objective, or one the mode cannot measure, aborts the run: a
    silently dropped objective is a search that optimizes something other than
    what was asked for.
    """
    names = tuple(user_selection) if user_selection else default_objectives_for_mode(search_mode)
    return tuple(_project(spec) for spec in OBJECTIVES.resolve_active(search_mode, names))


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class Candidate(Generic[ConfigT]):
    configuration: ConfigT
    objectives: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult(Generic[ConfigT]):
    """Generic result container for any search backend."""

    objectives: Sequence[ObjectiveSpec]
    best: Candidate[ConfigT]
    pareto_front: List[Candidate[ConfigT]] = field(default_factory=list)
    all_candidates: List[Candidate[ConfigT]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


def _rank_candidates(
    candidates: Sequence[Candidate[ConfigT]],
    objectives: Sequence[ObjectiveSpec],
) -> List[List[int]]:
    """Return ``ranks[i][j]`` — the 1-based rank of candidate *i* on objective *j* (dense; rank 1 is best)."""
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
    """Pick the candidate whose worst rank across all objectives is minimal.

    Ties are broken by the sum of ranks (prefer the most uniformly strong).
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

    best_idx = min(tied, key=lambda i: sum(ranks[i]))
    return candidates[best_idx]


