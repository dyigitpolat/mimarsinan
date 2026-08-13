"""Search result containers, and the legacy projection of the objectives registry.

The objective catalogue itself lives in ``deployment_record.objectives`` — one
registry over the deployment artifact, asked which axes a view can carry. This
module PROJECTS the searchable part of it onto the frozen ``ObjectiveSpec``
tuple every optimizer reads (``spec.name``/``spec.goal``), and keeps the
per-search-mode DEFAULTS, which are a search policy rather than a record fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

from mimarsinan.deployment_record.objectives import (
    ACCURACY_OBJECTIVE_KEY,
    OBJECTIVES,
    ObjectiveSpecV2,
    run_capability_probe,
)
from mimarsinan.deployment_record.platform_physics.profile import PlatformPhysics

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

#: "The caller did not state the run's physics" — distinct from ``None``, which
#: MEANS the run declared none and so cannot back a vendor-priced axis.
_UNDECLARED: Any = object()


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


def resolve_active_specs(
    search_mode: str,
    user_selection: Optional[Sequence[str]] = None,
    *,
    physics: Optional[PlatformPhysics] = _UNDECLARED,
) -> Tuple[ObjectiveSpecV2, ...]:
    """The ACTIVE registry specs — the axes an evaluation must produce, loudly.

    An unknown objective, or one the mode cannot measure, aborts the run: a
    silently dropped objective is a search that optimizes something other than
    what was asked for. Callers that only need ``name``/``goal`` use
    :func:`resolve_active_objectives`; callers that must READ the axis off a
    view (the evaluation contract) need the spec itself.

    ``physics`` is THIS RUN's declaration, and passing it narrows availability to
    what the run can actually back — a vendor-priced axis on a run that declares
    no profile is refused BY NAME here rather than producing a number no vendor
    stands behind. Omitting it keeps the question capability-level (what the MODE
    could carry), which is what an offer-the-whole-catalog caller asks.
    """
    names = tuple(user_selection) if user_selection else default_objectives_for_mode(search_mode)
    probe = (
        None if physics is _UNDECLARED
        else run_capability_probe(search_mode, physics)
    )
    return OBJECTIVES.resolve_active(search_mode, names, probe=probe)


def resolve_active_objectives(
    search_mode: str,
    user_selection: Optional[Sequence[str]] = None,
    *,
    physics: Optional[PlatformPhysics] = _UNDECLARED,
) -> Tuple[ObjectiveSpec, ...]:
    """:func:`resolve_active_specs`, projected onto the optimizer-facing pair."""
    return tuple(
        _project(spec)
        for spec in resolve_active_specs(search_mode, user_selection, physics=physics)
    )


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


def rank_objective_rows(
    rows: Sequence[Mapping[str, float]],
    objectives: Sequence[ObjectiveSpec],
) -> List[List[int]]:
    """Return ``ranks[i][j]`` — the 1-based rank of row *i* on objective *j* (dense; rank 1 is best)."""
    n = len(rows)
    ranks: List[List[int]] = [[0] * len(objectives) for _ in range(n)]

    for j, spec in enumerate(objectives):
        values = [float(row.get(spec.name, 0.0)) for row in rows]
        reverse = spec.goal == "max"
        order = sorted(range(n), key=lambda i: values[i], reverse=reverse)

        current_rank = 1
        for pos, idx in enumerate(order):
            if pos > 0 and values[order[pos]] != values[order[pos - 1]]:
                current_rank = pos + 1
            ranks[idx][j] = current_rank

    return ranks


def order_by_minimax_rank(
    rows: Sequence[Mapping[str, float]],
    objectives: Sequence[ObjectiveSpec],
) -> List[int]:
    """Indices of *rows*, best-balanced first: minimal worst rank, then rank sum.

    The one minimax ordering in the program — candidate selection, the Pareto
    orderings the LLM backends report, and the live-panel front all read it, so
    "best" means the same thing everywhere.
    """
    ranks = rank_objective_rows(rows, objectives)
    return sorted(
        range(len(rows)), key=lambda i: (max(ranks[i]), sum(ranks[i])),
    )


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

    order = order_by_minimax_rank([c.objectives for c in candidates], objectives)
    return candidates[order[0]]


