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


