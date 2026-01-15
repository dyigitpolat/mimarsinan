from __future__ import annotations

from typing import Any, Dict, Generic, Protocol, Sequence, TypeVar

from mimarsinan.search.results import ObjectiveSpec


ConfigT = TypeVar("ConfigT")


class SearchProblem(Protocol, Generic[ConfigT]):
    """
    A search problem defines:
    - how to validate a decoded candidate (fast feasibility)
    - how to evaluate objectives for a candidate

    Note: decision variable encoding/decoding is handled by specific optimizers (e.g. NSGA-II)
    or by problem-specific helpers in later iterations.
    """

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        ...

    def validate(self, configuration: ConfigT) -> bool:
        ...

    def evaluate(self, configuration: ConfigT) -> Dict[str, float]:
        ...

    def meta(self, configuration: ConfigT) -> Dict[str, Any]:
        """
        Optional: return extra metadata to store alongside objectives (e.g., timing, feasibility flags).
        """
        return {}


