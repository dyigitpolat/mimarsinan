from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Protocol, Sequence, TypeVar

from mimarsinan.search.results import ObjectiveSpec


ConfigT = TypeVar("ConfigT")


@dataclass
class ValidationResult:
    """Result of a full feasibility check.

    Returned by :meth:`SearchProblem.validate_detailed` to communicate
    both the yes/no feasibility decision and, on failure, a human-readable
    explanation that can feed into the agentic constraint-learning loop.
    """

    is_valid: bool
    error_message: Optional[str] = None
    failure_phase: Optional[str] = None  # "structural", "model_build", "hw_conversion", "hw_packing"


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

    def validate_detailed(self, configuration: ConfigT) -> ValidationResult:
        """Full feasibility check with rich error information.

        The default implementation delegates to :meth:`validate`.
        Subclasses may override to perform heavier checks (model building,
        hardware packing) and return structured failure diagnostics.
        """
        return ValidationResult(is_valid=self.validate(configuration))

    def evaluate(self, configuration: ConfigT) -> Dict[str, float]:
        ...

    def constraint_violation(self, configuration: ConfigT) -> float:
        """
        Return a numerical constraint violation score for *configuration*.

        * ``<= 0`` means the candidate is **feasible**.
        * ``> 0``  means the candidate is **infeasible**; larger values indicate
          a more severe violation.

        The default implementation delegates to :meth:`validate`: 0 if valid,
        1 if invalid.  Override in subclasses to return a *continuous* violation
        metric so that evolutionary optimisers can guide the search toward the
        feasibility boundary.
        """
        return 0.0 if self.validate(configuration) else 1.0

    def meta(self, configuration: ConfigT) -> Dict[str, Any]:
        """
        Optional: return extra metadata to store alongside objectives (e.g., timing, feasibility flags).
        """
        return {}


