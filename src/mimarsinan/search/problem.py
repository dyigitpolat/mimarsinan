from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, Protocol, Sequence, TypeVar

from mimarsinan.search.results import ObjectiveSpec


ConfigT = TypeVar("ConfigT")


@dataclass
class ValidationResult:
    """Feasibility decision plus, on failure, a human-readable explanation."""

    is_valid: bool
    error_message: Optional[str] = None
    failure_phase: Optional[str] = None


class SearchProblem(Protocol, Generic[ConfigT]):
    """Validates decoded candidates and evaluates their objectives."""

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        ...

    def validate(self, configuration: ConfigT) -> bool:
        ...

    def validate_detailed(self, configuration: ConfigT) -> ValidationResult:
        """Full feasibility check with rich error information; defaults to :meth:`validate`."""
        return ValidationResult(is_valid=self.validate(configuration))

    def evaluate(self, configuration: ConfigT) -> Dict[str, float]:
        ...

    def constraint_violation(self, configuration: ConfigT) -> float:
        """Continuous constraint-violation score: ``<= 0`` feasible, ``> 0`` infeasible."""
        return 0.0 if self.validate(configuration) else 1.0

    def meta(self, configuration: ConfigT) -> Dict[str, Any]:
        """Optional extra metadata to store alongside objectives."""
        return {}


