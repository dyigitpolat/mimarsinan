from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Sequence, TypeVar

from mimarsinan.search.problem import SearchProblem
from mimarsinan.search.results import ObjectiveSpec


ConfigT = TypeVar("ConfigT")


class ScalarEvaluator(Protocol[ConfigT]):
    def validate(self, configuration: ConfigT) -> bool:
        ...

    def evaluate(self, configuration: ConfigT) -> float:
        ...


@dataclass
class EvaluatorProblem(SearchProblem[ConfigT]):
    evaluator: ScalarEvaluator[ConfigT]
    objective: ObjectiveSpec

    @property
    def objectives(self) -> Sequence[ObjectiveSpec]:
        return (self.objective,)

    def validate(self, configuration: ConfigT) -> bool:
        return bool(self.evaluator.validate(configuration))

    def evaluate(self, configuration: ConfigT) -> Dict[str, float]:
        return {self.objective.name: float(self.evaluator.evaluate(configuration))}

    def meta(self, configuration: ConfigT) -> Dict[str, Any]:
        # Most legacy evaluators already cache internally; keep meta minimal.
        return {}


