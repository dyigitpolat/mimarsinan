"""Contract-driven step planning: an ordered step registry filtered by each step's applicability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


class StepPlanContractError(AssertionError):
    """A resolved step sequence's requires/promises DAG is not satisfiable."""


@dataclass(frozen=True)
class StepSpec:
    """One registry entry: a named step that declares its own applicability.

    ``applies`` defaults to the step class's ``applies_to``; an explicit override
    is only for the backend tail (validated in ``BACKEND_REGISTRY``).
    """

    name: str
    step_class: type
    applies: Callable[[DeploymentPlan], bool] | None = None

    def applies_to(self, plan: DeploymentPlan) -> bool:
        if self.applies is not None:
            return self.applies(plan)
        return self.step_class.applies_to(plan)

    def to_pair(self) -> tuple[str, type]:
        return (self.name, self.step_class)


class StepPlan:
    """Ordered registry of pipeline steps, filtered by each step's applicability.

    Entries are ``StepSpec`` items or a callable that yields ``(name, class)`` pairs
    (the backend tail, which validates up-front as a side effect).
    """

    def __init__(
        self,
        entries: list[StepSpec | Callable[[DeploymentPlan], list[tuple[str, type]]]],
    ) -> None:
        self._entries = list(entries)

    def resolve(self, plan: DeploymentPlan) -> list[tuple[str, type]]:
        """Return the ordered ``(name, class)`` specs this plan needs."""
        specs: list[tuple[str, type]] = []
        for entry in self._entries:
            if isinstance(entry, StepSpec):
                if entry.applies_to(plan):
                    specs.append(entry.to_pair())
            else:
                specs.extend(entry(plan))
        return specs

    def validate_data_contract(self, plan: DeploymentPlan) -> list[tuple[str, type]]:
        """Resolve the step sequence and assert its requires/promises DAG holds at assembly time.

        Every consumed entry must be promised by an earlier selected step; fails
        loud naming the missing producer. Returns the resolved ``(name, class)`` sequence.
        """
        specs = self.resolve(plan)
        available: dict[str, str] = {}
        for name, step_class in specs:
            requires, promises, updates, clears = step_class.declared_contract()
            for requirement in requires:
                if requirement not in available:
                    raise StepPlanContractError(
                        f"Pipeline step '{name}' requires '{requirement}', "
                        f"but no earlier selected step promises it. "
                        f"Available entries at this point: "
                        f"{sorted(available)}. "
                        f"Add a step that promises '{requirement}' before '{name}', "
                        f"or fix the requires/promises class-level declarations."
                    )
            for entry in promises:
                available[entry] = name
            for entry in updates:
                available[entry] = name
            for entry in clears:
                available.pop(entry, None)
        return specs

    def step_classes(self) -> list[type]:
        """Every step class the registry can contribute (for coverage checks)."""
        return [e.step_class for e in self._entries if isinstance(e, StepSpec)]
