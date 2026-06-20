"""Contract-driven step planning (Vector V5).

The deployment step sequence used to be hand-assembled with interleaved
per-flag ``append``s. V5 makes each step own its applicability —
``PipelineStep.applies_to(plan)`` — and a ``StepPlan`` filter an ordered
registry. "Which steps does this config need" becomes data each step owns, not
an 80-line conditional.

Composition:
- **V1** ``DeploymentPlan`` is the input every ``applies_to`` reads (resolved
  decisions, never raw ``config.get``).
- **V2** ``SpikingModePolicy`` owns the (firing × sync) activation-family
  decision (``plan.is_lif_style`` delegates to it), so the LIF / TTFS-cycle vs
  analytical-chain branch lives in the policy, not the planner.
- **V3** ``BACKEND_REGISTRY`` owns the backend tail: it validates every enabled
  backend against the capability matrix UP-FRONT (raising on an unsupported
  backend×mode at assembly) and returns the applicable backend step specs.

The registry order IS the pipeline order; ``resolve`` keeps that order and
drops the steps whose ``applies_to`` is false — byte-identical to the former
hand-assembly (locked by the golden-matrix test).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan


class StepPlanContractError(AssertionError):
    """A resolved step sequence's requires/promises DAG is not satisfiable."""


@dataclass(frozen=True)
class StepSpec:
    """One registry entry: a named step that declares its own applicability.

    ``applies`` defaults to the step class's ``applies_to`` classmethod (the
    step owns the decision); an explicit override is only for the backend tail,
    whose applicability + up-front validation live in ``BACKEND_REGISTRY``.
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

    The registry is either a flat ``StepSpec`` list (each filtered by its own
    ``applies_to``) or a callable that yields ``(name, class)`` pairs for a plan
    (the V3 backend tail, which must validate up-front as a side effect).
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
        """Resolve the step sequence and assert its requires/promises DAG holds.

        Vector V5: each step declares its data contract at the CLASS level
        (``PipelineStep.declared_contract``), so the satisfiability of the whole
        DAG — every consumed entry is promised by an EARLIER selected step — can
        be checked at ASSEMBLY time, before any step is instantiated or run. This
        mirrors the runtime ``Pipeline.verify`` / ``set_up_requirements``
        semantics (promises and updates make an entry available downstream;
        clears retract it) but fails loud with an actionable message naming the
        missing producer instead of a bare "requires X" assertion deep in a run.

        Returns the resolved ``(name, class)`` sequence (so callers can reuse it).
        """
        specs = self.resolve(plan)
        available: dict[str, str] = {}  # entry -> producing step name
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
