"""[TS1] The accountant's answers: per ask (an admission) and per boundary (a stop)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Admission:
    """What one charge did: admitted or refused, and whether it bought a token.

    ``bool(admission)`` is "may the work run". A DISTINCT admission holds the
    token it reserved and hands it back through :meth:`release` when the
    evaluator broke instead of evaluating (an apparatus exception): nothing
    was evaluated, so the run did not spend it.
    """

    admitted: bool
    distinct: bool
    key: str = ""
    #: The EvaluationBudget that issued it (typed loosely: it imports this leaf).
    budget: Optional[Any] = None

    def __bool__(self) -> bool:
        return self.admitted

    def release(self) -> None:
        if self.distinct and self.budget is not None:
            self.budget.release(self.key)


#: The answer when nobody meters the run: every ask is admitted and free.
UNMETERED = Admission(admitted=True, distinct=False)


@dataclass
class BoundaryStop:
    """A driver's own boundary, asked once per boundary: is the budget spent?

    The accountant admits or refuses one evaluation at a time; WHERE a run
    ends is every driver's own decision — a generation, a batch, a proposal.
    Ask this only where the run would otherwise CONTINUE, and ``stopped`` is
    exactly what the ledger means by ``stopped_at_boundary``: the budget
    denied work the run wanted to do.
    """

    #: The run's EvaluationBudget (typed loosely: it imports this leaf).
    budget: Optional[Any] = None
    _stopped: bool = field(default=False, init=False, repr=False)

    def should_stop(self) -> bool:
        """Stop here? Answering True is what makes this run a budget-bound one."""
        if self.budget is None or not self.budget.exhausted:
            return False
        self._stopped = True
        return True

    @property
    def stopped(self) -> bool:
        return self._stopped


__all__ = ["Admission", "BoundaryStop", "UNMETERED"]
