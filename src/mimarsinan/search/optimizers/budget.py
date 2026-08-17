"""[TS1] The one evaluation accountant a search spends against, and its ledger.

A campaign compares optimizers at EQUAL SPEND, so the currency needs one
definition: a DISTINCT decoded evaluation — the evaluator work a candidate
identity costs the first time it is asked for. Duplicates are what a cache
absorbs, never what a budget pays for. The accountant is told which of the two
happened at the ONE seam that already knows (the problem's evaluation cache);
no driver counts for itself.

The ledger seals FACTS: wall, counts, duplicate rate, the declared limit,
whether the run stopped at a boundary, and the LLM usage a driver reports.
Money is deliberately absent — dollars are priced research-side from a price
table, so a sealed run can be re-priced without re-running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EvaluationBudget:
    """Meters distinct decoded evaluations against an optional limit.

    Metering only: an exhausted budget never refuses an evaluation. Stopping is
    the driver's decision at ITS natural boundary (a generation, a batch, a
    proposal round), and the ledger records the exact spend that resulted.
    """

    limit: Optional[int] = None
    _distinct: int = field(default=0, init=False)
    _raw: int = field(default=0, init=False)

    def on_distinct(self, key: str) -> None:
        """Charge the run for the first evaluation of *key* — real evaluator work."""
        self._distinct += 1
        self._raw += 1

    def on_duplicate(self, key: str) -> None:
        """Record a re-ask of *key*: the call happened, the work did not."""
        self._raw += 1

    @property
    def distinct_spent(self) -> int:
        return self._distinct

    @property
    def raw_calls(self) -> int:
        return self._raw

    @property
    def exhausted(self) -> bool:
        """Has the DISTINCT spend reached the limit (an unlimited budget never has)?"""
        return self.limit is not None and self._distinct >= int(self.limit)

    @property
    def duplicate_rate(self) -> float:
        """The share of evaluation calls the cache answered."""
        if self._raw == 0:
            return 0.0
        return (self._raw - self._distinct) / self._raw


def charge_evaluation(
    budget: Optional[EvaluationBudget], key: str, *, hit: bool,
) -> None:
    """Tell the run's accountant which kind of evaluation *key* just was.

    THE charging seam: every problem calls this at its cache, so "no budget
    attached" costs one None check instead of a branch per problem.
    """
    if budget is None:
        return
    if hit:
        budget.on_duplicate(key)
    else:
        budget.on_distinct(key)


def problem_budget(problem: object) -> Optional[EvaluationBudget]:
    """The accountant a problem carries, if any — the ONE way a driver finds it."""
    budget = getattr(problem, "evaluation_budget", None)
    if budget is None or isinstance(budget, EvaluationBudget):
        return budget
    raise TypeError(
        f"evaluation_budget must be an EvaluationBudget or None, got "
        f"{type(budget).__name__}"
    )


@dataclass(frozen=True)
class LlmUsage:
    """What a search asked of a model — never what it was billed."""

    model: str
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "calls": int(self.calls),
            "tokens_in": int(self.tokens_in),
            "tokens_out": int(self.tokens_out),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LlmUsage":
        return cls(
            model=str(payload["model"]),
            calls=int(payload.get("calls", 0)),
            tokens_in=int(payload.get("tokens_in", 0)),
            tokens_out=int(payload.get("tokens_out", 0)),
        )


@dataclass(frozen=True)
class ResourceLedger:
    """What one search actually spent, sealed as facts an analysis can normalize."""

    wall_s: float
    evaluations_raw: int
    evaluations_distinct: int
    duplicate_rate: float
    budget_limit: Optional[int] = None
    stopped_at_boundary: bool = False
    llm: Optional[LlmUsage] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_s": float(self.wall_s),
            "evaluations_raw": int(self.evaluations_raw),
            "evaluations_distinct": int(self.evaluations_distinct),
            "duplicate_rate": float(self.duplicate_rate),
            "budget_limit": None if self.budget_limit is None else int(self.budget_limit),
            "stopped_at_boundary": bool(self.stopped_at_boundary),
            "llm": None if self.llm is None else self.llm.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResourceLedger":
        llm = payload.get("llm")
        limit = payload.get("budget_limit")
        return cls(
            wall_s=float(payload["wall_s"]),
            evaluations_raw=int(payload["evaluations_raw"]),
            evaluations_distinct=int(payload["evaluations_distinct"]),
            duplicate_rate=float(payload["duplicate_rate"]),
            budget_limit=None if limit is None else int(limit),
            stopped_at_boundary=bool(payload.get("stopped_at_boundary", False)),
            llm=None if llm is None else LlmUsage.from_dict(llm),
        )


def seal_ledger(
    budget: Optional[EvaluationBudget],
    *,
    wall_s: float,
    stopped_at_boundary: bool,
    llm: Optional[LlmUsage] = None,
) -> Optional[ResourceLedger]:
    """The run's ledger, or None when no accountant metered it.

    A run nobody metered has no counts, and a ledger of zeros would be a claim
    about a search that was never measured — so there is no ledger at all.
    """
    if budget is None:
        return None
    return ResourceLedger(
        wall_s=float(wall_s),
        evaluations_raw=budget.raw_calls,
        evaluations_distinct=budget.distinct_spent,
        duplicate_rate=budget.duplicate_rate,
        budget_limit=budget.limit,
        stopped_at_boundary=bool(stopped_at_boundary),
        llm=llm,
    )
