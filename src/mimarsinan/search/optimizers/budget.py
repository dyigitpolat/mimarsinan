"""[TS1] The one evaluation accountant a search spends against, and its ledger.

A campaign compares optimizers at EQUAL SPEND, so the currency needs one
definition: a DISTINCT decoded evaluation — the evaluator work a candidate
IDENTITY costs the first time a run spends it. The identity is what makes the
count whole: a problem answers about a candidate through more than one channel
(a constraint screen, then an evaluation), so whichever channel spends the work
first pays for it, and every later ask a cache answers is a duplicate.
Duplicates are what a cache absorbs, never what a budget pays for.

The ledger seals FACTS: wall, counts, duplicate rate, the declared limit,
whether the budget cut the run short, and the LLM usage a driver reports. It
describes ONE interval — the search — so a driver seals it where its clock
stops. Money is deliberately absent — dollars are priced research-side from a
price table, so a sealed run can be re-priced without re-running it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set


@dataclass
class EvaluationBudget:
    """Meters distinct decoded evaluations against an optional limit.

    Metering only: an exhausted budget never refuses an evaluation. Stopping is
    the driver's decision at ITS natural boundary (a generation, a batch, a
    proposal round), and the ledger records the exact spend that resulted.
    One accountant meters one run.
    """

    limit: Optional[int] = None
    _charged: Set[str] = field(default_factory=set, init=False, repr=False)
    _duplicates: int = field(default=0, init=False)

    def on_distinct(self, key: str) -> None:
        """Charge the run for *key*'s evaluator work — once per identity, ever.

        Idempotent BY IDENTITY: a candidate resolved by the constraint channel
        and then scored by the evaluate channel cost this run one evaluation,
        and a second channel charging for it would price the same work twice.
        """
        self._charged.add(key)

    def on_duplicate(self, key: str) -> None:
        """Record a re-ask of *key*: the call happened, the work did not."""
        self._duplicates += 1

    @property
    def distinct_spent(self) -> int:
        return len(self._charged)

    @property
    def raw_calls(self) -> int:
        return len(self._charged) + self._duplicates

    @property
    def exhausted(self) -> bool:
        """Has the DISTINCT spend reached the limit (an unlimited budget never has)?"""
        return self.limit is not None and self.distinct_spent >= int(self.limit)

    @property
    def duplicate_rate(self) -> float:
        """The share of evaluation calls the cache answered."""
        if self.raw_calls == 0:
            return 0.0
        return self._duplicates / self.raw_calls


def charge_evaluation(
    budget: Optional[EvaluationBudget], key: str, *, hit: bool,
) -> None:
    """Tell the run's accountant which kind of evaluation *key* just was.

    THE charging seam: every problem calls this where a cache decides whether
    *key* costs work, so "no budget attached" costs one None check instead of a
    branch per problem.
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
    #: The budget CUT THE RUN SHORT. A run whose own termination ended it seals
    #: False even when the budget was spent exactly — compare
    #: ``evaluations_distinct`` against ``budget_limit`` for that.
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
    Sealed where the driver's clock stops, so every fact in it covers the same
    interval: what the SEARCH spent, not the bookkeeping that follows it.
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
