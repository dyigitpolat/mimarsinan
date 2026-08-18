"""[TS1] The one evaluation accountant a search spends against, and its ledger.

A campaign compares optimizers at EQUAL SPEND, so the currency needs one
definition: a DISTINCT decoded evaluation — the evaluator work a candidate
IDENTITY costs the first time a run spends it. The identity is what makes the
count whole: a problem answers about a candidate through more than one channel
(a constraint screen, an evaluation, a layout introspection), so whichever
channel spends the work first pays for it, and every later ask about that
identity is a DUPLICATE — it bought no new candidate, whether a cache answered
it or the channel resolved it again. An ask about an identity the run never
spent anything on is not an evaluation call at all.

The COMPARABLE axis is coarser than the call: a driver asks about one candidate
through as many CHANNELS as it likes, so a duplicate rate counted in calls
carries a structural floor equal to "how many channels does this driver use"
(measured: NSGA-II screens then scores, so it sealed 0.5556 for a search that
re-proposed 2 of 18 candidates). ``duplicate_rate`` therefore counts ROUNDS of
asking — every channel's first look at a candidate belongs to the round its
proposal opened, and a channel asking AGAIN about a candidate it already asked
about opens the next round. That number is the driver's re-proposal rate, and
it means the same thing to a one-channel driver and a three-channel one.

The ledger seals FACTS: wall, counts, both sides of the rate, the declared
limit, whether the budget cut the run short, and the LLM usage a driver
reports. It describes ONE interval — the search — so a driver seals it where
its clock stops. Money is deliberately absent — dollars are priced
research-side from a price table, so a sealed run can be re-priced without
re-running it.
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
    #: Spent identity -> the channels that have asked about it in its OPEN
    #: round. Keys are exactly the identities this run has paid for, so the
    #: distinct spend and the round bookkeeping cannot drift apart.
    _open_rounds: Dict[str, Set[str]] = field(
        default_factory=dict, init=False, repr=False,
    )
    _calls: int = field(default=0, init=False)
    _duplicates: int = field(default=0, init=False)
    _asks: int = field(default=0, init=False)
    _reasks: int = field(default=0, init=False)

    def charge(self, channel: str, key: str, *, hit: bool) -> None:
        """Route one channel's ask about *key*: what did it do to this run?

        THE routing, so no caller has to know the law:

        - unspent and a cache answered -> nothing happened. A candidate a cheap
          predicate refused before anything was built is not an evaluation the
          run made, and neither are the re-asks a cache answers about it.
        - unspent and work is about to run -> the DISTINCT evaluation this run
          pays for, opening the first round of asking about it.
        - already spent -> a DUPLICATE call, whatever answered it; it joins the
          open round, or opens the next one when this channel already asked.
        """
        if not self.has_spent(key):
            if hit:
                return
            self._calls += 1
            self._open_round(channel, key)
            return
        self._calls += 1
        self._duplicates += 1
        if channel in self._open_rounds[key]:
            self._open_round(channel, key)
            self._reasks += 1
        else:
            self._open_rounds[key].add(channel)

    def _open_round(self, channel: str, key: str) -> None:
        """Start a fresh round of asking about *key* — the unit the rate counts."""
        self._open_rounds[key] = {channel}
        self._asks += 1

    def has_spent(self, key: str) -> bool:
        """Has this run already paid for *key*'s evaluator work?"""
        return key in self._open_rounds

    @property
    def distinct_spent(self) -> int:
        return len(self._open_rounds)

    @property
    def raw_calls(self) -> int:
        """Every ask about a candidate this run evaluated — spends and re-asks."""
        return self._calls

    @property
    def exhausted(self) -> bool:
        """Has the DISTINCT spend reached the limit (an unlimited budget never has)?"""
        return self.limit is not None and self.distinct_spent >= int(self.limit)

    @property
    def identities_asked(self) -> int:
        """Rounds of asking about a candidate — every channel of one is one round."""
        return self._asks

    @property
    def identities_reasked(self) -> int:
        """Rounds that re-proposed a candidate the run had already asked about."""
        return self._reasks

    @property
    def duplicate_rate(self) -> float:
        """The share of asked candidates a driver asked for AGAIN.

        Counted over identities, never calls, so the number a campaign compares
        across optimizers does not read their channel counts.
        """
        if self._asks == 0:
            return 0.0
        return self._reasks / self._asks


def charge_evaluation(
    budget: Optional[EvaluationBudget], key: str, *, hit: bool, channel: str,
) -> None:
    """Tell the run's accountant what *channel*'s ask about *key* did.

    THE charging seam: a problem calls it at every cache that decides whether
    *key* costs work (``hit``), and at every uncached channel that is about to
    spend it (``hit=False``), naming the channel it answers through. The law is
    :meth:`EvaluationBudget.charge`; "no budget attached" costs one None check
    here instead of a branch per problem.
    """
    if budget is not None:
        budget.charge(channel, key, hit=hit)


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
    #: The rate's own numerator and denominator, sealed beside it: a derived
    #: fact an analysis must be able to re-derive, pool, or normalize.
    identities_asked: int
    identities_reasked: int
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
            "identities_asked": int(self.identities_asked),
            "identities_reasked": int(self.identities_reasked),
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
            identities_asked=int(payload["identities_asked"]),
            identities_reasked=int(payload["identities_reasked"]),
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
        identities_asked=budget.identities_asked,
        identities_reasked=budget.identities_reasked,
        duplicate_rate=budget.duplicate_rate,
        budget_limit=budget.limit,
        stopped_at_boundary=bool(stopped_at_boundary),
        llm=llm,
    )
