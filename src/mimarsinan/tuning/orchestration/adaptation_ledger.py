"""The adaptation ledger: one event per controller branch a tuning run took."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor

# Where a controller host keeps its ledger; absence means no telemetry.
LEDGER_ATTR = "_adaptation_ledger"
# Cache key a tuner-hosting step seals: ``<Step>.adaptation_ledger.json``.
LEDGER_ENTRY_KEY = "adaptation_ledger"

# Verdict outcomes: the ``_rollback_cycle`` outcome strings plus the commit exit.
COMMIT = "commit"
ROLLBACK = "rollback"
CATASTROPHIC = "catastrophic"

# Recovery kinds: a leg that trained, and the [LR-REFUSE] leg that could not.
RECOVERY_TRAINED = "trained"
RECOVERY_LR_REFUSED = "lr_refused"

# Refinement kinds: the controller adjusting its own search, not proposing.
BISECT_STEP = "bisect_step"
TARGET_RELAXATION = "target_relaxation"
LR_REFIND = "lr_refind"

# Escalation paths: every real branch out of the normal ramp. The first four
# also name how the rate search ENDED (``completed_via``), as do the next two.
LOSSLESS_ENTRY = "lossless_entry"
ROUND_BUDGET = "round_budget"
EPSILON_FLOOR = "epsilon_floor"
ONE_SHOT_REFUSED = "one_shot_refused"
REACHED_FULL_RATE = "reached_full_rate"
FIXED_LADDER_EXHAUSTED = "fixed_ladder_exhausted"
FORCED_FULL_RATE = "forced_full_rate"
KEEPBEST_RESTORE = "keepbest_restore"
WALK_RECOVERY_INSTALL = "walk_recovery_install"
LIF_RECOVERY_BUDGET_RESTORE = "lif_recovery_budget_restore"
ENDPOINT_ROLLBACK = "endpoint_rollback"
ENDPOINT_DIVERGENCE_RESCUE = "endpoint_divergence_rescue"
ENDPOINT_DECOUPLED = "endpoint_decoupled"


@dataclass(frozen=True)
class _Event:
    """Base for the frozen controller events: JSON-safe by construction."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Proposal(_Event):
    """One rate the scheduler proposed to a cycle (``run_cycle``'s entry)."""

    cycle_index: int
    rate: float
    committed: float
    increment: float


@dataclass(frozen=True)
class Verdict(_Event):
    """The accept/reject decision a cycle took, on the reading it took it from;
    ``threshold`` is ``None`` on the catastrophic branch, which short-circuits
    before the post-recovery gate is ever computed."""

    cycle_index: int
    outcome: str
    accepted: bool
    rolled_back: bool
    probe_metric: str
    reading: float
    threshold: Optional[float]
    probe_reads: int


@dataclass(frozen=True)
class Refinement(_Event):
    """One refinement the controller made to its own search, not a new proposal."""

    kind: str
    detail: str


@dataclass(frozen=True)
class Escalation(_Event):
    """One escalation/stall path the controller took out of its normal ramp."""

    path: str
    detail: str


@dataclass(frozen=True)
class Recovery(_Event):
    """One per-cycle recovery leg, as the driver observed the tuner consume it."""

    cycle_index: int
    kind: str
    steps: int
    lr: float


@dataclass(frozen=True)
class Endpoint(_Event):
    """One endpoint-recovery stage's training-step consumption."""

    stage: str
    steps: int
    budget_steps: int
    engaged: bool
    armed: bool
    reached: bool


class AdaptationLedger:
    """Accumulates the controller's events and the totals they sum to; every
    total is a sum over the event lists — never an independent counter — so a
    sealed artifact can be audited against its own events."""

    def __init__(self) -> None:
        self.proposals: List[Proposal] = []
        self.verdicts: List[Verdict] = []
        self.refinements: List[Refinement] = []
        self.escalations: List[Escalation] = []
        self.recoveries: List[Recovery] = []
        self.endpoints: List[Endpoint] = []
        self.completed_via: Optional[str] = None

    @property
    def cycle_index(self) -> int:
        """Index of the cycle in flight (``-1`` before the first proposal)."""
        return len(self.proposals) - 1

    def propose(self, *, rate: float, committed: float) -> None:
        rate, committed = float(rate), float(committed)
        self.proposals.append(Proposal(
            cycle_index=len(self.proposals), rate=rate, committed=committed,
            increment=rate - committed,
        ))

    def verdict(
        self, *, outcome: str, probe_metric: str, reading: float,
        threshold: Optional[float], probe_reads: int,
    ) -> None:
        self.verdicts.append(Verdict(
            cycle_index=self.cycle_index, outcome=str(outcome),
            accepted=outcome == COMMIT, rolled_back=outcome != COMMIT,
            probe_metric=str(probe_metric), reading=float(reading),
            threshold=None if threshold is None else float(threshold),
            probe_reads=int(probe_reads),
        ))

    def refine(self, kind: str, detail: str = "") -> None:
        self.refinements.append(Refinement(kind=str(kind), detail=str(detail)))

    def escalate(self, path: str, detail: str = "") -> None:
        self.escalations.append(Escalation(path=str(path), detail=str(detail)))

    def recover(self, *, kind: str, steps: int, lr: float) -> None:
        self.recoveries.append(Recovery(
            cycle_index=self.cycle_index, kind=str(kind), steps=int(steps),
            lr=float(lr),
        ))

    def endpoint(
        self, *, stage: str, steps: int, budget_steps: int, engaged: bool,
        armed: bool, reached: bool,
    ) -> None:
        self.endpoints.append(Endpoint(
            stage=str(stage), steps=int(steps), budget_steps=int(budget_steps),
            engaged=bool(engaged), armed=bool(armed), reached=bool(reached),
        ))

    def complete(self, path: str) -> None:
        """Name the path the rate search exited through (the Fig-7.2 question)."""
        self.completed_via = str(path)

    def totals(self) -> Dict[str, int]:
        """The run's spend, summed from the events.

        ``retries`` counts the scheduler's re-proposals (one per ``bisect_step``
        refinement, each followed by exactly one further attempt); ``probe_evals``
        counts the readings the acceptance gate consumed; ``recovery_steps`` is
        what the tuner reported training (a leg whose engine was not asked for a
        step count contributes 0 — code reality).
        """
        recovery_steps = sum(r.steps for r in self.recoveries)
        endpoint_steps = sum(e.steps for e in self.endpoints)
        return {
            "proposed": len(self.proposals),
            "accepted": sum(1 for v in self.verdicts if v.accepted),
            "rejected": sum(1 for v in self.verdicts if not v.accepted),
            "retries": sum(1 for r in self.refinements if r.kind == BISECT_STEP),
            "recovery_steps": recovery_steps,
            "probe_evals": sum(v.probe_reads for v in self.verdicts),
            "endpoint_steps": endpoint_steps,
            "total_steps": recovery_steps + endpoint_steps,
        }

    def stalls_by_path(self) -> Dict[str, int]:
        return dict(Counter(e.path for e in self.escalations))

    def to_dict(self) -> Dict[str, Any]:
        events = {
            "proposals": self.proposals, "verdicts": self.verdicts,
            "refinements": self.refinements, "escalations": self.escalations,
            "recoveries": self.recoveries, "endpoints": self.endpoints,
        }
        payload: Dict[str, Any] = {
            group: [e.to_dict() for e in items] for group, items in events.items()
        }
        payload["totals"] = self.totals()
        payload["stalls_by_path"] = self.stalls_by_path()
        payload["completed_via"] = self.completed_via
        return payload


def ledger_of(host) -> Optional[AdaptationLedger]:
    """The ledger a host keeps, else ``None`` — telemetry is opt-in, and a host
    without one runs byte-identically."""
    return getattr(host, LEDGER_ATTR, None)


def record_proposal(host, rate) -> Optional[AdaptationLedger]:
    """Record the cycle's proposal; returns the host's ledger for the rest of it."""
    ledger = ledger_of(host)
    if ledger is not None:
        ledger.propose(rate=float(rate), committed=float(host._committed_rate))
    return ledger


def record_recovery(ledger, ctx, steps: int) -> None:
    """Record the cycle's recovery leg. ``ctx.lr`` is 0.0 iff the LR sweep refused."""
    if ledger is None:
        return
    refused = float(ctx.lr) == 0.0
    ledger.recover(
        kind=RECOVERY_LR_REFUSED if refused else RECOVERY_TRAINED,
        steps=int(steps), lr=float(ctx.lr),
    )


def record_verdict(ledger, ctx, outcome: str) -> None:
    """Record the cycle's accept/reject verdict on the basis the sensor read."""
    if ledger is None:
        return
    catastrophic = outcome == CATASTROPHIC
    reading = ctx.instant_acc if catastrophic else ctx.post_acc
    ledger.verdict(
        outcome=outcome,
        probe_metric=AcceptanceSensor.probe_basis(
            catastrophic=catastrophic, paired=ctx.cand_correct is not None,
        ),
        reading=0.0 if reading is None else float(reading),
        threshold=None if catastrophic else float(ctx.rollback_threshold),
        probe_reads=int(ctx.pre_probe_drawn) + (1 if catastrophic else 2),
    )


def record_refinement(ledger, kind: str, detail: str = "") -> None:
    if ledger is not None:
        ledger.refine(kind, detail)


def record_escalation(ledger, path: str, detail: str = "") -> None:
    if ledger is not None:
        ledger.escalate(path, detail)


def record_completion(ledger, path: str) -> None:
    if ledger is not None:
        ledger.complete(path)


def record_endpoint(
    ledger, *, stage: str, steps: int, budget_steps: int, engaged: bool,
    armed: bool, reached: bool, rolled_back: bool, divergence_rescued: bool,
    decoupled: bool,
) -> None:
    """Record one endpoint stage: its step consumption plus each guard that fired."""
    if ledger is None:
        return
    ledger.endpoint(
        stage=stage, steps=steps, budget_steps=budget_steps, engaged=engaged,
        armed=armed, reached=reached,
    )
    for fired, path, detail in (
        (rolled_back, ENDPOINT_ROLLBACK, "ended below entry"),
        (divergence_rescued, ENDPOINT_DIVERGENCE_RESCUE, "restarted at a lower lr"),
        (decoupled, ENDPOINT_DECOUPLED, "surrogate decoupled from deployed"),
    ):
        if fired:
            ledger.escalate(path, f"{stage} {detail}")
