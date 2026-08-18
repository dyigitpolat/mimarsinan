"""AdaptationDriver — the thin orchestrator over the rate scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.tuning.orchestration import adaptation_ledger
from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler


@dataclass
class CycleContext:
    """Per-cycle scratch the driver threads through the host's phase methods."""

    rate: float
    t_cycle_start: float
    pre_state: object = None
    pre_cycle_acc: float = 0.0
    pre_probe_drawn: bool = False
    instant_acc: float | None = None
    is_catastrophic: bool = False
    lr: float = 0.0
    post_acc: float = 0.0
    cand_correct: list | None = None
    noise_margin: float = 0.0
    absolute_floor: float | None = None
    rollback_threshold: float = 0.0
    rolled_back: bool = False


class AdaptationDriver:
    """Drive an axis from ``committed`` toward 1.0 via the scheduler, then finalize."""

    def __init__(self, *, scheduler, attempt, finalize, committed: float = 0.0,
                 entry_short_circuit=None, ledger=None):
        self._scheduler = scheduler
        self._attempt = attempt
        self._finalize = finalize
        self._committed = float(committed)
        self._entry_short_circuit = entry_short_circuit
        self._ledger = ledger

    def run(self):
        # [recipe-economics] a lossless entry has nothing to smooth: the
        # scheduler is skipped and finalize applies the full rate directly.
        if self._entry_short_circuit is not None and self._entry_short_circuit():
            adaptation_ledger.record_escalation(
                self._ledger, adaptation_ledger.LOSSLESS_ENTRY,
                "transform lossless at entry; ladder+ramp skipped",
            )
            adaptation_ledger.record_completion(
                self._ledger, adaptation_ledger.LOSSLESS_ENTRY
            )
            return self._finalize()
        self._scheduler.run(self._committed, self._attempt)
        return self._finalize()

    @staticmethod
    def run_cycle(host, rate):
        """One adaptation cycle's predictor → corrector → commit/rollback skeleton.

        The host binds the decision services into the phase methods this drives.
        Returns the committed rate after the cycle (``rate`` on commit, the prior
        committed rate on rollback). Each of the three exits records exactly one
        ledger verdict; a host that keeps no ledger runs byte-identically."""
        ledger = adaptation_ledger.record_proposal(host, rate)
        ctx = host._begin_cycle(rate)
        host._probe_instant(ctx)
        if ctx.is_catastrophic:
            committed = host._rollback_cycle(ctx, "catastrophic")
            adaptation_ledger.record_verdict(ledger, ctx, "catastrophic")
            return committed
        steps_before = AdaptationDriver._recovery_steps(host, ledger)
        host._recover(ctx)
        adaptation_ledger.record_recovery(
            ledger, ctx, AdaptationDriver._recovery_steps(host, ledger) - steps_before,
        )
        host._measure_post(ctx)
        if ctx.rolled_back:
            committed = host._rollback_cycle(ctx, "rollback")
            adaptation_ledger.record_verdict(ledger, ctx, "rollback")
            return committed
        committed = host._commit_cycle(ctx)
        adaptation_ledger.record_verdict(ledger, ctx, "commit")
        return committed

    @staticmethod
    def _recovery_steps(host, ledger) -> int:
        """The host's running recovery-step total (0 when nothing is metering)."""
        return 0 if ledger is None else int(host._gradual_train_steps)

    @staticmethod
    def build_scheduler(
        *, epsilon, max_rounds, skip_one_shot, initial_step,
        policy_override=None, rates=None, ledger=None,
    ):
        """Select the rate-search policy: ``policy_override`` (``fixed_ladder`` /
        ``dense_grid``) wins; else a uniform ladder for the KD-blend family
        (``skip_one_shot``); else greedy-to-1.0 + bisect."""
        if policy_override == "fixed_ladder":
            return RateScheduler(
                epsilon=epsilon,
                policy="fixed_ladder",
                rates=rates,
                max_rounds=max_rounds,
                ledger=ledger,
            )
        if policy_override == "dense_grid":
            return RateScheduler(
                epsilon=epsilon,
                policy="dense_grid",
                initial_step=initial_step,
                max_rounds=max_rounds,
                ledger=ledger,
            )
        if skip_one_shot:
            return RateScheduler(
                epsilon=epsilon,
                policy="uniform_ladder",
                initial_step=initial_step,
                max_rounds=max_rounds,
                ledger=ledger,
            )
        return RateScheduler(
            epsilon=epsilon, policy="greedy_to_one", max_rounds=max_rounds,
            ledger=ledger,
        )
