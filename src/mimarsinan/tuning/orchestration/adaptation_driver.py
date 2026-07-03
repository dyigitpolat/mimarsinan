"""AdaptationDriver — the thin orchestrator over the rate scheduler."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler


@dataclass
class CycleContext:
    """Per-cycle scratch the driver threads through the host's phase methods."""

    rate: float
    t_cycle_start: float
    pre_state: object = None
    pre_cycle_acc: float = 0.0
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

    def __init__(self, *, scheduler, attempt, finalize, committed: float = 0.0):
        self._scheduler = scheduler
        self._attempt = attempt
        self._finalize = finalize
        self._committed = float(committed)

    def run(self):
        self._scheduler.run(self._committed, self._attempt)
        return self._finalize()

    @staticmethod
    def run_cycle(host, rate):
        """One adaptation cycle's predictor → corrector → commit/rollback skeleton.

        The host binds the decision services into the phase methods this drives.
        Returns the committed rate after the cycle (``rate`` on commit, the prior
        committed rate on rollback)."""
        ctx = host._begin_cycle(rate)
        host._probe_instant(ctx)
        if ctx.is_catastrophic:
            return host._rollback_cycle(ctx, "catastrophic")
        host._recover(ctx)
        host._measure_post(ctx)
        if ctx.rolled_back:
            return host._rollback_cycle(ctx, "rollback")
        return host._commit_cycle(ctx)

    @staticmethod
    def build_scheduler(
        *, epsilon, max_rounds, skip_one_shot, initial_step,
        policy_override=None, rates=None,
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
            )
        if policy_override == "dense_grid":
            return RateScheduler(
                epsilon=epsilon,
                policy="dense_grid",
                initial_step=initial_step,
                max_rounds=max_rounds,
            )
        if skip_one_shot:
            return RateScheduler(
                epsilon=epsilon,
                policy="uniform_ladder",
                initial_step=initial_step,
                max_rounds=max_rounds,
            )
        return RateScheduler(epsilon=epsilon, policy="greedy_to_one", max_rounds=max_rounds)
