"""AdaptationDriver — the thin orchestrator over the rate scheduler.

Composes the ``RateScheduler`` with a per-cycle attempt (predictor → corrector →
commit/rollback) and a finalize tail. Today the attempt and finalize are the
tuner's ``_adaptation`` / ``_finalize_run`` (which already delegate decisions to
``AcceptanceSensor``, recovery to ``RecoveryEngine``, snapshots to
``CheckpointGuard``, and rate application to the ``AdaptationAxis``). The fully
standalone form — constructing the driver from ``(model, axis, sensor, recovery,
guard, scheduler)`` and dissolving ``_adaptation`` into ``_cycle`` here — is the
remaining V6 step; this class is the seam that makes it incremental.
"""

from __future__ import annotations

from mimarsinan.tuning.orchestration.rate_scheduler import RateScheduler


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
    def build_scheduler(
        *, epsilon, max_rounds, skip_one_shot, initial_step, sensitivity_stepping=False
    ):
        """Select the rate-search policy: uniform ladder for the KD-blend family
        (``skip_one_shot``); ``last_successful_step`` when sensitivity-guided
        (P5b, cheaper on cliff-like axes); greedy-to-1.0 + bisect otherwise."""
        if skip_one_shot:
            return RateScheduler(
                epsilon=epsilon,
                policy="uniform_ladder",
                initial_step=initial_step,
                max_rounds=max_rounds,
            )
        if sensitivity_stepping:
            return RateScheduler(
                epsilon=epsilon, policy="last_successful_step", max_rounds=max_rounds
            )
        return RateScheduler(epsilon=epsilon, policy="greedy_to_one", max_rounds=max_rounds)
