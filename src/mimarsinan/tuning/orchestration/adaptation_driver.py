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
    def build_scheduler(*, epsilon, max_rounds, skip_one_shot, initial_step, policy_override=None):
        """Select the rate-search policy: ``policy_override`` (e.g. ``dense_grid``
        from a non-monotone characterization) wins; else a uniform ladder for the
        KD-blend family (``skip_one_shot``); else greedy-to-1.0 + bisect."""
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
