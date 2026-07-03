"""AcceptanceSensor — the statistical accept/reject/recovered decision service."""

from __future__ import annotations

from dataclasses import dataclass

from mimarsinan.tuning.orchestration.tuner_base import CATASTROPHIC_DROP_FACTOR


@dataclass
class BaselineRef:
    """Fixed reference captured once at run start (spec §8.1 anti-drift)."""

    se: float
    empirical_noise: float
    rollback_tolerance: float
    baseline: float


class AcceptanceSensor:
    """Pure decision functions over the budget's accuracy standard error."""

    def __init__(self, budget):
        self._budget = budget

    def calibrate_baseline(self, validate_fn, eval_n_batches) -> BaselineRef:
        """Two reference evaluations → SE, empirical noise, tolerance, baseline.

        Two consecutive ``validate_fn`` calls at rate 0.0; ``rollback_tolerance``
        clamped to ``[0.005, 0.05]`` around ``max(3·SE, 3·noise)``.
        """
        se = self._budget.accuracy_se()
        val_a = validate_fn(eval_n_batches)
        val_b = validate_fn(eval_n_batches)
        empirical_noise = abs(val_a - val_b)
        rollback_tolerance = max(min(max(3 * se, 3 * empirical_noise), 0.05), 0.005)
        baseline = (val_a + val_b) / 2.0
        return BaselineRef(se, empirical_noise, rollback_tolerance, baseline)

    @staticmethod
    def absolute_floor(baseline, pipeline_tolerance, pipeline_hard_floor):
        """Baseline-anchored absolute floor (the cumulative-drift guard).

        A hard floor above the rate-0 baseline is unachievable by any transform, so
        it is dropped here (it would roll back every cycle); the baseline-anchored
        floor then governs the per-cycle gate.
        """
        floors = []
        if baseline is not None and pipeline_tolerance is not None:
            floors.append(float(baseline) * (1.0 - float(pipeline_tolerance)))
        if pipeline_hard_floor is not None:
            if baseline is None or float(pipeline_hard_floor) <= float(baseline):
                floors.append(float(pipeline_hard_floor))
        if not floors:
            return None
        return max(floors)

    @staticmethod
    def is_catastrophic(instant_acc, target, factor=CATASTROPHIC_DROP_FACTOR) -> bool:
        """Pre-recovery fast-fail: the instant accuracy has collapsed below
        ``target·factor`` (a coarse margin, not an SE gate — the pre-recovery drop
        is expected to be large, so only unrecoverable collapse is caught)."""
        return instant_acc is not None and float(instant_acc) < target * factor

    @staticmethod
    def rollback_threshold(pre_cycle_acc, rollback_tolerance, absolute_floor):
        """Stricter of the relative (noise) and absolute (anti-drift) gates."""
        relative = pre_cycle_acc - rollback_tolerance
        if absolute_floor is not None:
            return max(relative, absolute_floor)
        return relative

    @staticmethod
    def ratchet_threshold(
        pre_cycle_acc,
        rollback_tolerance,
        best_committed_acc,
        cumulative_bound,
        absolute_floor,
    ):
        """Non-stalling ratchet: the MAX of three lower bounds.

        The per-step relative gate keeps the ramp climbing; the cumulative bound
        (``best_committed_acc - cumulative_bound``) caps accumulated drift below the
        best high-water mark; the absolute baseline floor is the hard third bound.
        """
        bounds = [pre_cycle_acc - rollback_tolerance]
        if best_committed_acc is not None:
            bounds.append(best_committed_acc - cumulative_bound)
        if absolute_floor is not None:
            bounds.append(absolute_floor)
        return max(bounds)

    @staticmethod
    def is_rollback(post_acc, threshold) -> bool:
        return post_acc < threshold

    @staticmethod
    def reached_target(post_acc, target, rollback_tolerance) -> bool:
        return post_acc >= target - rollback_tolerance

    @staticmethod
    def paired_drop_se(ref_correct, cand_correct):
        """McNemar drop estimate and its SE from per-example correctness.

        Length-N boolean sequences over the same examples. Returns ``(delta_hat,
        se)``, positive ``delta_hat`` = the candidate's drop ``(b10 - b01)/N``,
        ``sqrt(b10+b01)/N`` (b10 = ref-right-cand-wrong, b01 = ref-wrong-cand-right).
        """
        ref = [bool(v) for v in ref_correct]
        cand = [bool(v) for v in cand_correct]
        n = len(ref)
        if n == 0 or len(cand) != n:
            return 0.0, 0.0
        b10 = sum(1 for r, c in zip(ref, cand) if r and not c)
        b01 = sum(1 for r, c in zip(ref, cand) if (not r) and c)
        delta = (b10 - b01) / n
        se = (b10 + b01) ** 0.5 / n
        return delta, se

    @classmethod
    def paired_is_rollback(cls, ref_correct, cand_correct, k_commit, min_effect=0.0):
        """Reject iff the paired drop is BOTH statistically significant (exceeds
        ``k_commit`` SEs) AND practically meaningful (exceeds ``min_effect``, the
        global budget floor that keeps the tight paired SE from thrashing)."""
        delta, se = cls.paired_drop_se(ref_correct, cand_correct)
        return delta > k_commit * se and delta > min_effect
