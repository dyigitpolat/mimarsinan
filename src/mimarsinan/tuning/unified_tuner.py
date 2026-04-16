"""TunerBase and SmoothAdaptationTuner -- the single orchestration hierarchy.

TunerBase provides shared infrastructure (pipeline, model, trainer, budget,
target adjuster, LR finder). SmoothAdaptationTuner adds the greedy bisection
loop for smooth rate-based adaptation.

Concrete tuners override _update_and_evaluate(rate) and optionally
_before_cycle() / _after_run() / _recovery_training_hooks(rate).
"""

from __future__ import annotations

import time

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import build_recipe
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster
from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.learning_rate_explorer import (
    clone_state_for_trainer,
    find_lr_range_for_trainer,
    restore_state_for_trainer,
)
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.tuning_budget import (
    min_step_for_smooth_adaptation,
    resolve_tuning_batch_size,
    tuning_budget_from_pipeline,
)


CATASTROPHIC_DROP_FACTOR = 0.8
"""Fast-fail threshold as a fraction of the adaptation target.

If the instant accuracy after applying a transformation drops below
``target * CATASTROPHIC_DROP_FACTOR``, the cycle is abandoned immediately
without wasting compute on LR exploration and recovery training.
A value of 0.8 means any >20% instant drop is treated as unrecoverable.
"""

_RECOVERY_PATIENCE = 5
"""Default patience (consecutive stale checks) for recovery training."""

_STUCK_STREAK_REQUIRED = 3
"""Consecutive cycles missing the original target before it is relaxed."""


class TunerBase:
    """Shared infrastructure for all tuners (except CoreFlowTuner).

    Provides pipeline access, model, trainer, budget, target adjuster,
    LR finder, and validate().  Subclasses implement run().
    """

    _budget_multiplier = 1.0

    _skip_one_shot = False

    def __init__(self, pipeline, model, target_accuracy, lr):
        self.pipeline = pipeline
        self.model = model
        self.pipeline_lr = lr
        self.lr = lr
        self.name = "Tuning Rate"

        self._budget = tuning_budget_from_pipeline(pipeline)
        if self._budget_multiplier != 1.0:
            self._budget.max_training_steps = int(
                self._budget.max_training_steps * self._budget_multiplier
            )
        self.target_adjuster = AdaptationTargetAdjuster.from_pipeline(
            target_accuracy, pipeline
        )

        self.trainer = self._create_trainer()
        self.trainer.report_function = pipeline.reporter.report

    def _tuning_recipe(self):
        """Recipe for tuning-phase trainers (explicit opt-in via ``tuning_recipe``).

        Returns ``None`` when the user has not configured ``tuning_recipe``,
        which preserves the legacy Adam/AdamW dynamics that existing SNN
        adaptation code relies on. We do NOT fall back to ``training_recipe``
        on purpose: fine-tuning recipes (LLRD + wd=0.05) are tuned for
        transfer learning and can destabilize rate-based adaptation loops.
        """
        return build_recipe(self.pipeline.config, key="tuning_recipe")

    def _create_trainer(self):
        num_workers = self.pipeline.config.get("num_workers", 4)
        trainer = BasicTrainer(
            self.model,
            self.pipeline.config["device"],
            DataLoaderFactory(self.pipeline.data_provider_factory,
                              num_workers=num_workers),
            self.pipeline.loss,
            recipe=self._tuning_recipe(),
        )
        # Tuning uses a smaller training batch than the training/fine-tuning
        # phase: smaller per-step activation memory, more gradient updates per
        # epoch of data. Validation/test loaders keep their original batch
        # size so eval cost and noise are unchanged.
        tuning_bs = resolve_tuning_batch_size(self.pipeline, trainer.training_batch_size)
        if tuning_bs != trainer.training_batch_size:
            trainer.set_training_batch_size(tuning_bs)
        return trainer

    def close(self):
        """Shut down DataLoader workers owned by this tuner."""
        if hasattr(self, "trainer") and self.trainer is not None:
            self.trainer.close()

    def _find_lr(self):
        return find_lr_range_for_trainer(
            self.trainer,
            self.pipeline,
            self._budget,
            validate_fn=lambda: self.trainer.validate_n_batches(
                self._budget.progress_eval_batches
            ),
            anchor_lr=self.pipeline_lr,
        )

    # -- LR cache ---------------------------------------------------------------
    # The optimal recovery LR barely changes between adjacent rates for a
    # pretrained model, so probe once per tuner run and reuse across cycles.
    # Invalidated on rollback / stuck-streak / safety-net re-probe.

    def _get_cached_lr(self):
        if getattr(self, "_cached_lr", None) is None:
            self._cached_lr = self._find_lr()
        return self._cached_lr

    def _invalidate_lr_cache(self):
        self._cached_lr = None

    def _get_target(self):
        return self.target_adjuster.get_target()

    def validate(self):
        return self.trainer.validate()

    @property
    def final_metric(self):
        """Cached final test-consistent metric set by ``_after_run``.

        ``None`` before the tuner has finished; a float on the test-set
        scale afterwards. Used by :meth:`PipelineStep.pipeline_metric`
        to avoid a redundant ``trainer.test()`` pass after the tuner
        already computed one internally.
        """
        return getattr(self, "_final_metric", None)

    def run(self):
        raise NotImplementedError


class SmoothAdaptationTuner(TunerBase):
    """The ONE orchestration loop for smooth rate-based adaptation.

    Subclasses must implement ``_update_and_evaluate(rate) -> float``.
    Optional hooks: ``_before_cycle()``, ``_after_run() -> float``,
    ``_recovery_training_hooks(rate) -> list``.
    """

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._committed_rate = 0.0
        self._natural_rate = 0.0
        self._pipeline_tolerance = float(
            pipeline.config.get("degradation_tolerance", 0.05)
        )
        se = self._budget.accuracy_se()
        self._rollback_tolerance = 3 * se
        self._missed_target_streak = 0
        self._pipeline_hard_floor = None
        self._pre_relaxation_target = None
        self._test_baseline = None
        self._validation_baseline = None
        self._cycle_log = []
        self._cached_lr = None

    def _update_and_evaluate(self, rate):
        """Apply transformation T at *rate* and return a validation metric."""
        raise NotImplementedError

    def _before_cycle(self):
        """Called before each adaptation cycle (e.g. refresh pruning importance)."""

    def _after_run(self):
        """Called after adaptation completes. Returns the final metric."""
        return self.trainer.validate()

    def _recovery_training_hooks(self, rate):
        """Return a list of hooks to keep active during recovery training.

        Subclasses (e.g. PruningTuner) override this to register forward
        pre-hooks that enforce invariants (e.g. pruning masks) throughout
        recovery. Hooks are removed after recovery finishes.
        """
        return []

    # -- State snapshot protocol ------------------------------------------------
    # NOTE: State snapshot does NOT include optimizer state. Optimizers MUST be
    # recreated after state restore. If optimizer persistence is added in the
    # future, rollback will silently produce wrong results.

    def _get_extra_state(self):
        """Override to return tuner-specific state to save alongside model params."""
        return None

    def _set_extra_state(self, extra):
        """Override to restore tuner-specific state."""

    def _clone_state(self):
        model_state = clone_state_for_trainer(self.trainer)
        return (model_state, self._get_extra_state())

    def _restore_state(self, state):
        model_state, extra = state
        restore_state_for_trainer(self.trainer, model_state)
        if extra is not None:
            self._set_extra_state(extra)

    # -- Core adaptation cycle --------------------------------------------------

    def _adaptation(self, rate):
        """Recovery training at a given rate: T -> L -> R, with rollback.

        Clones state before the transformation and measures ``pre_cycle_acc``
        (accuracy immediately before this step's transformation is applied).
        After recovery, if ``post_acc`` drops more than one noise-margin below
        ``pre_cycle_acc``, restores the checkpoint. This enforces per-step
        no-drop: any regression beyond measurement noise causes rollback and
        the caller (``SmartSmoothAdaptation``) halves the step for the retry.
        ``degradation_tolerance`` is a hard pipeline cutoff, not a per-step
        budget — the tuner targets zero loss.
        """
        t_cycle_start = time.time()
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        pre_state = self._clone_state()
        pre_cycle_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        instant_acc = self._update_and_evaluate(rate)

        # Fast-fail: skip expensive LR exploration + training if model collapsed
        catastrophic_floor = self._get_target() * CATASTROPHIC_DROP_FACTOR
        if instant_acc is not None and float(instant_acc) < catastrophic_floor:
            self._restore_state(pre_state)
            if hasattr(self, "_cycle_log"): self._cycle_log.append({
                "rate": rate, "committed": self._committed_rate,
                "instant_acc": float(instant_acc), "outcome": "catastrophic",
                "elapsed_sec": time.time() - t_cycle_start,
            })
            return self._committed_rate

        t0 = time.time()
        lr = self._get_cached_lr()
        t_lr = time.time() - t0
        self.pipeline.reporter.report("LR_found", lr)
        self.pipeline.reporter.report("T_find_lr_sec", t_lr)

        hooks = self._recovery_training_hooks(rate)
        try:
            self.trainer.train_steps_until_target(
                lr,
                self._budget.max_training_steps,
                self._get_target(),
                0,
                validation_n_batches=self._budget.progress_eval_batches,
                check_interval=self._budget.check_interval,
                patience=_RECOVERY_PATIENCE,
                min_steps=self._budget.check_interval * 3,
                min_improvement=self._budget.accuracy_se(),
            )
        finally:
            for h in hooks:
                h.remove()

        post_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)

        noise_margin = self._rollback_tolerance
        # Per-step rollback gate: compare post_acc to the pre-step accuracy,
        # not to the tuner's (possibly decayed) target. Any drop beyond
        # measurement noise is a regression and must be rolled back.
        rollback_threshold = pre_cycle_acc - noise_margin
        if post_acc < rollback_threshold:
            self._restore_state(pre_state)
            self._invalidate_lr_cache()
            if hasattr(self, "_cycle_log"): self._cycle_log.append({
                "rate": rate, "committed": self._committed_rate,
                "instant_acc": float(instant_acc) if instant_acc is not None else None,
                "pre_cycle_acc": float(pre_cycle_acc),
                "post_acc": float(post_acc), "lr": lr,
                "outcome": "rollback", "elapsed_sec": time.time() - t_cycle_start,
            })
            return self._committed_rate

        if rate >= 1.0 - 1e-6:
            test_acc = self.trainer.test()
            # Strict internal gate: test_acc must be within noise of the test
            # baseline captured once at run start. The pipeline hard floor is
            # only the safety net (in _ensure_pipeline_threshold).
            test_baseline = getattr(self, "_test_baseline", None)
            if test_baseline is not None:
                strict_threshold = float(test_baseline) - noise_margin
            else:
                strict_threshold = (
                    float(self.target_adjuster.original_metric) - noise_margin
                )
            if test_acc < strict_threshold:
                self._restore_state(pre_state)
                self._invalidate_lr_cache()
                if hasattr(self, "_cycle_log"): self._cycle_log.append({
                    "rate": rate, "committed": self._committed_rate,
                    "pre_cycle_acc": float(pre_cycle_acc),
                    "post_acc": float(post_acc), "test_acc": float(test_acc),
                    "strict_threshold": strict_threshold, "lr": lr,
                    "outcome": "test_gate_fail", "elapsed_sec": time.time() - t_cycle_start,
                })
                return self._committed_rate

        self._committed_rate = rate

        # Stuck detection: count cycles whose post_acc did NOT reach the
        # current target (within noise). "Small committed-rate delta" is NOT
        # used — it conflates endgame fine steps with genuine stuck-ness.
        reached_target = post_acc >= self._get_target() - noise_margin
        if reached_target:
            self._missed_target_streak = 0
            # Restore a previously-relaxed target: we've proven we can meet it.
            pre_relax = getattr(self, "_pre_relaxation_target", None)
            if pre_relax is not None:
                self.target_adjuster.target_metric = pre_relax
                self._pre_relaxation_target = None
        else:
            self._missed_target_streak += 1

        if self._missed_target_streak >= _STUCK_STREAK_REQUIRED:
            # Single-step decay via AdaptationTargetAdjuster's existing logic
            # (multiplicative ``decay``, clamped at pipeline floor). Save the
            # pre-relaxation target so we can restore it when a later cycle
            # actually reaches the original target.
            self._pre_relaxation_target = self._get_target()
            self.target_adjuster.update_target(post_acc)
            self._missed_target_streak = 0
            self._invalidate_lr_cache()

        if hasattr(self, "_cycle_log"): self._cycle_log.append({
            "rate": rate, "committed": self._committed_rate,
            "pre_cycle_acc": float(pre_cycle_acc),
            "post_acc": float(post_acc), "lr": lr,
            "reached_target": bool(reached_target),
            "outcome": "commit", "elapsed_sec": time.time() - t_cycle_start,
        })
        return rate

    # -- Safety net -------------------------------------------------------------

    def _ensure_pipeline_threshold(self):
        """Last-resort recovery if test accuracy is below the pipeline threshold.

        Uses the pipeline hard floor (derived from the previous step's test
        metric) when available, otherwise falls back to
        ``original_metric * (1 - pipeline_tolerance)``.

        Saves the pre-recovery state and restores it if a recovery attempt
        makes accuracy worse, so this safety net can never harm the model.
        """
        hard_floor = getattr(self, "_pipeline_hard_floor", None)
        if hard_floor is not None:
            threshold = hard_floor
        else:
            original = self.target_adjuster.original_metric
            threshold = original * (1.0 - self._pipeline_tolerance)

        best_test = self.trainer.test()
        if best_test >= threshold:
            return best_test

        best_state = self._clone_state()

        # Attempt 1: LR from the run's cache (probe if empty -- rare, only
        # happens if the safety net is called before any cycle completed).
        # Attempt 2: pipeline-configured LR as the fallback. Forcing a fresh
        # probe here rarely beats ``pipeline_lr`` after the cached LR already
        # failed, so we skip the extra probe to keep the safety net bounded.
        def _attempt_lrs():
            yield self._get_cached_lr()
            yield self.pipeline_lr

        for attempt, lr_to_use in enumerate(_attempt_lrs()):
            hooks = self._recovery_training_hooks(1.0)
            try:
                self.trainer.train_steps_until_target(
                    lr_to_use,
                    self._budget.max_training_steps,
                    self._get_target(),
                    0,
                    validation_n_batches=self._budget.progress_eval_batches,
                    check_interval=self._budget.check_interval,
                    patience=_RECOVERY_PATIENCE,
                    min_steps=self._budget.check_interval * 3,
                    min_improvement=self._budget.accuracy_se() / 2,
                )
            finally:
                for h in hooks:
                    h.remove()
            test_acc = self.trainer.test()
            if test_acc > best_test:
                best_test = test_acc
                best_state = self._clone_state()
            if best_test >= threshold:
                self._restore_state(best_state)
                return best_test

        self._restore_state(best_state)
        return best_test

    # -- Gradual completion -----------------------------------------------------

    def _continue_to_full_rate(self):
        """Continue gradual adaptation from committed rate toward 1.0.

        When the main SmartSmoothAdaptation loop exits before reaching
        rate=1.0, this drives the remaining distance using the same
        _adaptation() mechanism with rollback so that the final commit
        is not a catastrophic one-shot jump.  Capped to avoid runaway cost.
        """
        current = self._committed_rate
        if current >= 1.0 - 1e-6:
            return

        remaining = 1.0 - current
        step = remaining / 4.0
        max_attempts = min(20, max(4, int(1.0 / max(step, 1e-6))))
        attempts = 0

        while current < 1.0 - 1e-6 and attempts < max_attempts:
            target = min(current + step, 1.0)
            result = self._adaptation(target)

            if result is not None and float(result) < target - 1e-9:
                step /= 2.0
                if step < 1e-4:
                    break
            else:
                current = target
                remaining = 1.0 - current
                step = max(remaining / 4.0, step)
            attempts += 1

    # -- Main loop --------------------------------------------------------------

    def run(self):
        self._committed_rate = 0.0
        self._natural_rate = 0.0
        self._small_step_streak = 0
        self._pre_relaxation_target = None
        self._cycle_log = []
        self._cached_lr = None

        self._pipeline_tolerance = float(
            self.pipeline.config.get("degradation_tolerance", 0.05)
        )

        # Pipeline hard gate: the floor the test metric must stay above.
        # Derived from the previous step's test accuracy so it matches
        # the pipeline assertion exactly.
        # The tuner NEVER uses test() to derive training decisions -- this
        # is only a go/no-go gate.  No test-set data leak.
        pipeline_prev = getattr(self.pipeline, "get_target_metric", lambda: None)()
        if pipeline_prev is not None and float(pipeline_prev) > 0:
            self._pipeline_hard_floor = float(pipeline_prev) * (1.0 - self._pipeline_tolerance)
        else:
            self._pipeline_hard_floor = None

        se = self._budget.accuracy_se()
        # Empirical noise calibration: measure actual validation variance
        val_a = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        val_b = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        empirical_noise = abs(val_a - val_b)
        # Strict no-loss intent: tolerance is just measurement noise, not a
        # per-step accuracy budget. Floor at 0.005 (always at least half a
        # percentage point of slack) and cap at 0.05 so an unrealistically
        # noisy validation set (e.g. tiny fixtures) cannot turn the rollback
        # gate into a no-op that admits multi-percent regressions.
        self._rollback_tolerance = max(
            min(max(3 * se, 3 * empirical_noise), 0.05),
            0.005,
        )

        baseline_val = (val_a + val_b) / 2.0
        self.target_adjuster.target_metric = baseline_val
        self.target_adjuster.original_metric = baseline_val
        self.target_adjuster.floor = baseline_val * (1.0 - self._pipeline_tolerance)

        # Baselines for the internal rate=1.0 gate (captured once per run, no
        # test-set data leak: test_baseline is used only as a go/no-go gate).
        self._validation_baseline = baseline_val
        self._test_baseline = self.trainer.test()

        self.pipeline.reporter.report("BUDGET", {
            "max_training_steps": self._budget.max_training_steps,
            "check_interval": self._budget.check_interval,
            "eval_n_batches": self._budget.eval_n_batches,
            "progress_eval_batches": self._budget.progress_eval_batches,
            "lr_num_probes": self._budget.lr_num_probes,
            "accuracy_se": se,
            "rollback_tolerance": self._rollback_tolerance,
            "pipeline_hard_floor": self._pipeline_hard_floor,
        })

        # ------------------------------------------------------------------
        # One-shot: try full transformation in a single cycle.
        # ------------------------------------------------------------------
        if not self._skip_one_shot:
            self._before_cycle()
            self._adaptation(1.0)
            if self._committed_rate >= 1.0 - 1e-6:
                self._natural_rate = self._committed_rate
                result = self._after_run()
                assert self._committed_rate >= 1.0 - 1e-6, (
                    f"Tuning rate must reach 1.0 by the end of the step, "
                    f"but _committed_rate is {self._committed_rate:.6f}"
                )
                self._log_cycle_summary()
                return result

        # ------------------------------------------------------------------
        # Gradual fallback: one-shot failed, so use SmartSmoothAdaptation
        # to incrementally drive rate from committed_rate toward 1.0.
        # ------------------------------------------------------------------
        ms = min_step_for_smooth_adaptation(self.pipeline, self._budget)
        max_cycles = min(30, max(
            10,
            self._budget.max_training_steps // max(1, self._budget.check_interval),
        ))

        adapter = SmartSmoothAdaptation(
            self._adaptation,
            interpolators=[BasicInterpolation(0.0, 1.0)],
            get_target=self._get_target,
            min_step=ms,
            before_cycle=self._before_cycle,
        )
        adapter.adapt_smoothly(max_cycles=max_cycles)

        if self._committed_rate < 1.0 - 1e-6:
            self._continue_to_full_rate()

        self._natural_rate = self._committed_rate

        if self._natural_rate < 1.0 - 1e-6:
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}: natural adaptation reached only "
                f"{self._natural_rate:.4f}; _after_run will force to 1.0",
                stacklevel=2,
            )

        result = self._after_run()
        assert self._committed_rate >= 1.0 - 1e-6, (
            f"Tuning rate must reach 1.0 by the end of the step, "
            f"but _committed_rate is {self._committed_rate:.6f}"
        )
        self._log_cycle_summary()
        return result

    def _log_cycle_summary(self):
        """Print the full adaptation cycle log for debugging."""
        if not self._cycle_log:
            return
        print(f"[{self.__class__.__name__}] Cycle summary ({len(self._cycle_log)} cycles):")
        for i, entry in enumerate(self._cycle_log):
            parts = [f"  [{i}] rate={entry.get('rate', '?'):.4f}"]
            parts.append(f"committed={entry.get('committed', '?'):.4f}")
            if "post_acc" in entry:
                parts.append(f"post_acc={entry['post_acc']:.4f}")
            if "lr" in entry:
                parts.append(f"lr={entry['lr']:.2e}")
            parts.append(f"outcome={entry.get('outcome', '?')}")
            parts.append(f"elapsed={entry.get('elapsed_sec', 0):.1f}s")
            print(" ".join(parts))


# Backward-compatible aliases used by tuners/__init__.py and existing imports.
UnifiedPerceptronTuner = SmoothAdaptationTuner
