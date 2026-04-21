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

    # -- Absolute-floor helpers -------------------------------------------------

    def _absolute_post_acc_floor(self):
        """Baseline-anchored absolute floor for the per-cycle rollback gate.

        Returns the stricter of two floors:

        * ``_validation_baseline * (1 - degradation_tolerance)`` — the same
          relative tolerance the pipeline enforces, but anchored to the
          pre-adaptation baseline instead of ``pre_cycle_acc``. This is
          the primary guard against cumulative drift across many cycles.
        * ``_pipeline_hard_floor`` — the cross-step budget floor (see
          ``pipelining/accuracy_budget.py``). When set, it is at least as
          strict as the baseline-anchored floor and encodes the combined
          per-step + cross-step budget.

        Returns ``None`` when neither floor is available (the tuner was
        constructed without ``run()``, e.g. in unit tests). Callers fall
        back to the relative gate in that case.
        """
        baseline = getattr(self, "_validation_baseline", None)
        pipeline_floor = getattr(self, "_pipeline_hard_floor", None)
        tolerance = getattr(self, "_pipeline_tolerance", None)

        floors = []
        if baseline is not None and tolerance is not None:
            floors.append(float(baseline) * (1.0 - float(tolerance)))
        if pipeline_floor is not None:
            floors.append(float(pipeline_floor))
        if not floors:
            return None
        return max(floors)

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
        relative_threshold = pre_cycle_acc - noise_margin
        # Absolute rollback gate: anchor the floor to the baseline captured
        # once at run start, so per-cycle drops within ``noise_margin``
        # cannot compound across cycles. The effective rollback threshold
        # is the *stricter* of the two gates. When the baseline has not
        # been seeded yet (e.g. in unit tests that invoke ``_adaptation``
        # without calling ``run()``), this reduces to the relative gate.
        absolute_floor = self._absolute_post_acc_floor()
        if absolute_floor is not None:
            rollback_threshold = max(relative_threshold, absolute_floor)
        else:
            rollback_threshold = relative_threshold
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
            # Strict internal gate at full rate: post_acc (validation) must
            # stay within the noise margin of the validation baseline captured
            # once at run start. Previously this gate ran ``trainer.test()``,
            # but that leaked test-set information into the adaptation loop.
            # The pipeline-level assertion in ``_run_step`` remains the
            # authoritative cross-step test-set gate.
            val_baseline = getattr(self, "_validation_baseline", None)
            if val_baseline is not None:
                strict_threshold = float(val_baseline) - noise_margin
            else:
                strict_threshold = (
                    float(self.target_adjuster.original_metric) - noise_margin
                )
            if post_acc < strict_threshold:
                self._restore_state(pre_state)
                self._invalidate_lr_cache()
                if hasattr(self, "_cycle_log"): self._cycle_log.append({
                    "rate": rate, "committed": self._committed_rate,
                    "pre_cycle_acc": float(pre_cycle_acc),
                    "post_acc": float(post_acc),
                    "strict_threshold": strict_threshold, "lr": lr,
                    "outcome": "strict_gate_fail", "elapsed_sec": time.time() - t_cycle_start,
                })
                return self._committed_rate

        self._committed_rate = rate
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)

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
            # Clamp the decayed target to the baseline-anchored absolute
            # floor in addition to the adjuster's own ratio-based floor.
            # Without this, the tuner can relax its target below the
            # cumulative drift guard that ``_adaptation`` enforces on
            # ``post_acc``, producing a state where cycles commit while
            # simultaneously reporting "missed target".
            abs_floor = self._absolute_post_acc_floor()
            if abs_floor is not None:
                self.target_adjuster.target_metric = max(
                    self.target_adjuster.target_metric, abs_floor
                )
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

    def _attempt_recovery_if_below_floor(self):
        """Last-resort recovery when **validation** is below the pipeline floor.

        Test-set isolation: this method NEVER calls ``trainer.test()``.
        The pipeline's step-level assertion in ``Pipeline._run_step``
        (via ``PipelineStep.pipeline_metric()``) is the single place
        that reads the test set.

        Behaviour:
        - If ``_pipeline_hard_floor`` is ``None`` (floor not seeded yet),
          return the latest ``validate()`` without attempting recovery.
        - If ``validate()`` already meets the floor, return that value.
        - Otherwise, try up to two short recovery training passes. Track
          the best validation seen and restore that state at exit.
        - If the best validation is still below the floor after both
          attempts, emit a ``UserWarning`` (the pipeline's step-level
          assertion is the definitive failure gate; this warning lets
          callers see that the tuner gave up).

        Returns the best validation metric observed — a float on the
        validation scale (NOT the test scale). Downstream consumers of
        ``_final_metric`` should not treat this as a test-scale number.
        """
        hard_floor = getattr(self, "_pipeline_hard_floor", None)

        # Use the budgeted multi-batch validation. ``trainer.validate()`` is
        # a single minibatch and its variance (3σ ≈ 3.8% on MNIST's 570-sample
        # batch) far exceeds ``_rollback_tolerance``, so gating recovery on
        # it fires on noise alone. ``validate_n_batches(eval_n_batches)``
        # is the same measurement used by every rollback / commit decision
        # in ``_adaptation``.
        n_eval = self._budget.eval_n_batches
        best_val = float(self.trainer.validate_n_batches(n_eval))
        if hard_floor is None:
            return best_val
        if best_val >= hard_floor:
            return best_val

        best_state = self._clone_state()

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
            val_acc = float(self.trainer.validate_n_batches(n_eval))
            if val_acc > best_val:
                best_val = val_acc
                best_state = self._clone_state()
            if best_val >= hard_floor:
                self._restore_state(best_state)
                return best_val

        self._restore_state(best_state)
        if best_val < hard_floor:
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}: could not recover validation "
                f"above pipeline floor after retries "
                f"(best_validation={best_val:.4f}, floor={hard_floor:.4f}). "
                "The pipeline's step-level test assertion will determine "
                "whether this step fails overall.",
                stacklevel=2,
            )
        return best_val

    # Backwards-compatible alias. All internal callers should use the new
    # ``_attempt_recovery_if_below_floor`` name; this alias keeps older
    # callers (and subclasses that override it) working without a churn.
    def _ensure_pipeline_threshold(self):
        return self._attempt_recovery_if_below_floor()

    # -- Post-step stabilization ------------------------------------------------

    def _stabilization_budget(self):
        """Number of gradient steps for the final rate=1.0 stabilization pass.

        Returns ``2 * self._budget.max_training_steps`` by default. Subclasses
        can return ``None`` (or ``0``) to disable stabilization entirely —
        useful for tuners that are cheap to rerun or whose transformation is
        a no-op at rate=1.0.
        """
        return 2 * int(self._budget.max_training_steps)

    def _stabilize_at_full_rate(self):
        """Extra training at rate=1.0 after ``_after_run`` has committed.

        Runs a single ``train_steps_until_target`` call with the cached LR
        and ``2 * max_training_steps`` of budget while the rate=1.0 recovery
        hooks are installed. The trainer's built-in best-state tracking
        means this pass can only improve or preserve the model it was given
        — it cannot introduce a regression beyond measurement noise.

        Contract / preconditions:

        * Only runs when ``_committed_rate`` is already at 1.0 (it is a
          stabilization pass, not a rate transition).
        * Never calls ``trainer.test()`` (single-measurement rule).
        * Recovery hooks are installed/removed using ``try/finally`` so
          pruning / decorator invariants are enforced throughout and
          released even on exception.
        * A pre-call validation and a post-call validation are compared;
          if the post-stabilization validation drops more than
          ``_rollback_tolerance`` below pre, the pre-stabilization state
          is restored.
        """
        if self._committed_rate < 1.0 - 1e-6:
            return
        budget = self._stabilization_budget()
        if budget is None or int(budget) <= 0:
            return
        budget = int(budget)

        # Stabilization LR: the cached LR is biased toward values selected
        # for rate transitions (large deltas). At committed rate=1.0 the
        # model is already near the quantized fixed point, so use a gentler
        # LR to avoid the "oscillate around the local minimum" failure mode
        # that the single-batch pre/post comparison used to mask as a
        # rollback. Clamp at ``pipeline_lr`` so a coarser cached LR cannot
        # run away here.
        lr = min(float(self._get_cached_lr()), float(self.pipeline_lr))

        n_eval = self._budget.eval_n_batches
        pre_state = self._clone_state()
        try:
            # Multi-batch validation: single-batch ``trainer.validate()`` has
            # 3σ ≈ 3.8% on MNIST (SE = 0.013 on 570 samples), which exceeds
            # ``_rollback_tolerance`` and was triggering noise-driven
            # rollbacks that threw away genuine improvements from the
            # stabilization pass.
            pre_val = float(self.trainer.validate_n_batches(n_eval))
        except Exception:
            pre_val = None

        # Stabilization wants the full budget: best-state tracking inside
        # ``train_steps_until_target`` already guards against regression, so
        # patience-based early-stop here only truncates slow-but-real
        # consolidation. Aggressive quantization (e.g. target_tq=4 on MNIST)
        # plateaus validation at a noise band ~0.01 wide, which exceeds
        # ``min_improvement``; with the default patience of 5 the pass
        # exits after ~100 effective steps and a ~3% gap is left on the
        # table. Raising patience to budget/check_interval (i.e. "never
        # patience-exit") turns stabilization into "use the whole budget
        # and restore the peak", which is what the invariant actually
        # promises. Use ``eval_n_batches`` (rollback-grade) for the per-
        # interval check so best-state selection doesn't latch onto a
        # single lucky 1-batch reading.
        progress_n = max(
            self._budget.progress_eval_batches,
            self._budget.eval_n_batches,
        )
        max_patience = max(_RECOVERY_PATIENCE, budget // max(1, self._budget.check_interval))
        hooks = self._recovery_training_hooks(1.0)
        try:
            self.trainer.train_steps_until_target(
                lr,
                budget,
                self._get_target(),
                0,
                validation_n_batches=progress_n,
                check_interval=self._budget.check_interval,
                patience=max_patience,
                min_steps=budget,
                min_improvement=self._budget.accuracy_se() / 2,
            )
        finally:
            for h in hooks:
                h.remove()

        if pre_val is not None:
            try:
                post_val = float(self.trainer.validate_n_batches(n_eval))
            except Exception:
                post_val = None
            if post_val is not None and post_val < pre_val - self._rollback_tolerance:
                self._restore_state(pre_state)
                self._invalidate_lr_cache()
                self.pipeline.reporter.report(
                    f"{self.name} stabilization rollback",
                    {"pre": pre_val, "post": post_val},
                )

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
        # Routed through ``pipeline.accuracy_budget`` so the floor is
        # ``max(previous_test * (1 - tolerance), reference - budget_total)``:
        # per-step and cross-step budgets combined.
        #
        # Before the budget is seeded (no positive test metric observed
        # yet -- e.g. Architecture Search / Model Building / Model
        # Configuration), the floor is left as ``None`` and the tuner
        # skips the pipeline-facing assertion; no test-set data leak
        # because the floor is derived only from measurements the
        # pipeline has already committed at previous-step exit.
        pipeline_prev = getattr(self.pipeline, "get_target_metric", lambda: None)()
        budget = getattr(self.pipeline, "accuracy_budget", None)
        if (
            pipeline_prev is not None
            and float(pipeline_prev) > 0
            and budget is not None
        ):
            self._pipeline_hard_floor = budget.step_floor(
                previous_metric=float(pipeline_prev),
                per_step_tolerance=self._pipeline_tolerance,
            )
            if self._pipeline_hard_floor <= 0.0:
                # Budget unseeded -> no cross-step floor; fall back to the
                # per-step floor so we retain the previous behavior.
                self._pipeline_hard_floor = float(pipeline_prev) * (
                    1.0 - self._pipeline_tolerance
                )
        elif pipeline_prev is not None and float(pipeline_prev) > 0:
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

        # Validation baseline for the internal rate=1.0 gate. The tuner
        # NEVER calls ``trainer.test()``; the pipeline's centralised
        # ``PipelineStep.pipeline_metric()`` runs the one-and-only test()
        # pass per step at step exit.
        self._validation_baseline = baseline_val

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
                self._stabilize_at_full_rate()
                self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
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
        self._stabilize_at_full_rate()
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
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
