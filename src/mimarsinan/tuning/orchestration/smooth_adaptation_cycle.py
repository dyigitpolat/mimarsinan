"""Smooth adaptation cycle, recovery, and rollback logic."""

from __future__ import annotations

import time
import warnings

from mimarsinan.tuning.trace import DecisionRecord, DecisionTrace
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor
from mimarsinan.tuning.orchestration.adaptation_driver import AdaptationDriver, CycleContext
from mimarsinan.tuning.orchestration.checkpoint_guard import CheckpointGuard
from mimarsinan.tuning.orchestration.ft_pass_wall import FtPassWallLog
from mimarsinan.tuning.orchestration.recovery_engine import (
    PERSIST_WITHIN_CYCLE,
    RESET_PER_CYCLE,
    PersistentOptimizerOwner,
    RecoveryEngine,
)
from mimarsinan.tuning.orchestration.tuner_base import (
    TunerBase,
    _RECOVERY_PATIENCE,
    _STUCK_STREAK_REQUIRED,
)


class SmoothAdaptationCycleMixin(TunerBase):
    """Adaptation cycle, rollback, and recovery training."""

    def __init__(self, pipeline, model, target_accuracy, lr):
        super().__init__(pipeline, model, target_accuracy, lr)
        self._checkpoint_guard = CheckpointGuard(
            self.trainer,
            scope=pipeline.config.get("checkpoint_scope", "full"),
            location=pipeline.config.get("checkpoint_location", "device"),
        )
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
        self._refind_lr_on_miss = bool(
            pipeline.config.get("tuning_refind_lr_on_miss", False)
        )
        self._recovery_lr_plateau = bool(
            pipeline.config.get("tuning_recovery_lr_plateau", False)
        )
        self._recovery_lr_plateau_factor = float(
            pipeline.config.get("tuning_recovery_lr_plateau_factor", 0.3)
        )
        self._recovery_lr_plateau_reductions = int(
            pipeline.config.get("tuning_recovery_lr_plateau_reductions", 2)
        )
        self._rollback_ratchet = bool(
            pipeline.config.get("tuning_rollback_ratchet", False)
        )
        self._rollback_cumulative_bound = float(
            pipeline.config.get("tuning_rollback_cumulative_bound", 0.05)
        )
        self._best_committed_acc = None
        self._tight_plateau = bool(
            pipeline.config.get("tuning_tight_plateau", False)
        )
        self._recovery_check_divisor = max(
            1, int(pipeline.config.get("tuning_recovery_check_divisor", 1))
        )
        self._keepbest_certified = bool(
            pipeline.config.get("tuning_keepbest_certified", False)
        )
        self._best_committed_state = None
        self._best_committed_metric = None
        self._stabilization_bounded = bool(
            pipeline.config.get("tuning_stabilization_bounded", False)
        )
        self._stabilization_ratio = float(
            pipeline.config.get("tuning_stabilization_ratio", 0.5)
        )
        self._gradual_train_steps = 0
        self._cycle_log = DecisionTrace.new()
        self._cached_lr = None
        self._paired_gate = bool(pipeline.config.get("tuning_use_paired_sensor", False))
        self._k_commit = float(pipeline.config.get("k_commit", 2.0))
        self._global_budget = float(pipeline.config.get("global_budget", 0.0))
        if self._global_budget < 0.0:
            raise ValueError(
                f"global_budget must be non-negative; got {self._global_budget}. "
                "0.0 disables the §8.2 floor (best-accuracy paired setting); a "
                "positive value enables anti-thrash protection."
            )
        self._confirm_indices = None
        self._ref_correct = None
        self._full_transform_probe = bool(
            pipeline.config.get("tuning_full_transform_probe", False)
        )
        self._full_transform_log = []
        self.trainer._val_cache_max_batches = self._budget.eval_n_batches
        self.trainer._gpu_val_cache = None

    def _update_and_evaluate(self, rate):
        """Apply transformation T at *rate* and return a validation metric."""
        raise NotImplementedError

    def _before_cycle(self):
        """Called before each adaptation cycle (e.g. refresh pruning importance)."""

    def _after_run(self):
        """Called after adaptation completes. Returns the final metric."""
        return self.trainer.validate()

    def _recovery_training_hooks(self, rate):
        """Hooks to keep active during recovery training (removed when it finishes).

        Subclasses override this to enforce invariants (e.g. pruning masks).
        """
        return []

    def _get_extra_state(self):
        """Override to return tuner-specific state to save alongside model params."""
        return None

    def _set_extra_state(self, extra):
        """Override to restore tuner-specific state."""

    def _ft_pass_wall_log(self) -> FtPassWallLog:
        """The per-fine-tuning-PASS wall log (AC5), lazily created per run.

        Keyed to the run's ``_run_t0`` epoch so a re-run on the same instance starts
        a fresh log instead of accumulating across runs."""
        run_epoch = getattr(self, "_run_t0", None)
        log = getattr(self, "_ft_wall_log", None)
        if log is None or getattr(self, "_ft_wall_log_epoch", None) != run_epoch:
            log = FtPassWallLog()
            self._ft_wall_log = log
            self._ft_wall_log_epoch = run_epoch
        return log

    @property
    def ft_pass_walls(self) -> list:
        """The per-FT-pass wall breakdown (``[{"label", "wall_s"}, ...]``)."""
        return self._ft_pass_wall_log().passes

    @property
    def max_ft_pass_wall_s(self) -> float:
        """The worst single fine-tuning pass wall in seconds (AC5's number); 0.0 when
        no FT pass ran — a well-defined float, never ``None``."""
        return self._ft_pass_wall_log().max_wall_s

    def ft_pass_wall_metrics(self) -> dict:
        """The AC5 metric bundle the run can persist alongside its artifacts: the
        max single-pass wall (the verdict number) + the per-pass breakdown."""
        return {
            "max_ft_pass_wall_s": self.max_ft_pass_wall_s,
            "passes": self.ft_pass_walls,
        }

    def _clone_state(self):
        return (self._checkpoint_guard.snapshot(), self._get_extra_state())

    def _restore_state(self, state):
        model_state, extra = state
        self._checkpoint_guard.restore(model_state)
        if extra is not None:
            self._set_extra_state(extra)

    def _baseline_or_none(self):
        baseline = getattr(self, "_validation_baseline", None)
        return float(baseline) if baseline is not None else None

    def _absolute_post_acc_floor(self):
        """Baseline-anchored absolute floor for the per-cycle rollback gate."""
        return AcceptanceSensor.absolute_floor(
            getattr(self, "_validation_baseline", None),
            getattr(self, "_pipeline_tolerance", None),
            getattr(self, "_pipeline_hard_floor", None),
        )

    def _recovery_plateau_kwargs(self):
        """Plateau LR-reduction kwargs for the recovery call: the configured
        factor/reductions when ``tuning_recovery_lr_plateau`` is on, else the
        no-op defaults (1.0/0 → break-on-patience, unchanged)."""
        if getattr(self, "_recovery_lr_plateau", False):
            return (
                float(getattr(self, "_recovery_lr_plateau_factor", 0.3)),
                int(getattr(self, "_recovery_lr_plateau_reductions", 2)),
            )
        return 1.0, 0

    def _ratchet_best_anchor(self):
        """Best-committed high-water mark for the non-stalling cumulative-drift
        bound, falling back to the validation baseline before any commit, and
        ``None`` when neither is available (the relative gate then stands alone)."""
        best = getattr(self, "_best_committed_acc", None)
        if best is not None:
            return float(best)
        baseline = getattr(self, "_validation_baseline", None)
        if baseline is not None:
            return float(baseline)
        return None

    def _recovery_check_interval(self):
        """Recovery check interval, tightened by ``tuning_recovery_check_divisor``
        when ``tuning_tight_plateau`` is on (CHANGE 3); else the budget interval."""
        interval = int(self._budget.check_interval)
        if getattr(self, "_tight_plateau", False):
            divisor = max(1, int(getattr(self, "_recovery_check_divisor", 1)))
            return max(1, interval // divisor)
        return interval

    def _accumulate_gradual_steps(self, recovery_result):
        """Add the step count a gradual recovery call reported to the running total.
        ``recovery_result`` is ``(accuracy, steps)`` only when ``return_steps`` was
        requested; otherwise the bare accuracy contributes nothing."""
        if isinstance(recovery_result, tuple) and len(recovery_result) == 2:
            self._gradual_train_steps += int(recovery_result[1])

    def _update_best_committed_acc(self, post_acc):
        """Tighten the rollback high-water mark on each commit (never lowers it)."""
        if not getattr(self, "_rollback_ratchet", False):
            return
        best = getattr(self, "_best_committed_acc", None)
        self._best_committed_acc = (
            float(post_acc) if best is None else max(float(best), float(post_acc))
        )

    def _certified_gate_metric(self, paired_post_acc=None):
        """The keep-best gate metric, reading the same basis the commit gate uses:
        the paired correctness fraction when ``tuning_use_paired_sensor`` is on, else
        ``probe()``. ``paired_post_acc`` reuses a commit-time fraction; ``None`` forces
        a fresh read."""
        if getattr(self, "_paired_gate", False) and self._ref_correct is not None:
            if paired_post_acc is not None:
                return float(paired_post_acc)
            cand = self.trainer.validate_correctness_on_indices(self._confirm_indices)
            return (sum(1 for c in cand if c) / len(cand)) if cand else 0.0
        return float(self.probe())

    def _snapshot_best_committed_if_improved(self, post_acc):
        """On a commit whose gate metric strictly beats the running best, clone the
        model state into ``_best_committed_state``. No-op unless
        ``tuning_keepbest_certified`` is on."""
        if not getattr(self, "_keepbest_certified", False):
            return
        metric = self._certified_gate_metric(paired_post_acc=post_acc)
        best = getattr(self, "_best_committed_metric", None)
        if best is None or metric > best:
            self._best_committed_metric = float(metric)
            self._best_committed_state = self._clone_state()

    _supports_persistent_optimizer = True

    def _optimizer_policy(self):
        """Persist optimizer moments across recovery within a cycle. Families whose
        recovery replaces the parameter set each cycle
        (``_supports_persistent_optimizer = False``) always reset."""
        if not getattr(self, "_supports_persistent_optimizer", True):
            return RESET_PER_CYCLE
        return PERSIST_WITHIN_CYCLE

    def _recovery_optimizer(self, lr):
        """Owned optimizer for the persist policy, else ``None`` (bit-exact path)."""
        if self._optimizer_policy() != PERSIST_WITHIN_CYCLE:
            return None
        owner = getattr(self, "_persistent_optimizer_owner", None)
        if owner is None:
            owner = PersistentOptimizerOwner(self.trainer)
            self._persistent_optimizer_owner = owner
        return owner.optimizer_for(lr)

    def _adaptation(self, rate):
        """Recovery training at a given rate with rollback on regression.

        The predictor → corrector → commit/rollback flow is owned by
        ``AdaptationDriver.run_cycle``; this tuner supplies the phase operations.
        """
        return AdaptationDriver.run_cycle(self, rate)

    def _begin_cycle(self, rate) -> CycleContext:
        """Snapshot pre-state and the pre-cycle baseline; report the cycle start."""
        t_cycle_start = time.time()
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())

        pre_state = self._clone_state()
        last = getattr(self, "_last_post_acc", None)
        pre_cycle_acc = float(last) if last is not None else self.probe()
        return CycleContext(
            rate=float(rate),
            t_cycle_start=t_cycle_start,
            pre_state=pre_state,
            pre_cycle_acc=pre_cycle_acc,
        )

    def _probe_instant(self, ctx: CycleContext) -> None:
        """Apply the transform and read the pre-recovery (instant) accuracy."""
        with self.trainer.validation_context("probe"):
            ctx.instant_acc = self.ramp(ctx.rate)
        ctx.is_catastrophic = AcceptanceSensor.is_catastrophic(
            ctx.instant_acc, self._get_target()
        )

    def _recover_to_target(self, target, rate):
        """The corrector primitive: find the cached LR and run recovery training
        toward ``target`` at ramp ``rate``. Returns ``(lr, result)``. SSOT for the
        per-cycle ``_recover`` and the ``recover_to`` seam verb.
        """
        t0 = time.time()
        lr = self._get_cached_lr()
        t_lr = time.time() - t0
        self.pipeline.reporter.report("LR_found", lr)
        self.pipeline.reporter.report("T_find_lr_sec", t_lr)

        hooks = self._recovery_training_hooks(rate)
        plateau_factor, plateau_reductions = self._recovery_plateau_kwargs()
        check_interval = self._recovery_check_interval()
        with self._ft_pass_wall_log().time_pass("recover"):
            recovery_result = RecoveryEngine.train_to_target(
                self.trainer,
                lr,
                target,
                max_steps=self._budget.max_training_steps,
                hooks=hooks,
                optimizer=self._recovery_optimizer(lr),
                validation_n_batches=self._budget.progress_eval_batches,
                check_interval=check_interval,
                patience=_RECOVERY_PATIENCE,
                min_steps=check_interval * 3,
                min_improvement=self._budget.accuracy_se(),
                plateau_lr_factor=plateau_factor,
                plateau_lr_reductions=plateau_reductions,
                return_steps=getattr(self, "_stabilization_bounded", False),
            )
        self._accumulate_gradual_steps(recovery_result)
        return lr, recovery_result

    def _recover(self, ctx: CycleContext) -> None:
        """Find the LR and run recovery training toward the target (through the
        ``recover_to`` seam verb, reading the discovered LR back for the trace)."""
        self.recover_to(self._get_target(), rate=ctx.rate)
        ctx.lr = self._last_recover_lr

    def _measure_post(self, ctx: CycleContext) -> None:
        """Post-recovery accuracy + the rollback decision (marginal or paired)."""
        ctx.noise_margin = self._rollback_tolerance
        ctx.absolute_floor = self._absolute_post_acc_floor()
        if getattr(self, "_rollback_ratchet", False):
            ctx.rollback_threshold = AcceptanceSensor.ratchet_threshold(
                ctx.pre_cycle_acc,
                ctx.noise_margin,
                self._ratchet_best_anchor(),
                float(getattr(self, "_rollback_cumulative_bound", 0.05)),
                ctx.absolute_floor,
            )
        else:
            ctx.rollback_threshold = AcceptanceSensor.rollback_threshold(
                ctx.pre_cycle_acc, ctx.noise_margin, ctx.absolute_floor
            )
        if getattr(self, "_paired_gate", False) and self._ref_correct is not None:
            cand_correct = self.trainer.validate_correctness_on_indices(self._confirm_indices)
            ctx.cand_correct = cand_correct
            ctx.post_acc = (
                sum(1 for c in cand_correct if c) / len(cand_correct)
            ) if cand_correct else 0.0
            ctx.rolled_back = AcceptanceSensor.paired_is_rollback(
                self._ref_correct, cand_correct, self._k_commit,
                min_effect=self._global_budget,
            )
        else:
            ctx.post_acc = self.probe()
            ctx.rolled_back = AcceptanceSensor.is_rollback(ctx.post_acc, ctx.rollback_threshold)

    def _rollback_cycle(self, ctx: CycleContext, outcome: str):
        """Restore the pre-cycle state and record the (catastrophic|rollback) trace."""
        self._restore_state(ctx.pre_state)
        if hasattr(self, "_cycle_log"):
            if outcome == "catastrophic":
                self._cycle_log.record(DecisionRecord(
                    cycle_index=len(self._cycle_log),
                    outcome="catastrophic",
                    rate=float(ctx.rate),
                    committed=float(self._committed_rate),
                    elapsed_sec=time.time() - ctx.t_cycle_start,
                    instant_acc=float(ctx.instant_acc),
                    pre_cycle_acc=float(ctx.pre_cycle_acc),
                    target=float(self._get_target()),
                    validation_baseline=self._baseline_or_none(),
                    rollback_tolerance=float(self._rollback_tolerance),
                ))
            else:
                self._cycle_log.record(DecisionRecord(
                    cycle_index=len(self._cycle_log),
                    outcome="rollback",
                    rate=float(ctx.rate),
                    committed=float(self._committed_rate),
                    elapsed_sec=time.time() - ctx.t_cycle_start,
                    instant_acc=float(ctx.instant_acc) if ctx.instant_acc is not None else None,
                    pre_cycle_acc=float(ctx.pre_cycle_acc),
                    post_acc=float(ctx.post_acc),
                    lr=float(ctx.lr),
                    target=float(self._get_target()),
                    validation_baseline=self._baseline_or_none(),
                    rollback_threshold=float(ctx.rollback_threshold),
                    absolute_floor=(float(ctx.absolute_floor) if ctx.absolute_floor is not None else None),
                    rollback_tolerance=float(ctx.noise_margin),
                ))
        return self._committed_rate

    def _commit_cycle(self, ctx: CycleContext):
        """Commit the rate, run the stuck-streak target relaxation, record + probe."""
        self._committed_rate = ctx.rate
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self._update_best_committed_acc(ctx.post_acc)
        self._snapshot_best_committed_if_improved(ctx.post_acc)

        reached_target = AcceptanceSensor.reached_target(
            ctx.post_acc, self._get_target(), ctx.noise_margin
        )
        if reached_target:
            self._missed_target_streak = 0
            pre_relax = getattr(self, "_pre_relaxation_target", None)
            if pre_relax is not None:
                self.target_adjuster.target_metric = pre_relax
                self._pre_relaxation_target = None
        else:
            self._missed_target_streak += 1
            if getattr(self, "_refind_lr_on_miss", False):
                self._invalidate_lr_cache()

        if self._missed_target_streak >= _STUCK_STREAK_REQUIRED:
            self._pre_relaxation_target = self._get_target()
            self.target_adjuster.update_target(ctx.post_acc)
            abs_floor = self._absolute_post_acc_floor()
            if abs_floor is not None:
                self.target_adjuster.target_metric = max(
                    self.target_adjuster.target_metric, abs_floor
                )
            self._missed_target_streak = 0
            if not getattr(self, "_refind_lr_on_miss", False):
                self._invalidate_lr_cache()

        if hasattr(self, "_cycle_log"): self._cycle_log.record(DecisionRecord(
            cycle_index=len(self._cycle_log),
            outcome="commit",
            rate=float(ctx.rate),
            committed=float(self._committed_rate),
            elapsed_sec=time.time() - ctx.t_cycle_start,
            pre_cycle_acc=float(ctx.pre_cycle_acc),
            post_acc=float(ctx.post_acc),
            lr=float(ctx.lr),
            reached_target=bool(reached_target),
            target=float(self._get_target()),
            validation_baseline=self._baseline_or_none(),
            rollback_threshold=float(ctx.rollback_threshold),
            absolute_floor=(float(ctx.absolute_floor) if ctx.absolute_floor is not None else None),
            rollback_tolerance=float(ctx.noise_margin),
        ))
        self._last_post_acc = float(ctx.post_acc)
        self._probe_full_transform(float(ctx.post_acc))
        return ctx.rate

    def _value_full_transform_eval(self):
        """The value-domain rate-1.0 accuracy from the committed state: clone, apply
        T at 1.0, evaluate, restore. Non-destructive."""
        pre = self._clone_state()
        try:
            return float(self._update_and_evaluate(1.0))
        finally:
            self._restore_state(pre)

    def _full_transform_eval(self):
        """The GENUINE full-transform accuracy, non-destructive. Base default is the
        value-domain measurement; ``KDBlendAdaptationTuner`` overrides it to run the
        deployed forward on an isolated clone."""
        return self._value_full_transform_eval()

    def _probe_full_transform(self, committed_acc):
        """Diagnostic: per-commit, measure the value-domain and GENUINE rate-1.0
        accuracies; ``proxy_gap = value_full_acc - genuine_full_acc`` surfaces the
        deployed cliff the value-domain proxy hides."""
        if not getattr(self, "_full_transform_probe", False):
            return
        if self._committed_rate >= 1.0 - 1e-6:
            return
        value_full_acc = self._value_full_transform_eval()
        genuine_full_acc = self._full_transform_eval()
        value_drop = committed_acc - value_full_acc
        genuine_drop = committed_acc - genuine_full_acc
        proxy_gap = value_full_acc - genuine_full_acc
        self._full_transform_log.append({
            "committed": float(self._committed_rate),
            "committed_acc": committed_acc,
            "value_full_acc": value_full_acc,
            "genuine_full_acc": genuine_full_acc,
            "value_drop": value_drop,
            "genuine_drop": genuine_drop,
            "proxy_gap": proxy_gap,
            "full_acc": value_full_acc,
            "drop": value_drop,
        })
        print(
            f"[{self.__class__.__name__}] FULL_TRANSFORM_PROBE "
            f"α={self._committed_rate:.4f} committed_acc={committed_acc:.4f} "
            f"value_full_acc={value_full_acc:.4f} genuine_full_acc={genuine_full_acc:.4f} "
            f"value_drop={value_drop:+.4f} genuine_drop={genuine_drop:+.4f} "
            f"proxy_gap={proxy_gap:+.4f}"
        )
        self.pipeline.reporter.report("FULL_TRANSFORM_PROBE", {
            "committed": round(float(self._committed_rate), 4),
            "committed_acc": round(committed_acc, 4),
            "value_full_acc": round(value_full_acc, 4),
            "genuine_full_acc": round(genuine_full_acc, 4),
            "value_drop": round(value_drop, 4),
            "genuine_drop": round(genuine_drop, 4),
            "proxy_gap": round(proxy_gap, 4),
            "full_acc": round(value_full_acc, 4),
            "drop": round(value_drop, 4),
        })

    def _attempt_recovery_if_below_floor(self):
        """Last-resort recovery when validation is below the pipeline floor."""
        hard_floor = getattr(self, "_pipeline_hard_floor", None)

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
            with self._ft_pass_wall_log().time_pass("below_floor_recover"):
                RecoveryEngine.train_to_target(
                    self.trainer,
                    lr_to_use,
                    self._get_target(),
                    max_steps=self._budget.max_training_steps,
                    hooks=hooks,
                    optimizer=self._recovery_optimizer(lr_to_use),
                    validation_n_batches=self._budget.progress_eval_batches,
                    check_interval=self._budget.check_interval,
                    patience=_RECOVERY_PATIENCE,
                    min_steps=self._budget.check_interval * 3,
                    min_improvement=self._budget.accuracy_se() / 2,
                )
            val_acc = float(self.trainer.validate_n_batches(n_eval))
            if val_acc > best_val:
                best_val = val_acc
                best_state = self._clone_state()
            if best_val >= hard_floor:
                self._restore_state(best_state)
                return best_val

        self._restore_state(best_state)
        if best_val < hard_floor:
            warnings.warn(
                f"{self.__class__.__name__}: could not recover validation "
                f"above pipeline floor after retries "
                f"(best_validation={best_val:.4f}, floor={hard_floor:.4f}). "
                "The pipeline's step-level test assertion will determine "
                "whether this step fails overall.",
                stacklevel=2,
            )
        return best_val

    def _ensure_pipeline_threshold(self):
        return self._attempt_recovery_if_below_floor()

