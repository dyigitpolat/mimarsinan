"""Smooth adaptation run loop and stabilization."""

from __future__ import annotations

import contextlib
import time
import warnings
from typing import TYPE_CHECKING, Any

from mimarsinan.tuning.trace import DecisionTrace
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor
from mimarsinan.tuning.orchestration.adaptation_driver import AdaptationDriver
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.tuning_budget import min_step_for_smooth_adaptation
from mimarsinan.tuning.orchestration.tuner_base import (
    TunerBase,
    _RECOVERY_PATIENCE,
)


class SmoothAdaptationRunMixin(TunerBase):
    if TYPE_CHECKING:
        # Host contract: cycle-scope services come from SmoothAdaptationCycleMixin
        # in the composed SmoothAdaptationTuner.
        def _clone_state(self) -> Any: ...
        def _restore_state(self, state) -> None: ...
        def _recovery_training_hooks(self, rate) -> list: ...
        def _adaptation(self, rate) -> Any: ...
        def _before_cycle(self) -> None: ...
        def _after_run(self) -> Any: ...
        def _certified_gate_metric(self, paired_post_acc=None) -> float: ...

    def _stabilization_budget(self):
        """Number of gradient steps for the final rate=1.0 stabilization pass
        (per round; ``_max_stabilization_rounds`` controls the round count).
        Subclasses may override (e.g. return ``None``/``0`` to disable)."""
        return 2 * int(self._budget.max_training_steps)

    def _post_stabilization_hook(self):
        """Called once after the final stabilization pass; subclasses may run
        deployment-aware post-training calibration here (it must only commit
        validation-improving changes — no training follows to undo them)."""

    def _stabilization_lr(self):
        """The stabilization pass LR: optionally re-found, capped at the pipeline
        LR; ``None`` = the sweep refused and the optional pass is skipped (fix C)."""
        if getattr(self, "_stabilization_refinds_lr", False):
            self._invalidate_lr_cache()
        return self._capped_cached_lr()

    @contextlib.contextmanager
    def _stabilization_rollback_guard(self):
        """Pre/post validation bracket shared by every stabilization shape:
        restore + report when the pass ends below entry - tolerance
        (non-destructive). Yields the entry validation read."""
        n_eval = self._budget.eval_n_batches
        pre_state = self._clone_state()
        pre_val = float(self.trainer.validate_n_batches(n_eval))
        yield pre_val
        post_val = float(self.trainer.validate_n_batches(n_eval))
        if post_val < pre_val - self._rollback_tolerance:
            self._restore_state(pre_state)
            self._invalidate_lr_cache()
            self.pipeline.reporter.report(
                f"{self.name} stabilization rollback",
                {"pre": pre_val, "post": post_val},
            )

    def _stabilize_at_full_rate(self):
        """Extra training at rate=1.0 after ``_after_run`` has committed.

        Runs up to ``_max_stabilization_rounds`` passes, each restarting from a fresh
        LR while validation still improves by more than ``accuracy_se()/2``. The
        pre/post rollback guard brackets all rounds (non-destructive).
        """
        if self._committed_rate < 1.0 - 1e-6:
            return
        budget = self._stabilization_budget()
        if budget is None or int(budget) <= 0:
            return
        if getattr(self, "_stabilization_bounded", False):
            self._stabilize_bounded_cosine()
            return
        budget = int(budget)

        lr = self._stabilization_lr()
        if lr is None:
            return  # [LR-REFUSE] optional stabilization is skipped (fix C)

        n_eval = self._budget.eval_n_batches
        progress_n = max(
            self._budget.progress_eval_batches,
            self._budget.eval_n_batches,
        )
        max_patience = max(_RECOVERY_PATIENCE, budget // max(1, self._budget.check_interval))
        rounds = max(1, int(getattr(self, "_max_stabilization_rounds", 1)))
        with self._stabilization_rollback_guard() as pre_val:
            last_val = pre_val
            for round_idx in range(rounds):
                hooks = self._recovery_training_hooks(1.0)
                RecoveryEngine.train_to_target(
                    self.trainer,
                    lr,
                    self._get_target(),
                    max_steps=budget,
                    validation_n_batches=progress_n,
                    check_interval=self._budget.check_interval,
                    patience=max_patience,
                    min_steps=budget,
                    min_improvement=self._budget.accuracy_se() / 2,
                    hooks=hooks,
                    final_validation=False,
                )
                if round_idx == rounds - 1:
                    break
                round_val = float(self.trainer.validate_n_batches(n_eval))
                if round_val - last_val <= self._budget.accuracy_se() / 2:
                    break
                last_val = round_val
                self._invalidate_lr_cache()
                lr = self._capped_cached_lr()
                if lr is None:
                    break  # [LR-REFUSE] no further stabilization rounds (fix C)

    def _stabilize_bounded_cosine(self):
        """A single bounded stabilization pass of ``N = int(stabilization_ratio *
        gradual_train_steps)`` steps with a cosine-decay LR. Hard cutoff, bracketed
        by the same pre/post rollback guard (non-destructive)."""
        ratio = float(getattr(self, "_stabilization_ratio", 0.5))
        gradual_steps = int(getattr(self, "_gradual_train_steps", 0))
        n_steps = int(ratio * gradual_steps)
        if n_steps <= 0:
            return

        lr = self._stabilization_lr()
        if lr is None:
            return  # [LR-REFUSE] optional stabilization is skipped (fix C)

        with self._stabilization_rollback_guard():
            hooks = self._recovery_training_hooks(1.0)
            # check_interval = n_steps makes the only validation the final step, so
            # the unreachable target cannot break the run early: exactly N steps run.
            RecoveryEngine.train_to_target(
                self.trainer,
                lr,
                self._get_target() + 1.0,
                max_steps=n_steps,
                validation_n_batches=self._budget.progress_eval_batches,
                check_interval=n_steps,
                patience=1,
                min_steps=n_steps,
                min_improvement=self._budget.accuracy_se() / 2,
                hooks=hooks,
                cosine_decay=True,
                final_validation=False,
            )

    def _continue_to_full_rate(self):
        """Continue gradual adaptation from committed rate toward 1.0."""
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

    def run(self):
        self._committed_rate = 0.0
        self._natural_rate = 0.0
        self._small_step_streak = 0
        self._pre_relaxation_target = None
        self._best_committed_acc = None
        self._best_committed_state = None
        self._best_committed_metric = None
        self._gradual_train_steps = 0
        self._cycle_log = DecisionTrace.new()
        self._cached_lr = None
        self._persistent_optimizer_owner = None
        self._phase_seconds = {}
        self._run_t0 = time.time()

        self._pipeline_tolerance = float(
            self.pipeline.config.get("degradation_tolerance", 0.05)
        )

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
                self._pipeline_hard_floor = float(pipeline_prev) * (
                    1.0 - self._pipeline_tolerance
                )
        elif pipeline_prev is not None and float(pipeline_prev) > 0:
            self._pipeline_hard_floor = float(pipeline_prev) * (1.0 - self._pipeline_tolerance)
        else:
            self._pipeline_hard_floor = None

        ref = AcceptanceSensor(self._budget).calibrate_baseline(
            self.trainer.validate_n_batches, self._budget.eval_n_batches
        )
        se = ref.se
        self._rollback_tolerance = ref.rollback_tolerance

        baseline_val = ref.baseline
        self._anchor_relaxation_target(baseline_val, pipeline_prev)

        self._validation_baseline = baseline_val

        if getattr(self, "_paired_gate", False):
            n_conf = int(self.pipeline.config.get("paired_confirm_batches", 0)) \
                or self._budget.eval_n_batches
            self._confirm_indices = list(range(int(n_conf)))
            self._ref_correct = self.trainer.validate_correctness_on_indices(
                self._confirm_indices
            )

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

        return self._run_with_scheduler()

    def _anchor_relaxation_target(self, baseline_val, real_target):
        """Set the missed-target relaxation anchor (``target_adjuster``) and its floor.

        Default: anchor target/original/floor on the rate-0 ``baseline_val``. With
        ``tuning_target_floor_on_real_target`` on and the real target above the
        collapsed baseline, keep the real-target anchor and cap the floor at
        ``max(baseline_floor, real_target * (1 - tol))``.
        """
        tol = self._pipeline_tolerance
        baseline_floor = baseline_val * (1.0 - tol)
        if (
            bool(self.pipeline.config.get("tuning_target_floor_on_real_target", False))
            and real_target is not None
            and float(real_target) > float(baseline_val)
        ):
            real = float(real_target)
            self.target_adjuster.target_metric = real
            self.target_adjuster.original_metric = real
            self.target_adjuster.floor = min(
                real, max(baseline_floor, real * (1.0 - tol))
            )
            return
        self.target_adjuster.target_metric = baseline_val
        self.target_adjuster.original_metric = baseline_val
        self.target_adjuster.floor = baseline_floor

    def _run_with_scheduler(self):
        """The single rate-search loop: greedy-to-1.0 + bisect for the standard axes,
        a uniform ladder for the ``_skip_one_shot`` (KD-blend) family. The shared
        ``_finalize_run`` tail forces the rate to 1.0 and stabilizes.
        """
        ms = min_step_for_smooth_adaptation(self.pipeline, self._budget)
        max_cycles = min(30, max(
            10,
            self._budget.max_training_steps // max(1, self._budget.check_interval),
        ))
        initial_step = float(getattr(self, "_initial_ramp_step", 0.5))
        epsilon = ms
        policy_override = None
        rates = None
        if getattr(self, "_fixed_ladder_policy", False):
            policy_override = "fixed_ladder"
            rates = getattr(self, "_fixed_ladder_rates", None) or [1.0]
        scheduler = AdaptationDriver.build_scheduler(
            epsilon=epsilon,
            max_rounds=max_cycles,
            skip_one_shot=getattr(self, "_skip_one_shot", False),
            initial_step=initial_step,
            policy_override=policy_override,
            rates=rates,
        )
        driver = AdaptationDriver(
            scheduler=scheduler,
            attempt=self._driver_attempt,
            finalize=self._finalize_run,
            committed=self._committed_rate,
        )
        return driver.run()

    def _driver_attempt(self, target):
        """One scheduler attempt: refresh per-cycle state, then run a cycle."""
        self._before_cycle()
        return self._adaptation(target)

    def _finalize_run(self):
        """Shared tail: force rate→1.0 via ``_after_run``, then stabilize."""
        self._natural_rate = self._committed_rate
        self._phase_seconds["gradual"] = time.time() - self._run_t0

        if self._natural_rate < 1.0 - 1e-6:
            warnings.warn(
                f"{self.__class__.__name__}: natural adaptation reached only "
                f"{self._natural_rate:.4f}; _after_run will force to 1.0",
                stacklevel=2,
            )

        t_after = time.time()
        result = self._after_run()
        self._phase_seconds["after_run"] = time.time() - t_after
        assert self._committed_rate >= 1.0 - 1e-6, (
            f"Tuning rate must reach 1.0 by the end of the step, "
            f"but _committed_rate is {self._committed_rate:.6f}"
        )
        t_stab = time.time()
        self._stabilize_at_full_rate()
        self._post_stabilization_hook()
        self._phase_seconds["stabilization"] = time.time() - t_stab
        result = self._restore_best_committed_state(result)
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report(f"{self.name} phase_seconds", self._phase_seconds)
        self._log_cycle_summary()
        return result

    def _restore_best_committed_state(self, finalize_result):
        """Certified non-destructive guard at run scope. When
        ``TUNING_POLICY.keepbest_certified`` is on and the finalized metric is worse
        than the best-committed state's (beyond the rollback tolerance), restore that
        state so the run never ships below the best committed cycle."""
        if not getattr(self, "_keepbest_certified", False):
            return finalize_result
        best_state = getattr(self, "_best_committed_state", None)
        best_metric = getattr(self, "_best_committed_metric", None)
        if best_state is None or best_metric is None:
            return finalize_result
        final_metric = self._certified_gate_metric()
        tol = float(getattr(self, "_rollback_tolerance", 0.0))
        if final_metric < best_metric - tol:
            self._restore_state(best_state)
            self.pipeline.reporter.report(
                f"{self.name} certified keepbest restore",
                {"final": final_metric, "best_committed": best_metric},
            )
            self._final_metric = best_metric
            return best_metric
        return finalize_result

    def _log_full_transform_trend(self):
        """Report whether the gradual ramp converges toward 1.0-viability. The verdict
        keys on the GENUINE drop (the deployed cliff), not the value-domain proxy;
        ``proxy_gap`` surfaces when the value drop shrinks but the genuine one does not."""
        log = getattr(self, "_full_transform_log", [])
        if len(log) < 2:
            return
        first, last = log[0], log[-1]
        genuine_shrinking = last["genuine_drop"] < first["genuine_drop"] - 1e-9
        value_shrinking = last["value_drop"] < first["value_drop"] - 1e-9
        verdict = "CONVERGING" if genuine_shrinking else "FLAT/DIVERGING"
        print(
            f"[{self.__class__.__name__}] full-transform probe ({len(log)} pts): "
            f"genuine_drop {first['genuine_drop']:+.4f}@α={first['committed']:.3f} → "
            f"{last['genuine_drop']:+.4f}@α={last['committed']:.3f}  "
            f"(value_drop {first['value_drop']:+.4f} → {last['value_drop']:+.4f}, "
            f"proxy_gap {first['proxy_gap']:+.4f} → {last['proxy_gap']:+.4f})  →  {verdict}"
        )
        self.pipeline.reporter.report(f"{self.name} full_transform_trend", {
            "n": len(log),
            "first_genuine_drop": round(first["genuine_drop"], 4),
            "last_genuine_drop": round(last["genuine_drop"], 4),
            "first_value_drop": round(first["value_drop"], 4),
            "last_value_drop": round(last["value_drop"], 4),
            "first_proxy_gap": round(first["proxy_gap"], 4),
            "last_proxy_gap": round(last["proxy_gap"], 4),
            "shrinking": genuine_shrinking,
        })
        if not genuine_shrinking:
            fooled = ""
            if value_shrinking:
                fooled = (
                    " The value-domain probe WAS FOOLED: value_drop DID shrink "
                    f"({first['value_drop']:+.4f} → {last['value_drop']:+.4f}) "
                    "while the genuine deployed drop did not — the value-domain "
                    "rate-1.0 measurement is not representative of the deployed "
                    "forward (proxy_gap "
                    f"{first['proxy_gap']:+.4f} → {last['proxy_gap']:+.4f})."
                )
            warnings.warn(
                f"{self.__class__.__name__}: the GENUINE full-transformation drop "
                f"did not shrink as the committed rate climbed "
                f"({first['genuine_drop']:+.4f}@α={first['committed']:.3f} → "
                f"{last['genuine_drop']:+.4f}@α={last['committed']:.3f}). The gradual "
                f"ramp may not be pulling the model toward 1.0-viability.{fooled}",
                stacklevel=2,
            )

    def _log_cycle_summary(self):
        """Print the full adaptation cycle log for debugging."""
        if getattr(self, "_full_transform_log", None):
            self._log_full_transform_trend()
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


