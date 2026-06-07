"""Smooth adaptation run loop and stabilization."""

from __future__ import annotations

import time
import warnings

from mimarsinan.tuning.basic_interpolation import BasicInterpolation
from mimarsinan.tuning.smart_smooth_adaptation import SmartSmoothAdaptation
from mimarsinan.tuning.orchestration.tuning_budget import min_step_for_smooth_adaptation
from mimarsinan.tuning.orchestration.tuner_base import (
    TunerBase,
    _RECOVERY_PATIENCE,
    _STUCK_STREAK_REQUIRED,
)


class SmoothAdaptationRunMixin(TunerBase):
    def _stabilization_budget(self):
        """Number of gradient steps for the final rate=1.0 stabilization pass
        (per round; ``_max_stabilization_rounds`` controls the round count).
        Subclasses may override (e.g. return ``None``/``0`` to disable)."""
        return 2 * int(self._budget.max_training_steps)

    def _post_stabilization_hook(self):
        """Called once after the final stabilization pass; subclasses may run
        deployment-aware post-training calibration here (it must only commit
        validation-improving changes — no training follows to undo them)."""

    def _stabilize_at_full_rate(self):
        """Extra training at rate=1.0 after ``_after_run`` has committed.

        Runs up to ``_max_stabilization_rounds`` passes (default 1 — the
        historical single pass); each extra round restarts from a freshly
        found LR and only happens while the previous round still improved
        validation by more than ``accuracy_se()/2`` (constant-LR passes
        plateau; an LR restart breaks the plateau — the same effect WQ's
        fresh LR search showed downstream). The pre/post rollback guard
        brackets ALL rounds, so the phase stays non-destructive.
        """
        if self._committed_rate < 1.0 - 1e-6:
            return
        budget = self._stabilization_budget()
        if budget is None or int(budget) <= 0:
            return
        budget = int(budget)

        # When finalize swapped in a deployed forward distinct from the ramp
        # (KD blend cascaded / cycle-accurate LIF), the cached LR was tuned on
        # the proxy forward and is stale — re-find it for the deployed dynamics.
        if getattr(self, "_stabilization_refinds_lr", False):
            self._invalidate_lr_cache()
        lr = min(float(self._get_cached_lr()), float(self.pipeline_lr))

        n_eval = self._budget.eval_n_batches
        pre_state = self._clone_state()
        try:
            pre_val = float(self.trainer.validate_n_batches(n_eval))
        except Exception:
            pre_val = None

        progress_n = max(
            self._budget.progress_eval_batches,
            self._budget.eval_n_batches,
        )
        max_patience = max(_RECOVERY_PATIENCE, budget // max(1, self._budget.check_interval))
        rounds = max(1, int(getattr(self, "_max_stabilization_rounds", 1)))
        last_val = pre_val
        for round_idx in range(rounds):
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
            if round_idx == rounds - 1 or last_val is None:
                break
            try:
                round_val = float(self.trainer.validate_n_batches(n_eval))
            except Exception:
                break
            if round_val - last_val <= self._budget.accuracy_se() / 2:
                break
            last_val = round_val
            self._invalidate_lr_cache()
            lr = min(float(self._get_cached_lr()), float(self.pipeline_lr))

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
        self._cycle_log = []
        self._cached_lr = None
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

        se = self._budget.accuracy_se()
        val_a = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        val_b = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        empirical_noise = abs(val_a - val_b)
        self._rollback_tolerance = max(
            min(max(3 * se, 3 * empirical_noise), 0.05),
            0.005,
        )

        baseline_val = (val_a + val_b) / 2.0
        self.target_adjuster.target_metric = baseline_val
        self.target_adjuster.original_metric = baseline_val
        self.target_adjuster.floor = baseline_val * (1.0 - self._pipeline_tolerance)

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

        if not self._skip_one_shot:
            self._before_cycle()
            self._adaptation(1.0)
            if self._committed_rate >= 1.0 - 1e-6:
                self._natural_rate = self._committed_rate
                self._phase_seconds["gradual"] = time.time() - self._run_t0
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
                self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
                self.pipeline.reporter.report(f"{self.name} phase_seconds", self._phase_seconds)
                self._log_cycle_summary()
                return result

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
            # A requested gradual ladder can never be finer than the budget's
            # min_step (else the loop would no-op and the coarse
            # _continue_to_full_rate fallback would take over).
            initial_step=max(ms, float(getattr(self, "_initial_ramp_step", 0.5))),
            growth=float(getattr(self, "_ramp_step_growth", 1.5)),
        )
        adapter.adapt_smoothly(max_cycles=max_cycles)

        if self._committed_rate < 1.0 - 1e-6:
            self._continue_to_full_rate()

        self._natural_rate = self._committed_rate
        self._phase_seconds["gradual"] = time.time() - self._run_t0

        if self._natural_rate < 1.0 - 1e-6:
            import warnings
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
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report(f"{self.name} phase_seconds", self._phase_seconds)
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


