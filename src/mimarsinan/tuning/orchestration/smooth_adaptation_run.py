"""Smooth adaptation run loop and stabilization."""

from __future__ import annotations

import time
import warnings

from mimarsinan.tuning.trace import DecisionTrace
from mimarsinan.tuning.orchestration.acceptance_sensor import AcceptanceSensor
from mimarsinan.tuning.orchestration.adaptation_driver import AdaptationDriver
from mimarsinan.tuning.orchestration.characterization import characterize
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
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
            # A disabled stabilization budget (e.g. the fast genuine-blend path, which
            # trains through the cascade for the whole ramp) wins over BOTH the
            # patience pass and the bounded-cosine variant — no off-recipe pass.
            return
        if getattr(self, "_stabilization_bounded", False):
            self._stabilize_bounded_cosine()
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
            )
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

    def _stabilize_bounded_cosine(self):
        """CHANGE 2: a SINGLE bounded stabilization pass of
        ``N = int(tuning_stabilization_ratio * gradual_train_steps)`` steps with a
        cosine-decay LR (from the chosen LR down to ~0 over exactly N steps). HARD
        cutoff — no patience extension, no LR restarts, no extra rounds — bracketed
        by the same pre/post rollback guard so the phase stays non-destructive."""
        ratio = float(getattr(self, "_stabilization_ratio", 0.5))
        gradual_steps = int(getattr(self, "_gradual_train_steps", 0))
        n_steps = int(ratio * gradual_steps)
        if n_steps <= 0:
            return

        if getattr(self, "_stabilization_refinds_lr", False):
            self._invalidate_lr_cache()
        lr = min(float(self._get_cached_lr()), float(self.pipeline_lr))

        n_eval = self._budget.eval_n_batches
        pre_state = self._clone_state()
        try:
            pre_val = float(self.trainer.validate_n_batches(n_eval))
        except Exception:
            pre_val = None

        hooks = self._recovery_training_hooks(1.0)
        # ``check_interval = n_steps`` => the only validation is the final step, so
        # the unreachable target can never break the run early: exactly N steps run.
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
        )

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
        self._best_committed_acc = None
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
        self.target_adjuster.target_metric = baseline_val
        self.target_adjuster.original_metric = baseline_val
        self.target_adjuster.floor = baseline_val * (1.0 - self._pipeline_tolerance)

        self._validation_baseline = baseline_val

        # P2b: capture the fixed-baseline correctness vector for the paired gate,
        # on the original (rate-0) model over a shared confirm subsample.
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

    def _run_with_scheduler(self):
        """The single rate-search loop: one RateScheduler replacing the legacy
        one-shot + grow/halve ramp + continue-to-full-rate loops.

        Greedy-to-1.0 + bisect for the standard axes; a uniform ladder for the
        ``_skip_one_shot`` (KD-blend) family. The shared finalize + stabilization
        tail (``_finalize_run``) forces the rate to 1.0 and stabilizes.
        """
        ms = min_step_for_smooth_adaptation(self.pipeline, self._budget)
        max_cycles = min(30, max(
            10,
            self._budget.max_training_steps // max(1, self._budget.check_interval),
        ))
        # The ladder uses its intended ramp step directly. The legacy
        # ``max(ms, …)`` guard (against a no-op step finer than the budget min
        # step) is no longer needed: the scheduler always attempts the first
        # jump, so a sub-``epsilon`` ramp step still runs (epsilon only bounds
        # the bisection). ``ms`` is an absurd ~1/3 for tiny models, which would
        # otherwise coarsen the KD-blend ramp past its committable foothold.
        initial_step = float(getattr(self, "_initial_ramp_step", 0.5))
        epsilon = ms
        policy_override = None
        rates = None
        if getattr(self, "_fixed_ladder_policy", False):
            # Schedule-not-search: a well-conditioned transformation (the genuine
            # blend fast path) walks an explicit rate ladder with no bisection and
            # no characterization — the fixed schedule is the policy.
            policy_override = "fixed_ladder"
            rates = getattr(self, "_fixed_ladder_rates", None) or [1.0]
        else:
            profile = self._maybe_characterize()
            if profile is not None:
                epsilon = float(profile.epsilon_hint)
                if not profile.monotonic:
                    # A1: a non-monotone axis breaks the greedy/bisect monotonicity
                    # premise — walk a dense grid at the characterized safe step.
                    policy_override = "dense_grid"
                    initial_step = float(profile.epsilon_hint)
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

    def _maybe_characterize(self):
        """Pre-phase axis characterization (spec §10), off unless flagged. Sweeps a
        coarse α grid (non-destructively: clone → probe → restore) and returns a
        ``Profile`` whose ``epsilon_hint``/``monotonic`` configure the scheduler.
        Bit-exact no-op when disabled — the default path never builds a Profile."""
        if not bool(self.pipeline.config.get("tuning_enable_characterization", False)):
            return None
        grid = self.pipeline.config.get(
            "tuning_characterization_grid", [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        baseline = getattr(self, "_validation_baseline", None)
        if not grid or baseline is None:
            return None

        budget = float(getattr(self, "_rollback_tolerance", 0.02))
        pre = self._clone_state()

        def drop_fn(alpha):
            return float(baseline) - float(self._update_and_evaluate(float(alpha)))

        try:
            profile = characterize(drop_fn, [float(a) for a in grid], budget=budget)
        finally:
            self._restore_state(pre)

        self.pipeline.reporter.report(f"{self.name} characterization", {
            "monotonic": profile.monotonic,
            "epsilon_hint": round(float(profile.epsilon_hint), 6),
            "max_slope": round(float(profile.max_slope), 4),
            "feasible_max": round(float(profile.feasible_max), 4),
        })
        return profile

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
        self.pipeline.reporter.report(f"{self.name} committed", self._committed_rate)
        self.pipeline.reporter.report(f"{self.name} phase_seconds", self._phase_seconds)
        self._log_cycle_summary()
        return result

    def _log_full_transform_trend(self):
        """Report whether the gradual ramp converges toward 1.0-viability. The
        verdict keys on the GENUINE drop (``committed_acc - genuine_full_acc``,
        the deployed cliff) — NOT the value-domain proxy. The value↔genuine
        divergence (``proxy_gap``) is surfaced alongside: when the value drop
        shrinks while the genuine drop does not, the value-domain probe "was
        fooled" — the deployed cascade stays as far off as ever
        (``tuning_full_transform_probe`` exists to surface exactly this)."""
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


