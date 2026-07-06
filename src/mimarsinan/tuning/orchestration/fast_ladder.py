"""The shared fast fixed-ladder driver mixin for every smooth rate tuner."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import torch

from mimarsinan.model_training.training_recipe import build_optimizer, build_recipe
from mimarsinan.models.nn.layers import freeze_batchnorm_running_stats
from mimarsinan.tuning.orchestration.mbh_gate import gated_fast_rate_attempt
from mimarsinan.tuning.orchestration.mbh_ledger import (
    capture_rung_nonzero_grad_fraction,
)
from mimarsinan.tuning.trace import DecisionRecord

if TYPE_CHECKING:
    from mimarsinan.tuning.orchestration.smooth_adaptation_cycle import (
        SmoothAdaptationCycleMixin,
    )
    from mimarsinan.tuning.orchestration.smooth_adaptation_run import (
        SmoothAdaptationRunMixin,
    )

    class _FastLadderHost(SmoothAdaptationCycleMixin, SmoothAdaptationRunMixin):
        """Static contract of the composed smooth tuner this mixes into."""

else:
    _FastLadderHost = object


class FastLadderMixin(_FastLadderHost):
    """The schedule-not-search fast ladder, shared by every smooth rate tuner.

    Opt in via ``_setup_fast_ladder(...)``; subclasses may override the per-step
    ``_fast_loss`` and the ``_fast_probe`` hook. Default-off ⇒ controller.
    """

    def _consume_optimization_driver(
        self, *, rates, steps_per_rate, eta_min_factor=0.0,
    ):
        """Read the pipeline-wide ``optimization_driver`` axis and configure the fast
        ladder from it, stashing the resolved driver on ``self._optimization_driver``.
        Default ``controller`` ⇒ ``_setup_fast_ladder(enabled=False)``."""
        # Lazy: deployment_plan pulls chip_simulation, which cannot be imported at
        # this module's top without a circular import.
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        driver = DeploymentPlan.of(self.pipeline).optimization_driver_for_family(
            rates=rates, steps_per_rate=steps_per_rate, eta_min_factor=eta_min_factor,
        )
        return self._adopt_optimization_driver(driver)

    def _adopt_optimization_driver(self, driver):
        """Stash a resolved ``OptimizationDriver`` and configure the fast ladder from its fields."""
        self._optimization_driver = driver
        self._setup_fast_ladder(
            enabled=driver.fast_ladder,
            rates=driver.fast_ladder_rates,
            steps_per_rate=driver.fast_ladder_steps_per_rate,
            eta_min_factor=driver.fast_ladder_eta_min_factor,
        )
        return driver

    def _setup_fast_ladder(self, *, enabled, rates, steps_per_rate,
                           eta_min_factor=0.0) -> None:
        """Configure the fixed-ladder fast schedule. ``rates`` is normalized to a
        trailing 1.0 so the ramp finishes through the fast attempt; ``eta_min_factor``
        floors the spanning cosine at ``eta_min_factor·lr`` (0.0 decays to ~0)."""
        self._fixed_ladder_policy = bool(enabled)
        ladder = [float(r) for r in (rates or [])] or [1.0]
        if abs(ladder[-1] - 1.0) > 1e-9:
            ladder = [*ladder, 1.0]
        self._fixed_ladder_rates = ladder
        self._fast_steps_per_rate = max(0, int(steps_per_rate))
        self._fast_eta_min_factor = max(0.0, float(eta_min_factor))
        self._fast_optimizer = None
        self._fast_lr_schedule = None
        self._fast_optimizer_steps = 0
        self._fast_blend_path = False
        self._fast_freeze_batchnorm = bool(
            self.pipeline.config.get("fast_ladder_freeze_bn", False)
        )

    def run(self):
        """Reset the per-run fast scratch so a re-run rebuilds it, then drive the one
        orchestrator (the fixed_ladder policy is selected in ``_run_with_scheduler``)."""
        if getattr(self, "_fixed_ladder_policy", False):
            self._fast_optimizer = None
            self._fast_lr_schedule = None
            self._fast_optimizer_steps = 0
            self._mbh_rung_index = -1
            self._mbh_gate_state = None
        return super().run()

    def _driver_attempt(self, target):
        """Per-rate attempt the scheduler drives: the fast fixed-ladder attempt when
        enabled, else the standard predictor→corrector cycle."""
        if getattr(self, "_fixed_ladder_policy", False):
            return self._fast_rate_attempt(target)
        return super()._driver_attempt(target)

    def _stabilization_budget(self):
        """The fast path trains through the deployed dynamics for the whole ramp
        (cliff ≈ 0), so skip the post-finalize stabilization pass."""
        if getattr(self, "_fixed_ladder_policy", False):
            return 0
        return super()._stabilization_budget()

    def _fast_set_rate(self, rate) -> None:
        """Uniform rate-setter for the fast ladder: ``_set_rate`` (KD-blend / clamp /
        activation-adaptation families) or ``_apply_rate`` (the manager-rate family),
        whichever the tuner defines. Resolved per call so a subclass override wins."""
        setter = getattr(self, "_set_rate", None) or getattr(self, "_apply_rate", None)
        if setter is None:
            raise AttributeError(
                f"{type(self).__name__} has neither _set_rate nor _apply_rate; "
                "the fast ladder cannot drive its rate."
            )
        setter(float(rate))

    def _build_fast_lr_schedule(self, optimizer, total_steps, eta_min=0.0):
        """Warmup (5%, linear) → cosine decay to ``eta_min`` over ``total_steps``
        step()s (``eta_min=0`` decays to ~0; >0 floors the endpoint LR)."""
        total = max(1, int(total_steps))
        warmup_steps = max(1, int(round(0.05 * total)))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total - warmup_steps), eta_min=float(eta_min),
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )

    def _ensure_fast_optimizer(self) -> None:
        """Build the single optimizer + spanning warmup/cosine LR once, sized to the
        whole ladder so the LR anneals smooth→~0 across ALL rates."""
        if self._fast_optimizer is not None:
            return
        device = self.pipeline.config["device"]
        self.model = self.model.to(device)
        lr = float(self.pipeline_lr)
        recipe = build_recipe(self.pipeline.config, key="tuning_recipe")
        if recipe is not None:
            self._fast_optimizer = build_optimizer(self.model, lr, recipe)
        else:
            self._fast_optimizer = self.trainer.build_step_optimizer(lr)
        total = max(1, len(self._fixed_ladder_rates) * self._fast_steps_per_rate)
        eta_min = lr * float(getattr(self, "_fast_eta_min_factor", 0.0))
        self._fast_lr_schedule = self._build_fast_lr_schedule(
            self._fast_optimizer, total, eta_min=eta_min,
        )
        self._fast_blend_path = True
        self._fast_optimizer_steps = 0

    def _fast_loss(self, x, y):
        """Per-step fast-ramp loss. Default = the trainer's installed loss; subclasses
        may override (e.g. TTFS to its plain-CE + genuine-CE objective)."""
        return self.trainer.loss_function(self.model, x, y)

    def _fast_probe(self, rate) -> None:
        """Optional per-rung observability hook (default no-op)."""

    def _fast_ramp(self, rate) -> None:
        """The fast-ladder predictor seam verb: apply T at ``rate`` and train the
        rung's ``steps_per_rate`` steps with the shared optimizer + spanning cosine
        and ``_fast_loss`` (no per-rung search — ``probe`` reads the post acc once)."""
        device = self.pipeline.config["device"]
        self._fast_set_rate(float(rate))
        optimizer, schedule = self._fast_optimizer, self._fast_lr_schedule
        assert optimizer is not None and schedule is not None, (
            "_ensure_fast_optimizer must run before _fast_ramp"
        )
        self._mbh_nonzero_grad_fraction = float("nan")
        for step_index in range(self._fast_steps_per_rate):
            x, y = self.trainer.next_training_batch()
            x, y = x.to(device), y.to(device)
            self.model.train()
            if getattr(self, "_fast_freeze_batchnorm", False):
                freeze_batchnorm_running_stats(self.model)
            loss = self._fast_loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            if step_index == 0:
                # A5 reach gauge on the rung's FIRST backward (ledger-flag-gated).
                capture_rung_nonzero_grad_fraction(self)
            optimizer.step()
            schedule.step()
            self._fast_optimizer_steps += 1

    def _continue_to_full_rate(self):
        """The gated fixed ladder never trains destructive intermediate rates:
        the forced jump to 1.0 skips the adaptive micro-ramp (finalize applies
        the full rate without training)."""
        if getattr(self, "_fixed_ladder_policy", False):
            return None
        return super()._continue_to_full_rate()

    def _fast_rate_attempt(self, target):
        """One fast-ladder rung, driven THROUGH the seam verbs (EF2) by the
        default [MBH-GATE] D-hat trust region: the fast predictor ``_fast_ramp``
        (apply T + train), the deployed full-transform read, then
        accept / midpoint-refine / constructive stall (X3 recipe promotion)."""
        return gated_fast_rate_attempt(self, float(target))

    def _mbh_full_transform_forward(self, clone):
        """[MBH] Force the deployed full transformation (rate 1.0) on an isolated
        ``clone`` and return the forward to evaluate; live state stays untouched.

        Default: replay this tuner's rate axis on the clone against a deepcopied
        manager (the manager-rate family); KD-blend families override with their
        finalize rebuild + genuine forward.
        """
        axis = getattr(self, "_axis", None)
        manager = getattr(self, "adaptation_manager", None)
        if axis is None or manager is None:
            raise AttributeError(
                f"{type(self).__name__} exposes no rate axis / adaptation manager; "
                "the [MBH] full-transform probe cannot prepare an isolated clone."
            )
        replica = axis.probe_replica(clone, copy.deepcopy(manager), self.pipeline.config)
        replica.set_rate(1.0)
        return clone

    def _record_fast_cycle(self, target, post_acc, t0, outcome="commit") -> None:
        """Record one decision per attempted rate so the fast path inherits the
        DecisionTrace (``outcome`` != commit only on [MBH-GATE] rejects)."""
        if not hasattr(self, "_cycle_log"):
            return
        optimizer = self._fast_optimizer
        assert optimizer is not None, (
            "_ensure_fast_optimizer must run before _record_fast_cycle"
        )
        self._cycle_log.record(DecisionRecord(
            cycle_index=len(self._cycle_log),
            outcome=str(outcome),
            rate=float(target),
            committed=float(self._committed_rate),
            elapsed_sec=time.time() - t0,
            pre_cycle_acc=getattr(self, "_last_post_acc", None),
            post_acc=float(post_acc),
            lr=float(optimizer.param_groups[0]["lr"]),
            target=float(self._get_target()),
            validation_baseline=self._baseline_or_none(),
        ))
