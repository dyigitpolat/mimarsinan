"""The shared fast fixed-ladder driver — lifted from ``KDBlendAdaptationTuner``.

E2 (Fix A): *how* the rate is driven 0→1 (the optimization driver) is a
pipeline-wide axis (``controller | fast``), not a KD-blend island. The
schedule-not-search fast ladder used to live on ``KDBlendAdaptationTuner`` and
was reachable only by the LIF/TTFS families; this mixin lifts that machinery to
the shared ``SmoothAdaptationTuner`` base so EVERY rate tuner (the analytical
clamp/shift/activation-quant chain, the manager-rate family, the KD-blend family)
can be driven either by the controller (default) or by the fast ladder.

``fixed_ladder`` policy: walk an explicit rate ladder through the ONE
orchestrator's ``RateScheduler`` (``smooth_adaptation_run._run_with_scheduler``
selects it from ``_fixed_ladder_policy``) with ONE shared optimizer + spanning
warmup/cosine LR, no per-cycle rollback / recovery / LR-find / stabilization.

The axis DEFAULTS to controller: ``_setup_fast_ladder`` is opt-in, so a tuner
that never calls it has ``_fixed_ladder_policy`` False (the ``getattr(..., False)``
reads in the run loop) and runs the unchanged controller path — byte-identical.
The verbs below resolve the tuner's rate-setter uniformly (``_set_rate`` for the
KD-blend / clamp / activation-adaptation families, ``_apply_rate`` for the
manager-rate family), so the lift does not depend on which family opts in.
"""

from __future__ import annotations

import time

import torch


class FastLadderMixin:
    """The schedule-not-search fast ladder, shared by every smooth rate tuner.

    Opt in by calling ``_setup_fast_ladder(enabled=..., rates=..., ...)`` from the
    tuner's ``__init__``/``_configure``; subclasses may override the per-step
    ``_fast_loss`` (default = the trainer's installed loss) and the ``_fast_probe``
    observability hook (default no-op). Default-off ⇒ controller ⇒ byte-identical.
    """

    def _consume_optimization_driver(
        self, *, rates, steps_per_rate, eta_min_factor=0.0,
    ):
        """READ the pipeline-wide ``optimization_driver`` axis (EF1) and configure the
        fast ladder from it. The single-switch families (the analytical clamp/
        activation-quant/activation-adaptation chain + the manager-rate family) call
        this in ``__init__`` so they CONSUME the axis instead of defaulting to the
        controller with no axis read. The resolved ``OptimizationDriver`` is stashed on
        ``self._optimization_driver`` (the family's recorded decision). Default
        ``controller`` ⇒ ``_setup_fast_ladder(enabled=False)`` ⇒ byte-identical."""
        from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan

        driver = DeploymentPlan.of(self.pipeline).optimization_driver_for_family(
            rates=rates, steps_per_rate=steps_per_rate, eta_min_factor=eta_min_factor,
        )
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
        trailing 1.0 so the ramp always finishes through the fast attempt (never the
        heavy ``_continue_to_full_rate`` controller). ``eta_min_factor`` floors the
        spanning cosine at ``eta_min_factor·lr`` so the final rate-1.0 rung is not
        starved of LR — 0.0 keeps the decay-to-0; >0 (a value-domain endpoint that
        needs real recovery) keeps training the deployed dynamics at the end."""
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

    def run(self):
        """Reset the per-run fast scratch (tuner-owned optimizer + spanning cosine)
        so a re-run rebuilds them, then drive the ONE orchestrator. NOT a fork — the
        fixed_ladder policy is selected in ``_run_with_scheduler``."""
        if getattr(self, "_fixed_ladder_policy", False):
            self._fast_optimizer = None
            self._fast_lr_schedule = None
            self._fast_optimizer_steps = 0
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
        from mimarsinan.model_training.training_recipe import (
            build_optimizer,
            build_recipe,
        )

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
        """Per-step fast-ramp loss. Default = the trainer's installed loss (for the
        KD-blend family this is the KD loss; subclasses may override, e.g. TTFS to
        its plain-CE + genuine-CE objective)."""
        return self.trainer.loss_function(self.model, x, y)

    def _fast_probe(self, rate) -> None:
        """Optional per-rung observability hook (default no-op)."""

    def _fast_rate_attempt(self, target):
        """Train ``steps_per_rate`` steps at ``target`` with the shared optimizer +
        spanning cosine and ``_fast_loss``, measure a post accuracy, and record a
        commit into the trace. Always commits ``target`` (no rollback)."""
        self._ensure_fast_optimizer()
        t0 = time.time()
        device = self.pipeline.config["device"]
        self._fast_set_rate(float(target))
        for _ in range(self._fast_steps_per_rate):
            x, y = self.trainer.next_training_batch()
            x, y = x.to(device), y.to(device)
            self.model.train()
            loss = self._fast_loss(x, y)
            self._fast_optimizer.zero_grad()
            loss.backward()
            self._fast_optimizer.step()
            self._fast_lr_schedule.step()
            self._fast_optimizer_steps += 1
        self._committed_rate = float(target)
        post_acc = float(self.trainer.validate_n_batches(self._budget.eval_n_batches))
        self._record_fast_cycle(float(target), post_acc, t0)
        self._last_post_acc = post_acc
        self._fast_probe(float(target))
        self._phase_seconds["fast_blend"] = (
            self._phase_seconds.get("fast_blend", 0.0) + (time.time() - t0)
        )
        return float(target)

    def _fast_stabilize(self, steps, loss_fn=None) -> None:
        """Post-finalize bounded recovery on the DEPLOYED forward, for fast ramps
        whose value-domain endpoint still needs the deployed dynamics trained (LIF:
        the cycle-accurate forward; TTFS proxy: refine the revived cascade past the
        proxy↔genuine cliff). Fresh optimizer + spanning cosine, ``loss_fn`` (default
        ``_fast_loss``); non-destructive (rolls back on regression)."""
        loss_fn = loss_fn or self._fast_loss
        steps = max(0, int(steps))
        if steps <= 0:
            return
        from mimarsinan.model_training.training_recipe import (
            build_optimizer,
            build_recipe,
        )

        device = self.pipeline.config["device"]
        self.model = self.model.to(device)
        lr = float(self.pipeline_lr)
        recipe = build_recipe(self.pipeline.config, key="tuning_recipe")
        opt = (
            build_optimizer(self.model, lr, recipe) if recipe is not None
            else self.trainer.build_step_optimizer(lr)
        )
        sched = self._build_fast_lr_schedule(opt, steps, eta_min=0.0)
        n_eval = self._budget.eval_n_batches
        pre_state = self._clone_state()
        try:
            pre_val = float(self.trainer.validate_n_batches(n_eval))
        except Exception:
            pre_val = None
        for _ in range(steps):
            x, y = self.trainer.next_training_batch()
            x, y = x.to(device), y.to(device)
            self.model.train()
            loss = loss_fn(x, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if pre_val is not None:
            try:
                post_val = float(self.trainer.validate_n_batches(n_eval))
            except Exception:
                post_val = None
            tol = float(getattr(self, "_rollback_tolerance", 0.0))
            if post_val is not None and post_val < pre_val - tol:
                self._restore_state(pre_state)

    def _record_fast_cycle(self, target, post_acc, t0) -> None:
        """Record one ``commit`` per scheduled rate so the fast path inherits the
        DecisionTrace."""
        if not hasattr(self, "_cycle_log"):
            return
        from mimarsinan.tuning.trace import DecisionRecord

        self._cycle_log.record(DecisionRecord(
            cycle_index=len(self._cycle_log),
            outcome="commit",
            rate=float(target),
            committed=float(self._committed_rate),
            elapsed_sec=time.time() - t0,
            pre_cycle_acc=getattr(self, "_last_post_acc", None),
            post_acc=float(post_acc),
            lr=float(self._fast_optimizer.param_groups[0]["lr"]),
            target=float(self._get_target()),
            validation_baseline=self._baseline_or_none(),
        ))
