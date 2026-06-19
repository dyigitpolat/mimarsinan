"""Shared KD blend-ramp adaptation.

Ramp each chip-targeted perceptron's ``base_activation`` from its current
behavior toward a target on-chip activation via a linear ``BlendActivation``
(rate 0→1), recovering accuracy with knowledge distillation against a frozen
snapshot taken before the swap. ``LIFAdaptationTuner`` and
``TTFSCycleAdaptationTuner`` share this machinery and differ only in the target
activation and the finalize step.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.tuning.forward_install import CascadeForwardInstall, LazyExecutorForward
from mimarsinan.tuning.orchestration.genuine_probe import (
    eval_forward_over_val,
    genuine_acc_on_clone,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner
from mimarsinan.tuning.perceptron_rate import rebuild_activations, set_blend_rate
from mimarsinan.tuning.teacher import freeze_module, snapshot_frozen_teacher

# Back-compat alias: the LIF/TTFS finalize forwards subclass this name.
_InstalledForward = LazyExecutorForward


class BlendActivation(nn.Module):
    """Linear blend of an old activation and a target activation controlled by ``rate``."""

    def __init__(
        self,
        old_activation: nn.Module,
        target_activation: nn.Module,
        rate: float = 0.0,
        *,
        target_type: str,
        old_type: str = "ReLU",
    ):
        super().__init__()
        self.old_activation = old_activation
        self.target_activation = target_activation
        self.rate = float(rate)
        self._target_type = target_type
        self._old_type = old_type

    @property
    def activation_type(self) -> str:
        return self._target_type if self.rate >= 1.0 else self._old_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rate <= 0.0:
            return self.old_activation(x)
        if self.rate >= 1.0:
            return self.target_activation(x)
        return (1.0 - self.rate) * self.old_activation(x) + self.rate * self.target_activation(x)


class _KDClassificationLoss:
    """KD + CE loss (T=3, α=0.3) for BasicTrainer.loss_function(model, x, y)."""

    def __init__(self, teacher: nn.Module, temperature: float = 3.0, alpha: float = 0.3):
        self.teacher = teacher
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        freeze_module(self.teacher)

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            teacher_param = next(self.teacher.parameters(), None)
            if teacher_param is not None and teacher_param.device != x.device:
                self.teacher.to(x.device)
            teacher_logits = self.teacher(x)
        student_logits = model(x)
        ce = F.cross_entropy(student_logits, y)
        T = self.temperature
        kd = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        return self.alpha * ce + (1.0 - self.alpha) * kd


class KDBlendAdaptationTuner(CascadeForwardInstall, SmoothAdaptationTuner):
    """Blend each perceptron's base activation toward a target, with KD recovery."""

    _target_activation_type = "Target"
    _old_activation_type = "ReLU"

    # Gradual-ramp philosophy: the ANN->SNN activation ramp must be
    # genuinely gradual — many small committed increments, each recovered by
    # training DURING the ramp. No one-shot jump to rate 1.0 (which relabels
    # the recovery as "stabilization"), and a small uniform ladder instead of
    # the historical 0.5-sized cliffs (each licensed to lose rollback_tolerance).
    _skip_one_shot = True
    _initial_ramp_step = 0.125
    _ramp_step_growth = 1.0
    # Post-finalize polish at the deployed dynamics: extra stabilization
    # rounds with LR restarts while validation still improves (the proxy↔
    # genuine gap means the deployed function needs real training after the
    # finalize swap; a constant-LR single pass plateaus early).
    _max_stabilization_rounds = 3

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self._final_metric = None
        self._finalize_cliff = None

        self._configure()
        # The ramp strategy bundles every seam that varies 0→1 (axis / blend /
        # ramp-forward / KD loss / pre-ramp setup); built from the (flag-derived)
        # config in _configure, BEFORE the seams below consult it.
        self._ramp = self._make_ramp_strategy()
        self._teacher = self._snapshot_teacher()
        self._install_blend()
        self.trainer.loss_function = self._make_kd_loss()
        self._after_install_blend()

        # Blend-rate application is owned by an axis (delegates to the
        # set_blend_rate SSOT). finalize stays on this tuner's inherited
        # _finalize (parity-critical forward-install), never on the axis.
        self._axis = self._make_axis()
        self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

    # ── Hooks (subclasses customize) ─────────────────────────────────────────

    def _configure(self) -> None:
        """Read config, set names/params, and any adaptation_manager flags."""

    def _make_ramp_strategy(self):
        """The ramp strategy bundling the 0→1 seams. Default = the value-domain
        proxy ramp; subclasses pick a strategy from the flags set in ``_configure``."""
        from mimarsinan.tuning.orchestration.ramp_strategy import ValueDomainProxyRamp

        return ValueDomainProxyRamp()

    def _make_target_activation(self, perceptron) -> nn.Module:
        raise NotImplementedError

    def _blend_old_activation(self, perceptron) -> nn.Module:
        """Old side of the blend. Default: the perceptron's current base activation."""
        return perceptron.base_activation

    def _make_blend(self, old: nn.Module, target: nn.Module, rate: float) -> nn.Module:
        return self._ramp.make_blend(self, old, target, rate)

    def _make_axis(self):
        """The rate axis the tuner walks (the ramp strategy owns the choice;
        default = the value-domain ``BlendAxis``)."""
        return self._ramp.make_axis(self)

    def _after_make_target(self, perceptron, target: nn.Module) -> None:
        """Per-perceptron hook after the blend is installed (before update_activation)."""

    def _wrap_encoding_input(self, perceptron) -> None:
        """Optional encoding-layer input wrapping (e.g. ChipInputQuantizer)."""

    def _append_encoding_input_module(self, perceptron, module: nn.Module) -> None:
        """Append a wire op (e.g. an STE input quantizer) after input_activation."""
        if isinstance(perceptron.input_activation, nn.Identity):
            perceptron.input_activation = module
        else:
            perceptron.input_activation = nn.Sequential(
                perceptron.input_activation, module,
            )

    def _after_install_blend(self) -> None:
        """Strategy-specific pre-ramp setup, then install the ramp forward (if any)
        after blends + KD loss are set."""
        self._ramp.after_install_blend_pre(self)
        self._install_ramp_forward()

    def _install_ramp_forward(self) -> None:
        fwd = self._ramp_forward()
        if fwd is not None:
            self._install_forward(fwd)

    def _make_kd_loss(self):
        return self._ramp.make_kd_loss(self)

    def _remove_forward(self) -> None:
        """Let the ramp strategy clean up its owned references, then unpatch."""
        self._ramp.on_remove_forward(self)
        super()._remove_forward()

    def _ramp_forward(self):
        """Cross-layer forward installed during the blend ramp (the ramp strategy
        owns the choice; default ``None`` = the value-domain ramp)."""
        return self._ramp.ramp_forward(self, self.model)

    def _finalize_forward_for(self, model):
        """The SHARED genuine-forward builder bound to ``model`` (``None`` for
        schedules with no separate genuine forward — synchronized TTFS /
        non-cycle-accurate LIF — which keep the class forward). The probe and the
        finalize install both route through this, so the probed forward and the
        deployed forward are identical by construction."""
        return None

    def _finalize_forward(self):
        """Cross-layer forward installed at finalize (``None`` keeps the class
        forward — the deployed dynamics for analytical/synchronized schedules)."""
        return self._finalize_forward_for(self.model)

    def _update_target_activations(self, model=None) -> None:
        target = self.model if model is None else model
        rebuild_activations(target, self.adaptation_manager, self.pipeline.config)

    def _before_finalize_rebuild(self, model=None) -> None:
        """Set adaptation-manager flags the activation rebuild must see (e.g. LIF
        sets ``lif_active`` so the rebuilt activations subsume the decorators).
        ``model`` is the rebuild target (``self.model`` at finalize, the isolated
        clone during the genuine probe)."""

    def _after_finalize_rebuild(self, model=None) -> None:
        """Per-model finalize work after activations are rebuilt, before the
        deployed forward is installed (e.g. LIF applies cycle-accurate trains).
        ``model`` is the rebuild target (``self.model`` at finalize, the clone
        during the genuine probe)."""

    def _finalize_rebuild(self, model=None) -> None:
        """The finalize-style activation rebuild on ``model`` (``self.model`` when
        omitted): pre-rebuild flags → rebuild → post-rebuild work. Shared by
        ``_finalize`` (on ``self.model``) and the genuine probe (on an isolated
        clone), so probe ≡ deploy. The hooks default to ``self.model`` so legacy
        no-arg patches stay valid."""
        self._before_finalize_rebuild(model)
        self._update_target_activations(model)
        self._after_finalize_rebuild(model)

    def _finalize(self) -> None:
        """At full rate: rebuild the committed activations, then install the
        genuine cross-layer forward so the committed metric, recovery, and every
        downstream step run the exact deployed dynamics. Subclasses customize via
        the ordered ``_before_finalize_rebuild`` / ``_after_finalize_rebuild``
        hooks rather than copying this body."""
        self._before_finalize_rebuild()
        self._update_target_activations()
        self._after_finalize_rebuild()
        fwd = self._finalize_forward()
        # A deployed forward distinct from the ramp invalidates the ramp's
        # cached LR; stabilization must re-find it on the deployed dynamics.
        self._stabilization_refinds_lr = fwd is not None
        if fwd is not None:
            self._install_forward(fwd)

    # ── Genuine full-transform probe ─────────────────────────────────────────

    # Manager flags the finalize-style rebuild may toggle (snapshotted/restored
    # around the clone rebuild so the shared adaptation_manager survives the
    # non-destructive probe). ``ttfs_active`` is set in ``_configure`` and never
    # toggled by the rebuild; ``lif_active`` is set by LIF's pre-rebuild hook.
    _finalize_manager_flags = ("lif_active", "ttfs_active")

    def _full_transform_eval(self):
        """The GENUINE full-transform accuracy: run the deployed finalize forward
        at rate 1.0 on an isolated clone. Falls back to the value-domain measure
        for schedules with no separate genuine forward (synchronized TTFS /
        non-cycle-accurate LIF). Non-destructive to the live model (the deepcopy
        in ``genuine_acc_on_clone``) AND to the shared adaptation_manager (the
        finalize-flag snapshot/restore below). The probed forward is built via the
        SAME ``_finalize_forward_for`` the deploy install uses, so probe ≡ deploy."""
        if self._finalize_forward_for(self.model) is None:
            return super()._full_transform_eval()

        device = self.pipeline.config["device"]
        manager = self.adaptation_manager
        flag_snapshot = {
            name: getattr(manager, name)
            for name in self._finalize_manager_flags
            if hasattr(manager, name)
        }

        def prepare(clone):
            set_blend_rate(clone, 1.0)
            self._finalize_rebuild(clone)

        try:
            return genuine_acc_on_clone(
                self.model,
                device,
                prepare=prepare,
                build_forward=lambda m: self._finalize_forward_for(m),
                evaluate=lambda fwd, m: eval_forward_over_val(
                    self.trainer, fwd, m,
                    self._budget.progress_eval_batches, device,
                ),
            )
        finally:
            for name, value in flag_snapshot.items():
                setattr(manager, name, value)

    # ── Shared machinery ─────────────────────────────────────────────────────

    def _snapshot_teacher(self) -> nn.Module:
        return snapshot_frozen_teacher(self.model, self.pipeline.config["device"])

    def _calibration_inputs(self, n_batches: int = 8) -> torch.Tensor:
        """A few concatenated validation batches as a calibration anchor (shared by
        the TTFS/LIF teacher-distribution matchers)."""
        device = self.pipeline.config["device"]
        batches = [
            x.to(device)
            for x, _ in self.trainer.iter_validation_batches(n_batches)
        ]
        return torch.cat(batches)

    def _install_blend(self) -> None:
        for perceptron in self.model.get_perceptrons():
            old = self._blend_old_activation(perceptron)
            target = self._make_target_activation(perceptron)
            perceptron.base_activation = self._make_blend(old, target, 0.0)
            self._after_make_target(perceptron, target)
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
            self._wrap_encoding_input(perceptron)

    def _set_rate(self, rate: float) -> None:
        self._axis.set_rate(rate)

    # ── Fast fixed-ladder path (schedule-not-search, shared by LIF + TTFS) ────
    # Walk an explicit rate ladder through the ONE orchestrator's ``fixed_ladder``
    # RateScheduler policy with ONE shared optimizer + spanning warmup/cosine LR,
    # no per-cycle rollback/recovery/LR-find/stabilization. Subclasses opt in by
    # calling ``_setup_fast_ladder`` in ``_configure`` and may override the per-step
    # ``_fast_loss`` (default = the installed KD loss) and ``_fast_probe`` hooks.
    def _setup_fast_ladder(self, *, enabled, rates, steps_per_rate,
                           eta_min_factor=0.0) -> None:
        """Configure the fixed-ladder fast schedule. ``rates`` is normalized to a
        trailing 1.0 so the ramp always finishes through the fast attempt (never the
        heavy ``_continue_to_full_rate`` controller). ``eta_min_factor`` floors the
        spanning cosine at ``eta_min_factor·lr`` so the final rate-1.0 rung is not
        starved of LR — 0.0 (TTFS, where genuine-CE carries the endpoint) keeps the
        decay-to-0; >0 (LIF value-domain ramp, where the endpoint needs real
        recovery) keeps training the deployed dynamics at the end of the ramp."""
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
        """Per-step fast-ramp loss. Default = the installed KD loss (used by LIF);
        TTFS overrides to its validated plain-CE + genuine-CE objective."""
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
        self._set_rate(float(target))
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

    def _fast_stabilize(self, steps) -> None:
        """Post-finalize bounded recovery on the DEPLOYED forward, for fast ramps
        whose value-domain endpoint still needs the deployed dynamics trained (LIF:
        the cycle-accurate forward; TTFS does NOT need this — its rate-1 blend IS the
        deployed cascade by construction). Fresh optimizer + spanning cosine, the
        installed KD ``_fast_loss``; non-destructive (rolls back on regression)."""
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
            loss = self._fast_loss(x, y)
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

    def _get_extra_state(self):
        return self._axis.get_extra_state()

    def _set_extra_state(self, extra):
        self._axis.set_extra_state(extra)

    def _update_and_evaluate(self, rate: float):
        self._set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _safe_eval(self):
        try:
            return float(self.trainer.validate_n_batches(self._budget.eval_n_batches))
        except Exception:
            return None

    def _after_run(self):
        try:
            self._continue_to_full_rate()
            self._set_rate(1.0)
            ramp_metric = self._safe_eval()
        finally:
            self._remove_forward()
        self._finalize()
        post_finalize = self._safe_eval()
        if ramp_metric is not None and post_finalize is not None:
            # Finalize cliff: ramp-end metric (ramp forward at r=1) minus the
            # metric right after installing the deployed finalize forward. For
            # the genuine-gradual ramp both run the deployed dynamics, so the
            # cliff is ~0; the value-domain proxy ramp shows the proxy gap here.
            self._finalize_cliff = ramp_metric - post_finalize
            self.pipeline.reporter.report(
                f"{self.name} finalize_cliff", self._finalize_cliff,
            )
        self._report_cliff_probe_consistency()
        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def _report_cliff_probe_consistency(self) -> None:
        """Drift sentinel: the per-commit genuine probe drop and the once-only
        finalize cliff measure the same proxy↔genuine gap from different anchors,
        so the last per-commit ``genuine_drop`` should track ``finalize_cliff``.
        A large divergence means the probe and the deploy disagree (warn, do not
        assert — a budget-noise gap on tiny models is expected)."""
        log = getattr(self, "_full_transform_log", None)
        if not log:
            return
        last_genuine_drop = float(log[-1]["genuine_drop"])
        finalize_cliff = self._finalize_cliff
        if finalize_cliff is None:
            return
        finalize_cliff = float(finalize_cliff)
        abs_diff = abs(last_genuine_drop - finalize_cliff)
        self.pipeline.reporter.report(f"{self.name} cliff_probe_consistency", {
            "last_genuine_drop": round(last_genuine_drop, 4),
            "finalize_cliff": round(finalize_cliff, 4),
            "abs_diff": round(abs_diff, 4),
        })
        if abs_diff > 0.1:
            import warnings
            warnings.warn(
                f"{self.__class__.__name__}: per-commit genuine probe drop "
                f"({last_genuine_drop:+.4f}) and finalize cliff "
                f"({finalize_cliff:+.4f}) diverge by {abs_diff:.4f}; the genuine "
                "probe may not be tracking the deployed finalize dynamics.",
                stacklevel=2,
            )

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
