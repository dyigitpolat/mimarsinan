"""Shared KD blend-ramp adaptation.

Ramp each chip-targeted perceptron's ``base_activation`` from its current
behavior toward a target on-chip activation via a linear ``BlendActivation``
(rate 0→1), recovering accuracy with knowledge distillation against a frozen
snapshot taken before the swap. ``LIFAdaptationTuner`` and
``TTFSCycleAdaptationTuner`` share this machinery and differ only in the target
activation and the finalize step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.tuning.forward_install import CascadeForwardInstall, LazyExecutorForward
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

    # SmartSmoothAdaptation philosophy: the ANN->SNN activation ramp must be
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
        self._teacher = self._snapshot_teacher()
        self._install_blend()
        self.trainer.loss_function = self._make_kd_loss()
        self._after_install_blend()

        # P1 flag: route blend-rate application through a BlendAxis (delegates to
        # the same set_blend_rate SSOT — byte-identical). finalize stays on this
        # tuner's inherited _finalize (parity-critical forward-install).
        self._axis = None
        if pipeline.config.get("tuning_use_axis", False):
            from mimarsinan.tuning.axes import BlendAxis

            self._axis = BlendAxis()
            self._axis.attach(self.model, self.adaptation_manager, self.pipeline.config)

    # ── Hooks (subclasses customize) ─────────────────────────────────────────

    def _configure(self) -> None:
        """Read config, set names/params, and any adaptation_manager flags."""

    def _make_target_activation(self, perceptron) -> nn.Module:
        raise NotImplementedError

    def _blend_old_activation(self, perceptron) -> nn.Module:
        """Old side of the blend. Default: the perceptron's current base activation."""
        return perceptron.base_activation

    def _make_blend(self, old: nn.Module, target: nn.Module, rate: float) -> nn.Module:
        return BlendActivation(
            old, target, rate,
            target_type=self._target_activation_type,
            old_type=self._old_activation_type,
        )

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
        """Install the ramp forward (if any) after blends + KD loss are set."""
        fwd = self._ramp_forward()
        if fwd is not None:
            self._install_forward(fwd)

    def _make_kd_loss(self):
        return _KDClassificationLoss(self._teacher)

    def _ramp_forward(self):
        """Cross-layer forward installed during the blend ramp.

        Default ``None`` = the value-domain ramp (the plain class forward
        through the per-perceptron ``BlendActivation``), the golden
        non-destructive ramp. Subclasses may install a cross-layer forward.
        """
        return None

    def _finalize_forward(self):
        """Cross-layer forward installed at finalize (``None`` keeps the class
        forward — the deployed dynamics for analytical/synchronized schedules)."""
        return None

    def _update_target_activations(self) -> None:
        rebuild_activations(self.model, self.adaptation_manager, self.pipeline.config)

    def _before_finalize_rebuild(self) -> None:
        """Set adaptation-manager flags the activation rebuild must see (e.g. LIF
        sets ``lif_active`` so the rebuilt activations subsume the decorators)."""

    def _after_finalize_rebuild(self) -> None:
        """Per-model finalize work after activations are rebuilt, before the
        deployed forward is installed (e.g. LIF applies cycle-accurate trains)."""

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

    # ── Shared machinery ─────────────────────────────────────────────────────

    def _snapshot_teacher(self) -> nn.Module:
        return snapshot_frozen_teacher(self.model, self.pipeline.config["device"])

    def _install_blend(self) -> None:
        for perceptron in self.model.get_perceptrons():
            old = self._blend_old_activation(perceptron)
            target = self._make_target_activation(perceptron)
            perceptron.base_activation = self._make_blend(old, target, 0.0)
            self._after_make_target(perceptron, target)
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)
            self._wrap_encoding_input(perceptron)

    def _get_rates(self) -> list[float]:
        return [p.base_activation.rate for p in self.model.get_perceptrons()]

    def _set_rate(self, rate: float) -> None:
        if getattr(self, "_axis", None) is not None:
            self._axis.set_rate(rate)
            return
        set_blend_rate(self.model, rate)

    def _get_extra_state(self):
        if getattr(self, "_axis", None) is not None:
            return self._axis.get_extra_state()
        return self._get_rates()

    def _set_extra_state(self, extra):
        if getattr(self, "_axis", None) is not None:
            self._axis.set_extra_state(extra)
            return
        for p, r in zip(self.model.get_perceptrons(), extra):
            p.base_activation.rate = float(r)

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
        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
