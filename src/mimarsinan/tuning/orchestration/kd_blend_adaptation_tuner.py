"""Shared KD blend-ramp adaptation.

Ramp each chip-targeted perceptron's ``base_activation`` from its current
behavior toward a target on-chip activation via a linear ``BlendActivation``
(rate 0→1), recovering accuracy with knowledge distillation against a frozen
snapshot taken before the swap. ``LIFAdaptationTuner`` and
``TTFSCycleAdaptationTuner`` share this machinery and differ only in the target
activation and the finalize step.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


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
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def __call__(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if next(self.teacher.parameters()).device != x.device:
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


class KDBlendAdaptationTuner(SmoothAdaptationTuner):
    """Blend each perceptron's base activation toward a target, with KD recovery."""

    _target_activation_type = "Target"
    _old_activation_type = "ReLU"

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self._final_metric = None

        self._configure()
        self._teacher = self._snapshot_teacher()
        self._install_blend()
        self.trainer.loss_function = self._make_kd_loss()
        self._after_install_blend()

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

    def _after_install_blend(self) -> None:
        """Hook after all blends + KD loss are installed (e.g. install a forward)."""

    def _make_kd_loss(self):
        return _KDClassificationLoss(self._teacher)

    def _finalize(self) -> None:
        """Hook at full rate: set subsume flags, update activations, install forwards."""

    # ── Shared machinery ─────────────────────────────────────────────────────

    def _snapshot_teacher(self) -> nn.Module:
        device = self.pipeline.config["device"]
        self.model.to("cpu")
        teacher = copy.deepcopy(self.model)
        self.model.to(device)
        teacher.to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

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
        for p in self.model.get_perceptrons():
            p.base_activation.rate = float(rate)

    def _get_extra_state(self):
        return self._get_rates()

    def _set_extra_state(self, extra):
        for p, r in zip(self.model.get_perceptrons(), extra):
            p.base_activation.rate = float(r)

    def _update_and_evaluate(self, rate: float):
        self._set_rate(rate)
        return self.trainer.validate_n_batches(self._budget.progress_eval_batches)

    def _after_run(self):
        self._continue_to_full_rate()
        self._set_rate(1.0)
        self._finalize()
        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
