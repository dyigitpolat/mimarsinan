"""Shared blend-ramp mechanism for the KD-blend tuner family (LIF/TTFS)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.tuning.teacher import freeze_module


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


class KDClassificationLoss:
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


def kd_loss_from_config(config, teacher: nn.Module) -> KDClassificationLoss:
    """SSOT KD+CE weighting: ``kd_ce_alpha`` / ``kd_temperature`` default to 0.3 / 3.0 (KD-heavy)."""
    return KDClassificationLoss(
        teacher,
        temperature=float(config.get("kd_temperature", 3.0)),
        alpha=float(config.get("kd_ce_alpha", 0.3)),
    )


def run_teacher_distmatch(tuner, matcher, *, n_batches: int = 8, **matcher_kwargs):
    """Distribution-match the deployed model to the tuner's frozen KD teacher, report, return stats."""
    cal_x = tuner._calibration_inputs(n_batches)
    stats = matcher(tuner.model, tuner._teacher, cal_x, tuner._T, **matcher_kwargs)
    tuner.pipeline.reporter.report(f"{tuner.name} distmatch", stats)
    return stats
