"""LIF adaptation tuner: blend ramp to LIFActivation with KD recovery."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.activations import (
    ChipInputQuantizer,
    LIFActivation,
    run_cycle_accurate,
)
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class _CycleAccurateForward:
    """Picklable ``model.forward`` override that drives ``run_cycle_accurate``
    on top of the model's class-level forward, used during the LIF blend ramp."""

    def __init__(self, model, T: int):
        self.model = model
        self.T = int(T)

    def _call_unpatched_forward(self, x):
        return type(self.model).forward(self.model, x)

    def __call__(self, x):
        return run_cycle_accurate(
            self.model, x, self.T,
            forward_fn=self._call_unpatched_forward,
        )


class _ChipAlignedNFForward:
    """Picklable ``model.forward`` override installed post-blend (rate==1.0).
    Routes NF through ``chip_aligned_nf_forward`` so downstream calibrators
    (WQ, NormFusion, SCM probes) see the same forward the chip simulators run."""

    def __init__(self, model, T: int):
        self.model = model
        self.T = int(T)

    def __call__(self, x):
        from mimarsinan.spiking.chip_aligned_nf import chip_aligned_nf_forward

        return chip_aligned_nf_forward(self.model, x, self.T)


class LIFBlendActivation(nn.Module):
    """Linear blend of old_activation and LIFActivation controlled by rate."""

    def __init__(
        self,
        old_activation: nn.Module,
        lif_activation: LIFActivation,
        rate: float = 0.0,
    ):
        super().__init__()
        self.old_activation = old_activation
        self.lif_activation = lif_activation
        self.rate = float(rate)

    @property
    def activation_type(self) -> str:
        return "LIF" if self.rate >= 1.0 else "ReLU"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rate <= 0.0:
            return self.old_activation(x)
        if self.rate >= 1.0:
            return self.lif_activation(x)
        return (1.0 - self.rate) * self.old_activation(x) + self.rate * self.lif_activation(x)


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


class LIFAdaptationTuner(SmoothAdaptationTuner):
    """Ramp base_activation to LIFActivation with KD recovery."""

    def __init__(self, pipeline, model, target_accuracy, lr, adaptation_manager):
        super().__init__(pipeline, model, target_accuracy, lr)
        self.adaptation_manager = adaptation_manager
        self.name = "LIF Adaptation"
        self._T = int(pipeline.config["simulation_steps"])
        self._thresholding_mode = str(pipeline.config.get("thresholding_mode", "<="))
        self._cycle_accurate = bool(pipeline.config.get("cycle_accurate_lif_forward", False))
        self._patched_forward = False
        self._final_metric = None

        device = self.pipeline.config["device"]
        self.model.to("cpu")
        self._teacher = copy.deepcopy(self.model)
        self.model.to(device)
        self._teacher.to(device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad_(False)

        self._install_blend()
        self.trainer.loss_function = _KDClassificationLoss(self._teacher)

        if self._cycle_accurate:
            self._install_cycle_accurate_forward()

    def _install_cycle_accurate_forward(self) -> None:
        """Patch model.forward to run_cycle_accurate for the duration of the blend ramp."""
        assert "forward" not in self.model.__dict__, (
            "LIFAdaptationTuner: model.forward is already patched; double-install "
            "would shadow the prior wrapper. Call _after_run on the previous tuner "
            "first."
        )
        self._patched_forward = True
        self.model.forward = _CycleAccurateForward(model=self.model, T=self._T)

    def _install_blend(self) -> None:
        for perceptron in self.model.get_perceptrons():
            old_base = perceptron.base_activation
            lif = LIFActivation(
                T=self._T,
                activation_scale=perceptron.activation_scale,
                thresholding_mode=self._thresholding_mode,
                firing_mode=str(self.pipeline.config.get("firing_mode", "Default")),
            )
            perceptron.base_activation = LIFBlendActivation(old_base, lif, rate=0.0)
            if self._cycle_accurate:
                lif.use_cycle_accurate_trains = True
            self.adaptation_manager.update_activation(self.pipeline.config, perceptron)

            if getattr(perceptron, "is_encoding_layer", False):
                quantizer = ChipInputQuantizer(
                    T=self._T,
                    activation_scale=perceptron.input_activation_scale,
                )
                if isinstance(perceptron.input_activation, nn.Identity):
                    perceptron.input_activation = quantizer
                else:
                    perceptron.input_activation = nn.Sequential(
                        perceptron.input_activation, quantizer,
                    )

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
        try:
            self._continue_to_full_rate()
            self._set_rate(1.0)
        finally:
            if getattr(self, "_patched_forward", False):
                try:
                    del self.model.forward
                except AttributeError:
                    pass
                self._patched_forward = False

        self.adaptation_manager.lif_active = True
        for p in self.model.get_perceptrons():
            self.adaptation_manager.update_activation(self.pipeline.config, p)
        if self._cycle_accurate:
            from mimarsinan.spiking.lif_utils import apply_cycle_accurate_trains_to_model

            apply_cycle_accurate_trains_to_model(self.model, True)
            assert "forward" not in self.model.__dict__, (
                "LIFAdaptationTuner._after_run: model.forward is already patched; "
                "did the blend-ramp wrapper leak?"
            )
            self.model.forward = _ChipAlignedNFForward(self.model, self._T)

        self._final_metric = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return self._final_metric

    def validate(self):
        if self._final_metric is not None:
            return self._final_metric
        return self.trainer.validate()
