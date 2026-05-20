"""Post-LIF chip-aligned KD finetune.

After ``LIFAdaptationTuner`` commits ``rate=1.0`` and the hybrid mapping is
built, this tuner installs ``ChipAlignedForward`` on the model and runs a
bounded recovery training pass against a frozen teacher snapshot of the
pre-finetune model. The goal is to nudge **host-side** encoding Perceptron
weights so the chip's actual SCM output (with per-core latency, integer
firing, uniform-encoded rate at structural boundaries) better matches what
the model was trained for. On-chip core weights stay frozen — gradients
only reach ``nn.Parameter``s reachable through the host-side compute ops.

This is the chip-aligned counterpart to ``LIFAdaptationTuner``: blend ramp
runs first (host perceptron forward), then this tuner takes over once the
LIF transition is complete.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.spiking.chip_aligned_forward import (
    install_chip_aligned_forward,
    uninstall_chip_aligned_forward,
)


class _KDChipAlignedLoss:
    """KD + CE loss against a pre-finetune teacher (LIF-adapted, pre-chip-aligned)."""

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


class LifChipAlignedFinetuneTuner:
    """Run a bounded KD finetune through ``ChipAlignedForward``.

    Lifecycle:

    1. Snapshot the model (pre-finetune teacher).
    2. Identify trainable params (encoding Perceptron host weights) — freeze
       the rest so the chip's quantized core weights are not perturbed.
    3. Install ``ChipAlignedForward`` with the hybrid mapping.
    4. Train ``num_epochs`` epochs with KD loss.
    5. Uninstall and restore parameter ``requires_grad`` state.
    """

    def __init__(
        self,
        pipeline,
        model: nn.Module,
        hybrid_mapping,
        *,
        num_epochs: int = 1,
        max_batches_per_epoch: int | None = None,
        lr: float | None = None,
    ):
        self.pipeline = pipeline
        self.model = model
        self.hybrid_mapping = hybrid_mapping
        self.num_epochs = int(num_epochs)
        self.max_batches_per_epoch = max_batches_per_epoch
        cfg = pipeline.config
        self.lr = float(lr if lr is not None else cfg.get("lr", 1e-4)) * 0.1
        self.device = cfg.get("device", "cpu")
        self._original_requires_grad: dict[int, bool] = {}
        self._teacher: nn.Module | None = None

    def _build_teacher(self) -> nn.Module:
        self.model.to("cpu")
        teacher = copy.deepcopy(self.model)
        self.model.to(self.device)
        teacher.to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    def _freeze_non_encoding_params(self) -> None:
        """Freeze every parameter that isn't a host-side encoding-Perceptron weight."""
        encoding_params: set[int] = set()
        get_perceptrons = getattr(self.model, "get_perceptrons", None)
        if callable(get_perceptrons):
            for p in get_perceptrons():
                if getattr(p, "is_encoding_layer", False):
                    for param in p.parameters():
                        encoding_params.add(id(param))
        for name, param in self.model.named_parameters():
            self._original_requires_grad[id(param)] = param.requires_grad
            param.requires_grad_(id(param) in encoding_params)

    def _restore_requires_grad(self) -> None:
        for _name, param in self.model.named_parameters():
            prior = self._original_requires_grad.get(id(param), True)
            param.requires_grad_(prior)

    def run(self) -> None:
        from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

        self._teacher = self._build_teacher()
        self._freeze_non_encoding_params()
        wrapper = install_chip_aligned_forward(self.model, self.pipeline.config, self.hybrid_mapping)
        loss_fn = _KDChipAlignedLoss(self._teacher)
        loader_factory = DataLoaderFactory(self.pipeline.data_provider_factory)
        train_loader = loader_factory.create(
            batch_size=int(self.pipeline.config.get("batch_size", 64)),
            split="training",
        )
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            uninstall_chip_aligned_forward(self.model)
            self._restore_requires_grad()
            return
        optimizer = torch.optim.AdamW(trainable, lr=self.lr)

        try:
            self.model.train()
            for _epoch in range(self.num_epochs):
                for batch_idx, (x, y) in enumerate(train_loader):
                    if self.max_batches_per_epoch is not None and batch_idx >= self.max_batches_per_epoch:
                        break
                    x = x.to(self.device)
                    y = y.to(self.device) if isinstance(y, torch.Tensor) else y
                    optimizer.zero_grad()
                    loss = loss_fn(self.model, x, y)
                    loss.backward()
                    optimizer.step()
        finally:
            uninstall_chip_aligned_forward(self.model)
            self._restore_requires_grad()
            self.model.eval()
