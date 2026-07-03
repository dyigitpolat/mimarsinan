"""Fast ResNet-50 ImageNet-from-scratch recipe (FFCV / fast.ai super-convergence style)."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.models.pretrained_bridge import load_pretrained_resnet50


@dataclass
class FastImageNetRecipe:
    """Super-convergence ResNet-50 recipe knobs.

    ``peak_lr`` is the peak for the listed (global, DDP-summed) ``batch_size``;
    scale it linearly with batch size.
    """

    num_classes: int = 1000
    epochs: int = 16

    optimizer: str = "sgd"
    momentum: float = 0.9
    weight_decay: float = 5e-5
    nesterov: bool = True

    peak_lr: float = 2.0
    warmup_frac: float = 0.10
    final_lr_frac: float = 0.0

    label_smoothing: float = 0.1
    batch_size: int = 512

    start_size: int = 160
    final_size: int = 192
    final_size_epochs: int = 2
    eval_size: int = 224

    use_amp: bool = True
    channels_last: bool = True
    prefer_ffcv: bool = True

    num_workers: int = 12

    def lr_schedule_steps(self, total_steps: int) -> int:
        return max(0, int(round(self.warmup_frac * total_steps)))


def one_cycle_lr_schedule(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    final_lr_frac: float = 0.0,
) -> float:
    """Return the LR for ``step`` under a one-cycle policy: linear ramp 0→peak over
    ``warmup_steps``, then cosine anneal peak→``peak*final_lr_frac``.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    step = max(0, min(int(step), total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    decay_total = (total_steps - 1) - warmup_steps
    if decay_total <= 0:
        return peak_lr
    progress = (step - warmup_steps) / decay_total
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (final_lr_frac + (1.0 - final_lr_frac) * cosine)


def progressive_resize_schedule(
    *,
    num_epochs: int,
    start_size: int,
    final_size: int,
    final_epochs: int,
    multiple: int = 32,
) -> list[int]:
    """Per-epoch training image size: linear ramp ``start_size``→``final_size`` over the
    first ``num_epochs - final_epochs`` epochs then hold; sizes snapped to ``multiple``, monotone non-decreasing.
    """
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {num_epochs}")
    if final_size < start_size:
        raise ValueError(f"final_size {final_size} < start_size {start_size}")
    final_epochs = max(1, min(int(final_epochs), num_epochs))

    ramp_epochs = num_epochs - final_epochs
    sizes: list[int] = []
    span = final_size - start_size
    for e in range(num_epochs):
        if e >= ramp_epochs or ramp_epochs <= 0:
            sizes.append(final_size)
            continue
        frac = e / ramp_epochs
        raw = start_size + span * frac
        snapped = int(round(raw / multiple) * multiple)
        snapped = max(start_size, min(final_size, snapped))
        sizes.append(snapped)

    for i in range(1, len(sizes)):
        if sizes[i] < sizes[i - 1]:
            sizes[i] = sizes[i - 1]
    sizes[-1] = final_size
    return sizes


def label_smoothing_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """Mean label-smoothed CE, matching ``nn.CrossEntropyLoss(label_smoothing=s)``."""
    return F.cross_entropy(logits, targets, label_smoothing=float(smoothing))


def build_resnet50_channels_last(num_classes: int = 1000) -> nn.Module:
    """ResNet-50 (random init, ``fc`` re-sized) in channels-last, via the repo's
    pretrained bridge so it matches the deployable trunk.
    """
    model = load_pretrained_resnet50(num_classes, pretrained=False).train()
    return model.to(memory_format=torch.channels_last)


def train_step(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    smoothing: float = 0.1,
    use_amp: bool = True,
    channels_last: bool = True,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> float:
    """Run one forward/backward/step on ``batch``; return the scalar loss.

    AMP only on CUDA; CPU stays fp32 so the loss-decreases unit test is deterministic.
    """
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if channels_last and x.dim() == 4:
        x = x.to(memory_format=torch.channels_last)

    optimizer.zero_grad(set_to_none=True)
    amp_on = bool(use_amp) and device.type == "cuda"

    if amp_on:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = label_smoothing_cross_entropy(model(x), y, smoothing=smoothing)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
    else:
        loss = label_smoothing_cross_entropy(model(x), y, smoothing=smoothing)
        loss.backward()
        optimizer.step()

    return float(loss.detach().item())


def _ffcv_available() -> bool:
    """True iff ``import ffcv`` succeeds in this env. Guarded — never hard-imports."""
    try:
        importlib.import_module("ffcv")
        return True
    except Exception:
        return False


def _build_ffcv_loaders(provider: Any, **kwargs: Any):
    """Build FFCV loaders via the repo's :class:`FFCVLoaderFactory`."""
    from mimarsinan.data_handling.ffcv.loader_factory import FFCVLoaderFactory

    num_workers = int(kwargs.get("num_workers", 12))
    device = kwargs.get("device", "cuda")
    batch_size = int(kwargs["batch_size"])

    factory = FFCVLoaderFactory(
        data_provider_factory=_StaticProviderFactory(provider),
        num_workers=num_workers,
        device=device,
    )
    return {
        "train": factory.create_training_loader(batch_size, provider),
        "val": factory.create_validation_loader(batch_size, provider),
        "test": factory.create_test_loader(batch_size, provider),
    }


def _build_torchvision_loaders(provider: Any, **kwargs: Any):
    """Optimized torchvision fallback DataLoaders, delegating split assembly to the
    provider's ``fast_fallback_dataloaders`` so crop/normalize policy stays single-sourced.
    """
    batch_size = int(kwargs["batch_size"])
    num_workers = int(kwargs.get("num_workers", 12))
    return provider.fast_fallback_dataloaders(
        batch_size=batch_size, num_workers=num_workers
    )


class _StaticProviderFactory:
    """Adapter exposing ``.create()`` for an already-constructed provider."""

    def __init__(self, provider: Any):
        self._provider = provider

    def create(self):
        return self._provider


def build_imagenet_dataloaders(
    *,
    provider: Any,
    batch_size: int,
    num_workers: int = 12,
    prefer_ffcv: bool = True,
    device: str = "cuda",
):
    """Return train/val/test loaders, preferring FFCV when available (guarded runtime
    probe) else the optimized torchvision fallback.
    """
    if prefer_ffcv and _ffcv_available():
        return _build_ffcv_loaders(
            provider, batch_size=batch_size, num_workers=num_workers, device=device
        )
    return _build_torchvision_loaders(
        provider, batch_size=batch_size, num_workers=num_workers
    )
