#!/usr/bin/env python
"""Torchrun-launchable DDP orchestrator for the fast ResNet-50 ImageNet run.

Wires the tested components in :mod:`mimarsinan.training.imagenet_fast_train`
(FastImageNetRecipe / one_cycle_lr_schedule / progressive_resize_schedule /
label_smoothing CE / build_resnet50_channels_last / train_step /
build_imagenet_dataloaders) into a full from-scratch ResNet-50 ImageNet run
targeting ~67% top-1 in well under an hour on 4x RTX PRO 6000 Blackwell.

Launch (4-GPU DDP)::

    torchrun --nproc_per_node=4 scripts/gpu/train_imagenet_fast.py \
        --out runs/imagenet/resnet50.pt

Design: the core loop (:func:`run` / :func:`train_one_epoch` / :func:`evaluate`)
is dependency-injected (loaders + model + recipe passed in) so it is unit-testable
on CPU with a tiny fake dataset and no torchrun/CUDA. The DDP / torchrun / CLI /
ImageNet-provider glue is a thin :func:`main` wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

# Make the repo's `src/` importable when run as a bare torchrun script.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import torch
import torch.nn as nn

from mimarsinan.training.imagenet_fast_train import (
    FastImageNetRecipe,
    build_imagenet_dataloaders,
    build_resnet50_channels_last,
    label_smoothing_cross_entropy,
    one_cycle_lr_schedule,
    progressive_resize_schedule,
    train_step,
)


# --------------------------------------------------------------------------- #
# Distributed env (torchrun) parsing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistInfo:
    """Parsed torchrun topology. Single-process when ``WORLD_SIZE`` is unset/1."""

    rank: int
    world_size: int
    local_rank: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def read_dist_env() -> DistInfo:
    """Read RANK / WORLD_SIZE / LOCAL_RANK from the environment (torchrun)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
    return DistInfo(rank=rank, world_size=max(1, world_size), local_rank=local_rank)


# --------------------------------------------------------------------------- #
# Core training primitives (dependency-injected; unit-testable on CPU)
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    lr_for_step: Callable[[int], float],
    smoothing: float = 0.1,
    use_amp: bool = True,
    channels_last: bool = True,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    global_step_offset: int = 0,
    max_steps: Optional[int] = None,
    epoch: int = 0,
    world_size: int = 1,
    log_fn: Optional[Callable[[dict], None]] = None,
    log_every: int = 0,
) -> float:
    """Run one epoch; set the per-step one-cycle LR; return the mean train loss.

    ``lr_for_step(global_step)`` is consulted once per optimizer step and pushed
    onto every param group BEFORE the step. ``global_step_offset`` is the number
    of steps completed in prior epochs (so the one-cycle schedule spans the run).
    ``max_steps`` caps the epoch (a smoke / dry real run); ``log_fn``+``log_every``
    emit a per-step progress line {epoch, step, loss, lr, imgs_per_s} so a long
    real run is observable without waiting a full ~13-min epoch for the summary.
    """
    model.train()
    total_loss = 0.0
    n_steps = 0
    for local_step, batch in enumerate(loader):
        if max_steps is not None and local_step >= max_steps:
            break
        step_t0 = time.time()
        lr = float(lr_for_step(global_step_offset + local_step))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss = train_step(
            model, batch, optimizer,
            device=device,
            smoothing=smoothing,
            use_amp=use_amp,
            channels_last=channels_last,
            scaler=scaler,
        )
        total_loss += loss
        n_steps += 1
        if log_fn is not None and log_every > 0 and local_step % log_every == 0:
            dt = max(1e-9, time.time() - step_t0)
            imgs = int(batch[0].shape[0]) * max(1, world_size)
            log_fn({"epoch": int(epoch), "step": int(local_step),
                    "loss": round(float(loss), 4), "lr": round(lr, 5),
                    "imgs_per_s": round(imgs / dt, 1)})
    return total_loss / max(1, n_steps)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Any,
    *,
    device: torch.device,
    channels_last: bool = True,
    world_size: int = 1,
) -> float:
    """Top-1 accuracy (%) over ``loader``; all-reduces correct/total under DDP."""
    model.eval()
    correct = torch.zeros(1, dtype=torch.long, device=device)
    total = torch.zeros(1, dtype=torch.long, device=device)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if channels_last and x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum()
        total += y.numel()

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)

    denom = int(total.item())
    if denom == 0:
        return float("nan")
    return 100.0 * int(correct.item()) / denom


def _should_eval(epoch: int, *, epochs: int, eval_every: int) -> bool:
    """True on every ``eval_every``-aligned epoch and ALWAYS on the final epoch.

    Alignment is ``epoch % eval_every == 0`` (0-based), so with ``eval_every=2``
    epochs 0 and 2 are evaluated and 1, 3 are skipped (the final epoch, whatever
    its index, is always evaluated because it carries the reported accuracy).
    """
    if epoch == epochs - 1:
        return True
    return eval_every >= 1 and (epoch % eval_every == 0)


# --------------------------------------------------------------------------- #
# The epoch loop (dependency-injected)
# --------------------------------------------------------------------------- #
def run(
    *,
    recipe: FastImageNetRecipe,
    model: nn.Module,
    build_train_loader: Callable[[int], Any],
    val_loader: Any,
    device: torch.device,
    out_path: str,
    is_main: bool = True,
    world_size: int = 1,
    eval_every: int = 1,
    log_fn: Callable[[dict], None] = print,
    use_amp: bool = True,
    channels_last: bool = True,
    set_epoch: Optional[Callable[[int], None]] = None,
    steps_per_epoch: Optional[int] = None,
    step_log_every: int = 0,
) -> dict:
    """Full ResNet-50 ImageNet epoch loop. Returns ``{val_top1, wall_seconds}``.

    ``build_train_loader(size)`` rebuilds the train loader at the epoch's
    progressive-resize size; ``set_epoch(epoch)`` (optional) reshuffles the
    DDP sampler. Only ``is_main`` writes the checkpoint to ``out_path``.
    """
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=recipe.peak_lr,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
        nesterov=recipe.nesterov,
    )
    amp_on = bool(use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_on) if amp_on else None

    sizes = progressive_resize_schedule(
        num_epochs=recipe.epochs,
        start_size=recipe.start_size,
        final_size=recipe.final_size,
        final_epochs=recipe.final_size_epochs,
    )

    # One-cycle LR spans the WHOLE run (all steps across all epochs). We need the
    # per-epoch step count; build the epoch-0 loader once up front to count it
    # unless the caller supplied it. The train loader is then (re)built once per
    # epoch at that epoch's progressive-resize size.
    pending_loader = build_train_loader(sizes[0])
    if steps_per_epoch is None:
        steps_per_epoch = _safe_len(pending_loader)
    total_steps = max(1, steps_per_epoch * recipe.epochs)
    warmup_steps = recipe.lr_schedule_steps(total_steps)

    def lr_for_step(global_step: int) -> float:
        return one_cycle_lr_schedule(
            global_step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=recipe.peak_lr,
            final_lr_frac=recipe.final_lr_frac,
        )

    wall_t0 = time.time()
    last_val_top1 = float("nan")

    for epoch in range(recipe.epochs):
        epoch_t0 = time.time()
        size = sizes[epoch]
        # Rebuild the train loader at this epoch's resize size (epoch 0 reuses the
        # already-built loader from the step-count probe above).
        if epoch == 0:
            train_loader = pending_loader
        else:
            train_loader = build_train_loader(size)
        if set_epoch is not None:
            set_epoch(epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            device=device,
            lr_for_step=lr_for_step,
            smoothing=recipe.label_smoothing,
            use_amp=use_amp,
            channels_last=channels_last,
            scaler=scaler,
            global_step_offset=epoch * steps_per_epoch,
            max_steps=steps_per_epoch,
            epoch=epoch,
            world_size=world_size,
            log_fn=log_fn,
            log_every=step_log_every,
        )

        if _should_eval(epoch, epochs=recipe.epochs, eval_every=eval_every):
            val_top1 = evaluate(
                model, val_loader, device=device,
                channels_last=channels_last, world_size=world_size,
            )
            last_val_top1 = val_top1
        else:
            val_top1 = float("nan")

        log_line = {
            "epoch": epoch,
            "img_size": int(size),
            "lr": float(lr_for_step(epoch * steps_per_epoch)),
            "train_loss": float(train_loss),
            "val_top1": float(val_top1),
            "epoch_seconds": round(time.time() - epoch_t0, 3),
        }
        log_fn(log_line)

    wall_seconds = time.time() - wall_t0

    if is_main:
        _save_checkpoint(model, val_top1=last_val_top1, out_path=out_path)

    return {"val_top1": float(last_val_top1), "wall_seconds": float(wall_seconds)}


def _safe_len(loader: Any) -> int:
    try:
        return len(loader)
    except TypeError:
        return 1


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module (peel DistributedDataParallel if wrapped)."""
    return getattr(model, "module", model)


def _save_checkpoint(model: nn.Module, *, val_top1: float, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": _unwrap(model).state_dict(), "val_top1": float(val_top1)},
        str(out),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DDP fast ResNet-50 ImageNet-from-scratch orchestrator."
    )
    defaults = FastImageNetRecipe()
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size,
                   help="GLOBAL batch (summed across GPUs under DDP).")
    p.add_argument("--workers", type=int, default=defaults.num_workers,
                   help="DataLoader workers PER process.")
    p.add_argument("--data-root", type=str, default="/data/ImageNet")
    p.add_argument("--out", type=str, default="runs/imagenet/resnet50.pt")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Evaluate every K epochs (the final epoch is always evaluated).")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Cap steps/epoch (0 = full epoch); >0 = smoke / dry real run.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Per-step progress log cadence (0 = off).")
    return p


# --------------------------------------------------------------------------- #
# Real-data glue: build the injectable callables over the ImageNet provider.
# --------------------------------------------------------------------------- #
def _build_real_train_loader_factory(provider, *, per_proc_batch, num_workers, dist):
    """Return build_train_loader(size) that rebuilds the train loader at ``size``.

    Progressive resize: the provider's train transform is fixed at 224, so we
    re-compose a RandomResizedCrop(size) pipeline over the provider's raw train
    subset and wrap it (DistributedSampler under DDP) into a fresh DataLoader.
    """
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, DistributedSampler
    from mimarsinan.data_handling.dataset_views import ApplyTransform

    raw_train = provider.raw_datasets()["train"]

    def build(size: int):
        tfm = provider._wrap_with_preprocessing([
            T.RandomResizedCrop(int(size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        dataset = ApplyTransform(raw_train, tfm)
        sampler = None
        shuffle = True
        if dist.is_distributed:
            sampler = DistributedSampler(
                dataset, num_replicas=dist.world_size, rank=dist.rank,
                shuffle=True, drop_last=True,
            )
            shuffle = False
        loader_kwargs = dict(
            batch_size=per_proc_batch, num_workers=num_workers,
            pin_memory=True, drop_last=True,
        )
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 4
        loader = DataLoader(
            dataset, sampler=sampler, shuffle=(sampler is None) and shuffle,
            **loader_kwargs,
        )
        return loader, sampler

    return build


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dist = read_dist_env()

    if dist.is_distributed:
        torch.distributed.init_process_group(backend="nccl")

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(dist.local_rank)
        device = torch.device("cuda", dist.local_rank)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    recipe = FastImageNetRecipe(epochs=args.epochs, batch_size=args.batch_size,
                                num_workers=args.workers)
    per_proc_batch = max(1, recipe.batch_size // dist.world_size)

    # --- Data provider + injectable loaders (torchvision fallback path) ------
    from mimarsinan.data_handling.data_providers.imagenet_data_provider import (
        ImageNet_DataProvider,
    )
    provider = ImageNet_DataProvider(args.data_root, batch_size=per_proc_batch)
    loaders = build_imagenet_dataloaders(
        provider=provider, batch_size=per_proc_batch,
        num_workers=args.workers, prefer_ffcv=False,
        device=("cuda" if use_cuda else "cpu"),
    )
    val_loader = loaders["val"]

    train_factory = _build_real_train_loader_factory(
        provider, per_proc_batch=per_proc_batch, num_workers=args.workers, dist=dist,
    )
    # The factory returns (loader, sampler); adapt to run()'s build/set_epoch API.
    _active_sampler = {"s": None}

    def build_train_loader(size: int):
        loader, sampler = train_factory(size)
        _active_sampler["s"] = sampler
        return loader

    def set_epoch(epoch: int) -> None:
        sampler = _active_sampler["s"]
        if sampler is not None:
            sampler.set_epoch(epoch)

    # --- Model (DDP-wrapped) -------------------------------------------------
    model = build_resnet50_channels_last(recipe.num_classes).to(device)
    if dist.is_distributed and use_cuda:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dist.local_rank])

    def log_fn(line: dict) -> None:
        if dist.is_main:
            print(json.dumps(line), flush=True)

    result = run(
        recipe=recipe,
        model=model,
        build_train_loader=build_train_loader,
        val_loader=val_loader,
        device=device,
        out_path=args.out,
        is_main=dist.is_main,
        world_size=dist.world_size,
        eval_every=args.eval_every,
        log_fn=log_fn,
        use_amp=recipe.use_amp,
        channels_last=recipe.channels_last,
        set_epoch=set_epoch,
        steps_per_epoch=(args.max_steps or None),
        step_log_every=args.log_every,
    )

    if dist.is_main:
        print(
            f"DONE top1={result['val_top1']:.3f}% "
            f"wall={result['wall_seconds']/60.0:.1f}min "
            f"(world_size={dist.world_size})",
            flush=True,
        )

    if dist.is_distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
