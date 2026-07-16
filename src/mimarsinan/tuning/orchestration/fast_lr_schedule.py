"""The fast ladder's spanning warmup+cosine LR schedule: build and scale."""

from __future__ import annotations

import torch


def build_fast_lr_schedule(optimizer, total_steps, eta_min=0.0):
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


def scale_fast_lr(optimizer, schedule, factor: float) -> None:
    """[WS-A A1] Armijo backoff for the retention gate: scale every optimizer
    group AND the spanning schedule's children (base_lrs + cosine eta_min)
    together — schedule.step() rewrites group lrs from base_lrs, so a naive
    group mutation would be overwritten next step. Uniform factor composes
    with LLRD's per-group lrs; the optimizer is never rebuilt (the rung
    snapshot deliberately preserves Adam moments)."""
    factor = float(factor)
    for group in optimizer.param_groups:
        group["lr"] *= factor
        if "initial_lr" in group:
            group["initial_lr"] *= factor
    for child in schedule._schedulers:
        child.base_lrs = [lr * factor for lr in child.base_lrs]
        if isinstance(child, torch.optim.lr_scheduler.CosineAnnealingLR):
            child.eta_min *= factor
