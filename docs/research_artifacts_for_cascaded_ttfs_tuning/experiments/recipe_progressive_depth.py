"""Genuine-cascade FT + KD with progressive shallow->deep unfreezing of layer weights."""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, teacher_logits, trainable_params,
)

NAME = "progressive_depth"


def _set_trainable_through(flow, depth):
    """Train perceptrons [0, depth); freeze layer weights/bias of the rest."""
    for i, p in enumerate(flow.get_perceptrons()):
        req = i < depth
        p.layer.weight.requires_grad_(req)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(req)


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          lr=2e-3, bs=256, init_frac=1 / 3):
    perceptrons = flow.get_perceptrons()
    n = len(perceptrons)
    start = max(1, round(n * init_frac))
    schedule = [min(n, start + round((n - start) * t / max(1, steps - 1)))
                for t in range(steps)]

    depth = -1
    opt = sched_lr = None
    for step, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        if schedule[step] != depth:
            depth = schedule[step]
            _set_trainable_through(flow, depth)
            opt = torch.optim.Adam(trainable_params(flow), lr=lr)
            for g in opt.param_groups:
                g["initial_lr"] = lr
            sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=steps, last_epoch=step - 1)
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward(); opt.step(); sched_lr.step()
    return flow
