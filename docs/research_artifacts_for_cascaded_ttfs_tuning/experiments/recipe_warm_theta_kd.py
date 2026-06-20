"""Recipe warm_theta_kd: joint co-training of per-channel theta + weights through the genuine cascade with KD."""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    teacher_logits, trainable_params,
)

NAME = "warm_theta_kd"


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          theta_lr=5e-2, w_lr=2e-3, alpha=0.3, bs=256):
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    theta_ids = {id(p) for p in thetas}
    weights = [p for p in trainable_params(flow) if id(p) not in theta_ids]
    opt = torch.optim.Adam([
        {"params": thetas, "lr": theta_lr},
        {"params": weights, "lr": w_lr},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for x, y in batches(xtr, ytr, bs, steps, seed):
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=alpha)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return flow
