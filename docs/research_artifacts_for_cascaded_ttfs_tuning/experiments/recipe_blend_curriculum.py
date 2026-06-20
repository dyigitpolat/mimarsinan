"""Recipe blend_curriculum: ramp logit blend from ~lossless staircase to genuine cascade."""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, staircase_logits, teacher_logits,
    trainable_params,
)

NAME = "blend_curriculum"


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          lr=2e-3, bs=256, ramp_frac=0.7):
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ramp_end = max(1, int(ramp_frac * steps))
    for i, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        a = min(1.0, i / ramp_end)
        blend = (1 - a) * staircase_logits(flow, x) + a * genuine_logits(flow, x, S)
        loss = kd_ce_loss(blend, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return flow
