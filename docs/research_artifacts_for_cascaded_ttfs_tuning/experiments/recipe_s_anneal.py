"""S-annealing recipe: FT the genuine cascade from a coarse->fine timing grid, ending at target S."""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, teacher_logits, trainable_params,
)

NAME = "s_anneal"


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          lr=2e-3, bs=256, mults=(4, 2, 1)):
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    stage_steps = [steps // len(mults)] * len(mults)
    stage_steps[-1] += steps - sum(stage_steps)
    for m, n in zip(mults, stage_steps):
        S_cur = max(S, S * m)
        for x, y in batches(xtr, ytr, bs, n, seed):
            logits = genuine_logits(flow, x, S_cur)
            loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return flow
