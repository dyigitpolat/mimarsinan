"""Reference recipe: genuine-cascade FT with continuous-teacher KD + cosine LR.

The current best automatic recipe (KD was the strongest depth regulariser:
d=9 S=16 0.84 -> 0.926). All other recipes are measured against this. Demonstrates
the recipe contract for recipe_harness.
"""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, teacher_logits, trainable_params,
)

NAME = "kd_baseline"


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed, lr=2e-3, bs=256):
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for x, y in batches(xtr, ytr, bs, steps, seed):
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return flow
