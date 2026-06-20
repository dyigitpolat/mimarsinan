"""Recipe surrogate_anneal: genuine-cascade FT + KD with a wide->sharp ATan surrogate anneal (revive dead neurons via wide gradient, sharpen to the deploy node)."""

from __future__ import annotations

import torch

from recipe_harness import (
    TTFSActivation, batches, genuine_logits, kd_ce_loss, teacher_logits,
    trainable_params,
)

NAME = "surrogate_anneal"


def _set_alpha(flow, a):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_surrogate_alpha(a)


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          lr=2e-3, bs=256, alpha_min=0.3, alpha_max=2.0):
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ratio = alpha_max / alpha_min
    for i, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        r = i / max(steps - 1, 1)
        _set_alpha(flow, alpha_min * ratio ** r)
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    _set_alpha(flow, alpha_max)
    return flow
