"""Staircase-BACKWARD STE: forward = genuine fire-once cascade, backward = clean
complete-sum staircase gradient. Round-2 proved the d=9 high-S plateau is the
fire-once surrogate GRADIENT (more budget HURTS), not the schedule. This injects
the staircase's clean ceiling gradient onto the EXACT genuine basin:

    ste = staircase + (genuine - staircase).detach()
    # forward value == genuine (the deploy path); gradient flows via staircase only

Trained with combo's machinery (per-channel theta co-train + progressive shallow->
deep unfreeze + KD + grad_clip). Eval is PURE genuine. If this crosses the ~0.95
plateau toward the 0.965 ceiling at d=9 S=32, the gradient was the bug.
"""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    staircase_logits, teacher_logits,
)
from recipe_combo import _weight_params_through

NAME = "staircase_ste"


def ste_logits(flow, x, S, mix=1.0):
    """Forward = genuine; backward = (1-mix)*genuine_surrogate + mix*staircase.
    mix=1 -> pure staircase backward; mix=0 -> pure genuine surrogate."""
    g = genuine_logits(flow, x, S)
    if mix <= 0.0:
        return g
    s = staircase_logits(flow, x)
    back = mix * s + (1.0 - mix) * g
    return back + (g - back).detach()


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          w_lr=2e-3, theta_lr=5e-2, bs=256, init_frac=1 / 3, alpha=0.3,
          grad_clip=1.0, mix=1.0):
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    start = max(1, round(n * init_frac))
    schedule = [min(n, start + round((n - start) * t / max(1, steps - 1)))
                for t in range(steps)]
    depth = -1
    opt = sched = None
    for t, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        if schedule[t] != depth:
            depth = schedule[t]
            weights = _weight_params_through(flow, depth)
            opt = torch.optim.Adam([
                {"params": thetas, "lr": theta_lr},
                {"params": weights, "lr": w_lr},
            ])
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=steps, last_epoch=t - 1)
        logits = ste_logits(flow, x, S, mix=mix)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=alpha)
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for grp in opt.param_groups for p in grp["params"]], grad_clip)
        opt.step(); sched.step()
    return flow
