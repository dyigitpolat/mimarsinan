"""LEVER L4 (SPEED): exploit the genuine-cascade free lunch + a minimal high-S refine.

The genuine cascade forward is MONOTONIC in deploy-S: weights fine-tuned at a LOW
S (short cascade, n_cycles = S + depth, so fast) keep their accuracy when deployed
at a HIGHER S (train-S16/deploy-S32 is free and holds ~0.937 at d=9). High-S FROM
SCRATCH is the bug (long-cascade fire-once surrogate degrades, 0.911 < free 0.937).

So: spend the bulk of the budget where steps are CHEAP (S=`low_S`, combo reaches its
~0.94 ceiling fast), bank the free lunch by deploying at the high target S, then climb
above the free baseline with a SHORT genuine refine AT the target S (few hundred steps,
low LR, KD+CE, theta still trainable, grad-clipped). The refine sits in a good basin
(the free-lunch weights), so it adds high-S timing capacity without the cold-cascade
collapse — buying accuracy/second far better than training at high S throughout.

Pareto knobs: `low_S` (cheap phase resolution), `low_frac` (fraction of `steps` spent
at low_S), `refine_lr`. With steps as the wall-time dial, this traces the
accuracy/wall-time Pareto for "best genuine acc reachable in <120s / <300s".
"""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    teacher_logits, trainable_params,
)
import recipe_combo

NAME = "fast_lossless"


def _refine_genuine(flow, xtr, ytr, base, teacher, S, steps, *, lr, bs, alpha,
                    grad_clip, seed):
    """Short genuine cascade FT AT the deploy resolution S, from the free-lunch basin.
    theta + unfrozen weights (already set by the low-S combo phase) are co-tuned."""
    if steps <= 0:
        return
    params = trainable_params(flow)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for x, y in batches(xtr, ytr, bs, steps, seed):
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=alpha)
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step(); sched.step()


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          low_S=16, low_frac=0.75, refine_lr=7e-4, w_lr=2e-3, theta_lr=5e-2,
          bs=256, alpha=0.3, grad_clip=1.0):
    """`steps` is the TOTAL step budget (the wall-time dial). `low_frac` of it runs
    the cheap combo at `low_S`; the rest is the genuine refine at the target S.

    If `low_S >= S` (or low_frac == 1) this degenerates to plain combo at low_S then
    a free-lunch deploy at S — the SPEED floor of the lever."""
    low_steps = int(round(steps * low_frac))
    refine_steps = steps - low_steps

    recipe_combo.train(
        flow, xtr, ytr, xva, yva, low_S, base, teacher,
        steps=max(1, low_steps), seed=seed,
        w_lr=w_lr, theta_lr=theta_lr, bs=bs, alpha=alpha, grad_clip=grad_clip)

    if S > low_S:
        _refine_genuine(
            flow, xtr, ytr, base, teacher, S, refine_steps,
            lr=refine_lr, bs=bs, alpha=alpha, grad_clip=grad_clip, seed=seed + 1)
    return flow
