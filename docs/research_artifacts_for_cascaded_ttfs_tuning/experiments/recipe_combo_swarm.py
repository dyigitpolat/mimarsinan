"""Combo with an S WARM-START curriculum (low->high), to break the high-S
optimisation barrier. Higher S raises the staircase ceiling toward the ANN, but
training the genuine cascade directly at high S degrades (long-cascade surrogate
gradient). So train combo at an EASY low S (good basin), then CONTINUE at the high
deploy-S from those cascade-adapted weights — the opposite of the failed high->low
s_anneal. Deploy S = the final (high) S.
"""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    teacher_logits,
)
from recipe_combo import _weight_params_through

NAME = "combo_swarm"


def _phase(flow, thetas, xtr, ytr, base, S, *, steps, seed, n_perc, init_frac,
           w_lr, theta_lr, alpha, grad_clip, step0):
    start = max(1, round(n_perc * init_frac))
    schedule = [min(n_perc, start + round((n_perc - start) * t / max(1, steps - 1)))
                for t in range(steps)]
    depth = -1
    opt = sched = None
    for t, (x, y) in enumerate(batches(xtr, ytr, 256, steps, seed + step0)):
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
        loss = kd_ce_loss(genuine_logits(flow, x, S), y, teacher_logits(base, x),
                          alpha=alpha)
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], grad_clip)
        opt.step(); sched.step()


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          w_lr=2e-3, theta_lr=5e-2, init_frac=1 / 3, alpha=0.3, grad_clip=1.0,
          warm_S=16, warm_frac=0.5):
    """Phase 1: warm_frac of steps at min(warm_S, S) (easy basin). Phase 2: the rest
    at the target S (the deploy resolution)."""
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    s_lo = min(int(warm_S), int(S))
    n1 = int(warm_frac * steps)
    kw = dict(n_perc=n, init_frac=init_frac, w_lr=w_lr, theta_lr=theta_lr,
              alpha=alpha, grad_clip=grad_clip)
    if s_lo < S and n1 > 0:
        _phase(flow, thetas, xtr, ytr, base, s_lo, steps=n1, seed=seed, step0=0, **kw)
    _phase(flow, thetas, xtr, ytr, base, S, steps=steps - n1, seed=seed, step0=n1, **kw)
    return flow
