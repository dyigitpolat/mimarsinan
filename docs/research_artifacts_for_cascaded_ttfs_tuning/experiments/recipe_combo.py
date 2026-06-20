"""Combine the two near-lossless winners: progressive shallow->deep weight
unfreezing (eases the deep-cascade optimisation) + joint per-channel theta
co-training (tunes the firing-gain, the collapse's root cause) + continuous KD.

progressive_depth and warm_theta_kd each reach ~0.948 at d=9 (within ~1pp of the
staircase) attacking DIFFERENT bottlenecks; this fuses them to close the last ~1pp.
theta (per-channel, high LR) trains throughout; weight params unfreeze shallow->deep.
"""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    teacher_logits,
)

NAME = "combo"


def _weight_params_through(flow, depth):
    """Layer weight/bias params of perceptrons [0, depth); set requires_grad and
    return the currently-trainable ones."""
    out = []
    for i, p in enumerate(flow.get_perceptrons()):
        req = i < depth
        p.layer.weight.requires_grad_(req)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(req)
        if req:
            out.append(p.layer.weight)
            if p.layer.bias is not None:
                out.append(p.layer.bias)
    return out


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          w_lr=2e-3, theta_lr=5e-2, bs=256, init_frac=1 / 3, alpha=0.3,
          grad_clip=None):
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    start = max(1, round(n * init_frac))
    schedule = [min(n, start + round((n - start) * t / max(1, steps - 1)))
                for t in range(steps)]

    depth = -1
    opt = sched_lr = None
    for step, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        if schedule[step] != depth:
            depth = schedule[step]
            weights = _weight_params_through(flow, depth)
            opt = torch.optim.Adam([
                {"params": thetas, "lr": theta_lr},
                {"params": weights, "lr": w_lr},
            ])
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]
            sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=steps, last_epoch=step - 1)
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=alpha)
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], grad_clip)
        opt.step(); sched_lr.step()
    return flow
