"""Multi-stage S-curriculum: a non-destructive S ladder (16 -> 24 -> 32 -> 48)
that warm-starts each stage from the previous one's weights, spends MORE budget at
the high-S stages (where the staircase ceiling lives, and where direct training
fails), and carries per-channel theta + progressive shallow->deep depth + KD
throughout.

Rationale (Round 2 facts): training the genuine cascade directly at high S degrades
(long-cascade fire-once surrogate gradient), yet the high-S staircase ceiling IS the
ANN (lossless) and IS reachable in principle. A LOW-S basin trains cleanly; each
small S bump is a gentle, in-basin continuation rather than a cold high-S start. The
two-phase warm-start (recipe_combo_swarm) already beats both baselines (0.942 vs
0.937 deploy-S16 / 0.911 from-scratch); a finer ladder with budget concentrated at
the top closes more of the remaining gap to 0.965.

This subsumes combo_swarm: a 2-rung ladder [warm_S, S] with frac [warm_frac, ...]
reproduces it. The default ladder is 3-4 rungs up to (and slightly past) the deploy S.
"""

from __future__ import annotations

import torch

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    teacher_logits,
)
from recipe_combo import _weight_params_through

NAME = "scurric"


def _phase(flow, thetas, xtr, ytr, base, S, *, steps, seed, n_perc, init_frac,
           w_lr, theta_lr, alpha, grad_clip, step0):
    """One S-rung: progressive shallow->deep weight unfreeze + theta co-train + KD,
    cosine-annealed within the rung. Non-destructive (operates in place)."""
    if steps <= 0:
        return
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


def _ladder(S, ladder, weights):
    """Clamp the rung S's to <= deploy S (deploying higher than trained is a free
    lunch, but a rung above S would train a regime we never deploy and waste budget),
    dedupe consecutively-equal rungs (summing their budget weights), and pair each
    surviving rung with its budget weight. Always ends with a rung exactly at S."""
    rungs = [min(int(s), int(S)) for s in ladder]
    if not rungs or rungs[-1] != int(S):
        rungs = rungs + [int(S)]
        weights = list(weights) + [weights[-1] if weights else 1.0]
    out_s, out_w = [], []
    for s, w in zip(rungs, weights):
        if out_s and s == out_s[-1]:
            out_w[-1] += w
        else:
            out_s.append(s); out_w.append(float(w))
    return out_s, out_w


def _budget_split(total, weights):
    """Integer step counts proportional to weights, summing to total (remainder to
    the largest-weight rungs so the high-S stages keep their concentrated budget)."""
    wsum = sum(weights) or 1.0
    raw = [total * w / wsum for w in weights]
    base = [int(r) for r in raw]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - base[i], weights[i]),
                   reverse=True)
    for i in range(rem):
        base[order[i % len(base)]] += 1
    return base


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          w_lr=2e-3, theta_lr=5e-2, init_frac=1 / 3, alpha=0.3, grad_clip=1.0,
          ladder=(16, 24, 32, 48), stage_weights=(1.0, 1.0, 2.0, 2.0)):
    """Run the S ladder. `ladder` = the S values per stage; `stage_weights` =
    relative training budget per stage (heavier at high S). Each stage warm-starts
    from the previous (theta + weights persist in `flow`)."""
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    rungs, weights = _ladder(S, ladder, stage_weights)
    counts = _budget_split(steps, weights)
    kw = dict(n_perc=n, init_frac=init_frac, w_lr=w_lr, theta_lr=theta_lr,
              alpha=alpha, grad_clip=grad_clip)
    step0 = 0
    for s_rung, n_steps in zip(rungs, counts):
        _phase(flow, thetas, xtr, ytr, base, s_rung, steps=n_steps, seed=seed,
               step0=step0, **kw)
        step0 += n_steps
    return flow
