"""LEVER L1: staircase ceiling co-trained with a genuine->staircase BRIDGE distill.

The analytical STAIRCASE forward (cycle_accurate OFF) reaches the ANN ceiling at
high S with CLEAN gradients — unlike the genuine fire-once cascade, whose surrogate
gradient degrades over the long high-S cascade (n_cycles=S+depth).

FINDING that reshaped this recipe (measured at d=9 S=32, seed 0):
  - PHASE-1-only staircase training DOES reach the ceiling (staircase acc 0.965),
    but genuine-deploying those weights is CHANCE (0.102) — and stays chance at
    S=32/64/128/256. The per-sample correlation of genuine-vs-staircase logits is
    ~0.0 (NOT a death cascade — 47-86% of channels fire — but the greedy single-
    spike firing reconstructs a DIFFERENT function than the complete-sum staircase).
  - So "at high S genuine -> staircase" is FALSE for pure-staircase weights: a
    post-hoc per-channel theta revive cannot recover correlation-0 output, and a
    900-step genuine polish from the pure-staircase basin only reaches 0.866
    (WORSE than the 0.937 train-S16/deploy-S32 free lunch). Pure staircase weights
    land in a basin the genuine operator disagrees with maximally.

THE BRIDGE (what actually works): keep the GENUINE cascade in the loop from step 0
so the weights never leave a genuine~=staircase basin, while the CLEAN staircase
gradient supplies the ceiling signal:

  loss = alpha*CE(genuine) + (1-alpha)*KD(genuine -> teacher_ANN)        # deploy path
       + lambda_stair * CE(staircase)                                    # clean ceiling grad
       + beta * KD(genuine -> staircase.detach())                        # bridge: pull
                                                                         #   genuine onto its
                                                                         #   own lossless mode
Per-output-channel theta (the firing-gain, the collapse's root cause) co-trains
throughout with a high LR; weights unfreeze shallow->deep (eases the deep cascade).
The staircase term reaches the ceiling with a clean gradient; the bridge term keeps
the genuine output glued to that staircase output so what is learned is DEPLOYABLE.

VERDICT (seed 0): the LITERAL lever (staircase warm-start THEN bridge) is REFUTED —
a non-zero ``warmup_frac`` poisons the basin (genuine drops; d=9 800-step 0.933 ->
0.866 with a 25% warm-start). The staircase is only useful as an IN-THE-LOOP bridge
term, so ``warmup_frac`` defaults to 0. The salvaged co-train reaches genuine 0.933
@800 steps (137s) and 0.952 @1500 steps (254s) at d=9 S=32 — within ~1.3pp of the
0.965 ceiling, beating the 0.937 free-lunch and 0.911 from-scratch baselines. d=6 is
unstable under the bridge term (0.84-0.88, BELOW combo's 0.954): the deep cascade is
where the staircase anchor pays off; the shallow case is better served by ``combo``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    staircase_logits, teacher_logits, trainable_params,
)

NAME = "staircase_bridge"


def _kd(logits, target, T):
    return F.kl_div(F.log_softmax(logits / T, -1), F.softmax(target / T, -1),
                    reduction="batchmean") * T * T


def _weight_params_through(flow, depth):
    """Layer weight/bias params of perceptrons [0, depth); set requires_grad and
    return the currently-trainable ones (shallow->deep unfreeze)."""
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


def _staircase_warmup(flow, xtr, ytr, base, *, steps, bs, seed, lr, alpha):
    """Short clean-gradient staircase warm-start: get the WEIGHTS into the ceiling
    region fast before the (slower) genuine bridge co-train takes over."""
    if steps <= 0:
        return
    for p in flow.get_perceptrons():
        p.layer.weight.requires_grad_(True)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(True)
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    for x, y in batches(xtr, ytr, bs, steps, seed):
        loss = kd_ce_loss(staircase_logits(flow, x), y, teacher_logits(base, x),
                          alpha=alpha)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          warmup_frac=0.0, w_lr=2e-3, theta_lr=5e-2, bs=256, init_frac=1 / 3,
          alpha=0.3, T=3.0, lambda_stair=0.5, beta=1.0, theta_floor=1e-3,
          grad_clip=2.0):
    warm = max(0, round(steps * warmup_frac))
    bridge = max(1, steps - warm)

    _staircase_warmup(flow, xtr, ytr, base, steps=warm, bs=bs, seed=seed,
                      lr=w_lr, alpha=alpha)

    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    start = max(1, round(n * init_frac))
    schedule = [min(n, start + round((n - start) * t / max(1, bridge - 1)))
                for t in range(bridge)]

    depth = -1
    opt = sched_lr = weights = None
    for step, (x, y) in enumerate(batches(xtr, ytr, bs, bridge, seed + 1)):
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
                opt, T_max=bridge, last_epoch=step - 1)
        stair = staircase_logits(flow, x)           # clean ceiling gradient
        gen = genuine_logits(flow, x, S)            # deploy path (re-enables cycle_accurate)
        loss = (kd_ce_loss(gen, y, teacher_logits(base, x), alpha=alpha)
                + lambda_stair * F.cross_entropy(stair, y)
                + beta * _kd(gen, stair.detach(), T))
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(weights + list(thetas), grad_clip)
        opt.step(); sched_lr.step()
        with torch.no_grad():
            for t in thetas:
                t.clamp_(min=theta_floor)
    return flow
