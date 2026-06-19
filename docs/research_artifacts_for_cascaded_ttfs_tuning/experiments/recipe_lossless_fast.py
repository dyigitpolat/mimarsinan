"""SYNTHESIS: reach the high-S staircase (== ANN, lossless) cascaded single-spike
TTFS ceiling in MINIMAL wall-time, by composing the three Round-2 levers that each
beat the baselines but none of which alone is lossless AND fast.

CONFIRMED DIAGNOSIS (build on, do not contradict):
  - The genuine cascade forward is MONOTONIC in deploy-S: weights FT-d at a LOW S
    keep their accuracy when deployed at a HIGHER S (train-S16/deploy-S32 is FREE and
    holds ~0.937 at d=9). Deploying higher than trained never hurts.
  - The high-S STAIRCASE forward (complete-sum, cycle_accurate OFF) is the ANN ceiling
    and is MONOTONIC in S with CLEAN gradients (d=9 S32 0.965 == cont).
  - THE BUG IS HIGH-S TRAINING: the fire-once surrogate gradient degrades over the long
    high-S cascade (n_cycles = S + depth), so genuine-from-scratch at S=32 (0.911) is
    WORSE than the free-lunch deploy (0.937) and far below the S=32 ceiling (0.965).
    The optimum EXISTS and is reachable; direct high-S training fails to reach it.

WHAT EACH LEVER BOUGHT (ceiling 0.965; free 0.937; from-scratch 0.911 @ d9 s32):
  - L3 scurric (S-ladder warm-start, budget heavy at the top): 0.950 / 149s. Strongest
    d=9 ceiling-approacher (gap 0.015) — each small S bump is a gentle IN-BASIN
    continuation, never a cold high-S start. But its budget sat at the EXPENSIVE high-S
    rungs (~255ms/step), capping the speed.
  - L4 fast_lossless (cheap low-S bulk + short high-S refine): 0.948 / 88s, and d=6 is
    LOSSLESS (0.965). Per-step cost scales with S (n_cycles = S + depth), so spending
    the bulk where steps are CHEAP (S=16, ~120ms) and only a SHORT refine at the deploy
    S buys accuracy/second far better. But the d=9 high-S refine climbs only slowly.
  - L1 staircase_bridge (in-loop clean staircase ceiling gradient + genuine->staircase
    bridge KD): 0.952 @1500 steps. The ONLY lever that supplies the ceiling gradient
    directly; pays off on the DEEP cascade (where optimisation is hardest) but is
    UNSTABLE at d=6. (A staircase WARM-START poisons the basin — refuted; only the
    in-loop bridge term works, warmup_frac=0.)

THE SYNTHESIS (this file): a budget-aware S-ladder (scurric's in-basin low->high
continuation) where the budget is WEIGHTED TOWARD THE CHEAP LOW-S RUNGS (fast_lossless's
accuracy/second), and the TOP (deploy-S) rung optionally co-trains the in-loop
staircase-bridge anchor (L1) — which helps exactly the deep cascade where the ladder's
residual gap lives, and is auto-disabled on shallow models where it is unstable.

  per rung s in ladder (s <= deploy S, in-basin warm-start from the previous):
    loss = alpha*CE(genuine_s) + (1-alpha)*KD(genuine_s -> teacher_ANN)   # deploy path
         + [TOP RUNG ONLY, deep models]
           lambda_stair*CE(staircase) + beta*KD(genuine -> staircase.detach())  # ceiling
  per-channel theta co-trains throughout (high LR; firing-gain = the collapse root);
  weights unfreeze shallow->deep within each rung; cosine LR per rung.

The default ladder ends at the deploy S, with cheap rungs (16,24) carrying most of the
budget and the deploy rung a shorter, bridge-anchored refine. Reproduces the priors:
ladder=(deploy,) stage_weights=(1,) with bridge off == ~combo at deploy-S; a 2-rung
[16,S] == combo_swarm; uniform top-heavy weights == scurric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recipe_harness import (
    batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    staircase_logits, teacher_logits,
)
from recipe_combo import _weight_params_through

NAME = "lossless_fast"


def _kd(logits, target, T):
    return F.kl_div(F.log_softmax(logits / T, -1), F.softmax(target / T, -1),
                    reduction="batchmean") * T * T


def _phase(flow, thetas, xtr, ytr, base, S, *, steps, seed, n_perc, init_frac,
           w_lr, theta_lr, alpha, grad_clip, step0, bridge, T, lambda_stair, beta,
           theta_floor):
    """One S-rung: progressive shallow->deep weight unfreeze + theta co-train + KD,
    cosine-annealed within the rung; non-destructive (operates in place). When
    `bridge` is on, also adds the clean staircase ceiling gradient + genuine->staircase
    bridge KD (used on the TOP deploy-S rung of deep models)."""
    if steps <= 0:
        return
    start = max(1, round(n_perc * init_frac))
    schedule = [min(n_perc, start + round((n_perc - start) * t / max(1, steps - 1)))
                for t in range(steps)]
    depth = -1
    opt = sched = weights = None
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
        gen = genuine_logits(flow, x, S)
        loss = kd_ce_loss(gen, y, teacher_logits(base, x), alpha=alpha)
        if bridge:
            stair = staircase_logits(flow, x)          # clean ceiling gradient
            gen = genuine_logits(flow, x, S)           # re-enable cycle_accurate
            loss = (kd_ce_loss(gen, y, teacher_logits(base, x), alpha=alpha)
                    + lambda_stair * F.cross_entropy(stair, y)
                    + beta * _kd(gen, stair.detach(), T))
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], grad_clip)
        opt.step(); sched.step()
        if bridge:
            with torch.no_grad():
                for th in thetas:
                    th.clamp_(min=theta_floor)


def _ladder(S, ladder, weights):
    """Clamp rung S's to <= deploy S, dedupe consecutive equal rungs (summing their
    budget weights), and always end with a rung exactly at S (the deploy resolution)."""
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
    """Integer step counts proportional to weights, summing to total (remainder to the
    largest-fractional rungs)."""
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
          ladder=(16, 24, 32, 48), stage_weights=(1.0, 1.0, 2.0, 2.0),
          bridge_top=False, bridge_min_depth=8,
          T=3.0, lambda_stair=0.5, beta=1.0, theta_floor=1e-3):
    """Budget-aware S-ladder with an optional staircase-bridge top rung.

    `ladder` = the warm-up S values (each clamped to <= deploy S); a final rung at S is
    always appended. `stage_weights` = relative budget per rung. The default is the L3
    scurric ladder (top-heavy: budget concentrated at the high-S rungs, where the
    staircase ceiling lives and direct training fails) — the strongest d=9 ladder. Each
    rung is a gentle IN-BASIN continuation from the previous (theta + weights persist).

    For SHALLOW / lower-deploy-S models the optimisation is easier and cheaper; FRONT-
    loading the budget to the low-S rungs (stage_weights heavy on the small S) trades
    ~nothing in accuracy for wall-time (the per-step cost scales with S = n_cycles).

    `bridge_top` adds L1's in-loop staircase ceiling gradient on the deploy rung. It is
    OFF by default: the L1 bridge term is fragile and, composed into a budget-fragmented
    ladder, it DESTABILISES the deep deploy-S rung (d=9 S=32: 0.95 -> 0.87). It only
    paid off in L1 with ~1500 steps dedicated entirely to the bridge, not as a top rung.
    """
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    rungs, weights = _ladder(S, ladder, stage_weights)
    counts = _budget_split(steps, weights)
    use_bridge_top = bridge_top and n >= bridge_min_depth and rungs[-1] >= int(S)
    kw = dict(n_perc=n, init_frac=init_frac, w_lr=w_lr, theta_lr=theta_lr,
              alpha=alpha, grad_clip=grad_clip, T=T, lambda_stair=lambda_stair,
              beta=beta, theta_floor=theta_floor)
    step0 = 0
    last = len(rungs) - 1
    for i, (s_rung, n_steps) in enumerate(zip(rungs, counts)):
        bridge = use_bridge_top and i == last
        _phase(flow, thetas, xtr, ytr, base, s_rung, steps=n_steps, seed=seed,
               step0=step0, bridge=bridge, **kw)
        step0 += n_steps
    return flow
