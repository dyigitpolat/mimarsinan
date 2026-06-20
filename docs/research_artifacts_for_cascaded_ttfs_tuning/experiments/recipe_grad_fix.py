"""Recipe grad_fix: make GENUINE FT at HIGH S climb to the staircase ceiling.

The bug (confirmed): the fire-once surrogate gradient degrades over the long high-S
cascade (n_cycles = S + depth), so genuine FT-from-scratch at S=32 regresses to ~0.911
-- BELOW even deploying S=16 weights at S=32 (0.937) and far under the S=32 staircase
ceiling (~0.965 == the ANN == lossless). The optimum exists (train-low/deploy-high keeps
it); training just can't reach it through the degraded long-cascade surrogate.

This recipe = the `combo` skeleton (per-channel theta co-train + progressive shallow->
deep weight unfreeze + KD + grad_clip) PLUS three gradient fixes, each independently
toggleable so we can report which one carries:

  (a) surrogate-alpha anneal WIDE -> SHARP. The ATan surrogate's `alpha` is INVERSE-width
      (atan_backward grad ~ alpha/2 / (1+(pi/2*alpha*x)^2)): SMALL alpha = WIDE/smooth
      gradient (every neuron in the long cascade gets credit even when far from its fire
      boundary -> revives the death cascade), LARGE alpha = SHARP (matches the hard deploy
      node). Anneal alpha_min -> alpha_max so early training is well-conditioned across
      depth and late training sharpens onto the genuine node.  `alpha_anneal=True`.

  (b) staircase-anchored JOINT loss (the carry). Train the SAME weights through BOTH
      forwards: genuine-CE/KD (deploy path, degraded gradient) + a staircase term with a
      CLEAN gradient that pulls toward the ceiling basin. Two staircase pulls, both on:
        - staircase KD to the continuous teacher  (anchor weights in the ANN basin)
        - genuine->staircase self-distill          (close the greedy-firing residual)
      The clean staircase gradient does the long-range credit assignment the surrogate
      can't; the genuine term keeps the deploy node honest.  `stair_w>0`, `selfdistill_w>0`.

  (c) per-DEPTH gradient normalization. Deep/late perceptrons' weight grads are swamped by
      the long-cascade surrogate decay; renormalize each perceptron's weight-grad to unit
      RMS (scaled by `gnorm_scale`) so deep layers are not starved.  `grad_norm=True`.

Defaults turn ALL THREE on (the shipped recipe); the __main__ ablation isolates each.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from recipe_harness import (
    TTFSActivation, batches, genuine_logits, kd_ce_loss, promote_theta_per_channel,
    staircase_logits, teacher_logits,
)

NAME = "grad_fix"


def _set_alpha(flow, a):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_surrogate_alpha(a)


def _set_cycle_accurate(flow, mode):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(mode)


def _weight_params_through(flow, depth):
    """Layer weight/bias params of perceptrons [0, depth); set requires_grad and
    return the (param, perceptron_index) pairs that are currently trainable."""
    out = []
    for i, p in enumerate(flow.get_perceptrons()):
        req = i < depth
        p.layer.weight.requires_grad_(req)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(req)
        if req:
            out.append((p.layer.weight, i))
            if p.layer.bias is not None:
                out.append((p.layer.bias, i))
    return out


def _normalize_grad_per_depth(weight_pairs, n_layers, scale):
    """(c) Rescale each weight/bias grad to unit RMS so the long-cascade surrogate
    decay doesn't starve deep layers. scale keeps the overall step size sane."""
    for param, _i in weight_pairs:
        if param.grad is None:
            continue
        rms = param.grad.detach().pow(2).mean().sqrt()
        if rms > 1e-12:
            param.grad.mul_(scale / rms)


def _staircase_joint_loss(flow, x, y, base, *, alpha, stair_w, selfdistill_w, T,
                          gen_logits):
    """(b) clean-gradient staircase anchor: staircase-KD-to-teacher + genuine->staircase
    self-distill. Returns the added loss; restores cycle_accurate=True for the genuine path."""
    extra = gen_logits.new_zeros(())
    if stair_w <= 0 and selfdistill_w <= 0:
        return extra
    stair = staircase_logits(flow, x)            # clean-gradient complete-sum forward
    if stair_w > 0:
        teach = teacher_logits(base, x).to(stair.dtype)
        kd = F.kl_div(F.log_softmax(stair / T, -1), F.softmax(teach / T, -1),
                      reduction="batchmean") * T * T
        extra = extra + stair_w * (alpha * F.cross_entropy(stair, y) + (1 - alpha) * kd)
    if selfdistill_w > 0:
        # pull the genuine (deploy) output onto this network's lossless staircase output
        target = stair.detach()
        sd = F.kl_div(F.log_softmax(gen_logits / T, -1), F.softmax(target / T, -1),
                      reduction="batchmean") * T * T
        extra = extra + selfdistill_w * sd
    _set_cycle_accurate(flow, True)              # staircase_logits flipped it OFF
    return extra


def train(flow, xtr, ytr, xva, yva, S, base, teacher, *, steps, seed,
          w_lr=2e-3, theta_lr=5e-2, bs=256, init_frac=1 / 3, alpha=0.3,
          grad_clip=1.0, T=3.0,
          alpha_anneal=True, alpha_min=0.3, alpha_max=2.0,
          stair_w=1.0, selfdistill_w=1.0,
          grad_norm=True, gnorm_scale=1e-3):
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    perceptrons = flow.get_perceptrons()
    n = len(perceptrons)
    start = max(1, round(n * init_frac))
    schedule = [min(n, start + round((n - start) * t / max(1, steps - 1)))
                for t in range(steps)]
    ratio = (alpha_max / alpha_min) if alpha_anneal else 1.0

    depth = -1
    opt = sched_lr = weight_pairs = None
    for step, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        if schedule[step] != depth:
            depth = schedule[step]
            weight_pairs = _weight_params_through(flow, depth)
            weights = [p for p, _ in weight_pairs]
            opt = torch.optim.Adam([
                {"params": thetas, "lr": theta_lr},
                {"params": weights, "lr": w_lr},
            ])
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]
            sched_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=steps, last_epoch=step - 1)

        if alpha_anneal:
            r = step / max(steps - 1, 1)
            _set_alpha(flow, alpha_min * ratio ** r)

        gen = genuine_logits(flow, x, S)
        loss = kd_ce_loss(gen, y, teacher_logits(base, x), alpha=alpha)
        loss = loss + _staircase_joint_loss(
            flow, x, y, base, alpha=alpha, stair_w=stair_w,
            selfdistill_w=selfdistill_w, T=T, gen_logits=gen)

        opt.zero_grad()
        loss.backward()
        if grad_norm:
            _normalize_grad_per_depth(weight_pairs, n, gnorm_scale)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], grad_clip)
        opt.step()
        sched_lr.step()

    if alpha_anneal:
        _set_alpha(flow, alpha_max)
    _set_cycle_accurate(flow, True)
    return flow


if __name__ == "__main__":
    import copy
    import time

    from recipe_harness import build, genuine_acc
    from lif_vs_ttfs import ttfs_staircase_acc
    from chase_2pp import _val_split

    DEPLOY_S16_AT_S32 = {6: 0.965, 9: 0.937}   # train-S16/deploy-S32 free-lunch baseline
    FROM_SCRATCH_S32 = {6: None, 9: 0.911}     # genuine FT-from-scratch at S32

    STEPS = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 400
    CONFIGS = [(9, 32), (6, 32)]

    # Each variant isolates one fix on top of the combo skeleton; "all" = shipped recipe.
    VARIANTS = {
        "combo_base":  dict(alpha_anneal=False, stair_w=0.0, selfdistill_w=0.0, grad_norm=False),
        "a_alpha":     dict(alpha_anneal=True,  stair_w=0.0, selfdistill_w=0.0, grad_norm=False),
        "b_stairKD":   dict(alpha_anneal=False, stair_w=1.0, selfdistill_w=0.0, grad_norm=False),
        "b_selfdist":  dict(alpha_anneal=False, stair_w=0.0, selfdistill_w=1.0, grad_norm=False),
        "b_both":      dict(alpha_anneal=False, stair_w=1.0, selfdistill_w=1.0, grad_norm=False),
        "c_gradnorm":  dict(alpha_anneal=False, stair_w=0.0, selfdistill_w=0.0, grad_norm=True),
        "all":         dict(),  # all defaults on
    }
    only = __import__("sys").argv[2:] or None

    for depth, S in CONFIGS:
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=0)
        ceiling = ttfs_staircase_acc(flow, xte, yte, S)
        print(f"\n=== d={depth} S={S}  cont={cont:.4f}  staircase_ceiling={ceiling:.4f}  "
              f"steps={STEPS} ===")
        print(f"  baselines: deploy-S16@S32={DEPLOY_S16_AT_S32.get(depth)}  "
              f"from-scratch-S32={FROM_SCRATCH_S32.get(depth)}")
        xt, yt, xv, yv = _val_split(xtr, ytr, seed=0)
        for vname, kw in VARIANTS.items():
            if only and vname not in only:
                continue
            f = copy.deepcopy(flow)
            t0 = time.time()
            train(f, xt, yt, xv, yv, S, base, teacher, steps=STEPS, seed=0, **kw)
            acc = genuine_acc(f, xte, yte, S)
            dt = time.time() - t0
            reaches = acc >= ceiling - 0.01
            print(f"  {vname:>12}: genuine={acc:.4f}  gap_to_cont={cont - acc:+.4f}  "
                  f"reaches_ceiling={reaches}  wall={dt:.1f}s")
