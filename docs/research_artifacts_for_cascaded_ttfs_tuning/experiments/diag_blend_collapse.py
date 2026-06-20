"""Diagnose D2: WHY does the synchronized-to-cascaded OUTPUT blend collapse at depth?

recipe_blend_curriculum.py trains on ``(1-a)*staircase + a*genuine`` with ``a:0->1``
and collapses to ~0.08 deployed at d=9. This script, at d=9 S=16:

  - tracks deployed genuine_acc AND staircase_acc as ``a`` ramps 0->1 (find the
    collapse point);
  - tracks, at each checkpoint, per-layer decoded-value distributions for BOTH the
    genuine cascade branch and the staircase branch (mean activity + dead fraction
    per depth) so we can SEE whether the genuine branch is dead until a is high and
    then suffers a sudden destructive handoff, vs the KD-to-continuous fighting the
    genuine firing;
  - measures the loss-gradient contribution of each branch over the ramp (is the
    genuine branch ever actually trained, or does the staircase dominate the
    blended logit so the genuine path gets ~0 effective gradient?);
  - tests the proposed fix: train weights on the LOSSLESS staircase forward to
    convergence (staircase-init), THEN short genuine FT, and reports whether it
    avoids the collapse.

Run: source env/bin/activate && python docs/.../experiments/diag_blend_collapse.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

from recipe_harness import (  # noqa: E402
    batches, build, genuine_acc, genuine_logits, kd_ce_loss, staircase_logits,
    teacher_logits, trainable_params,
)
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402


# ----------------------------------------------------------------------------- #
# Per-layer decoded-value capture (depth-ordered)                               #
# ----------------------------------------------------------------------------- #
def _perc_nodes_in_depth_order(flow, S):
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    return drv, [n for n in drv._driver._exec if perceptron_of(n) is not None]


def genuine_layer_values(flow, x, S):
    """Genuine cascade decoded value per perceptron (scale units), depth-ordered.

    Uses the driver's node_value_recorder: each entry is ``(accum/T)*scale`` — the
    exact deployed decode. Keyed by mapper node; ordered by exec position == depth.
    """
    drv, perc_nodes = _perc_nodes_in_depth_order(flow, S)
    _, rec = drv.forward_with_node_values(x.double())
    return [rec[n].detach() for n in perc_nodes if n in rec]


def staircase_layer_values(flow, x):
    """Staircase (complete-sum, cycle_accurate=False) decoded value per activation,
    depth-ordered. Forward-hook capture on the TTFSActivation modules."""
    acts = [m for m in flow.modules() if isinstance(m, TTFSActivation)]
    caps: dict = {}
    handles = [
        a.register_forward_hook(
            lambda _m, _i, o, i=i: caps.setdefault(i, o.detach()))
        for i, a in enumerate(acts)
    ]
    try:
        staircase_logits(flow, x)
    finally:
        for h in handles:
            h.remove()
    return [caps[i] for i in sorted(caps)]


def _layer_stats(vals):
    """(mean activity, dead-fraction) per layer."""
    return [(float(v.mean()), float((v.abs() <= 1e-9).double().mean())) for v in vals]


def _fmt_stats(stats):
    return "  ".join(f"d{i}:{m:.2f}/{d:.0%}" for i, (m, d) in enumerate(stats))


# ----------------------------------------------------------------------------- #
# Branch-gradient probe: how much does each branch drive the blended loss?       #
# ----------------------------------------------------------------------------- #
def branch_grad_norms(flow, x, y, S, a, base):
    """At blend coefficient ``a``, grad-norm on the trainable params from the
    STAIRCASE branch alone vs the GENUINE branch alone (each scaled by its blend
    weight, as the optimiser sees it). Reveals which branch actually trains."""
    params = trainable_params(flow)
    teach = teacher_logits(base, x)

    def gnorm(logits, weight):
        for p in params:
            if p.grad is not None:
                p.grad = None
        loss = weight * kd_ce_loss(logits, y, teach, alpha=0.3)
        if float(weight) == 0.0:
            return 0.0
        g = torch.autograd.grad(loss, params, retain_graph=False,
                                allow_unused=True)
        return float(torch.sqrt(sum((gi.pow(2).sum() for gi in g if gi is not None))))

    stair = (1 - a) * 0 + staircase_logits(flow, x)  # forward in staircase mode
    gn_stair = gnorm(stair, 1 - a)
    gen = genuine_logits(flow, x, S)
    gn_gen = gnorm(gen, a)
    return gn_stair, gn_gen


# ----------------------------------------------------------------------------- #
# Diagnostic 1: track the ramp                                                   #
# ----------------------------------------------------------------------------- #
def diag_ramp(flow, xtr, ytr, xte, yte, S, base, teacher, *, steps, seed,
              lr=2e-3, bs=256, ramp_frac=0.7, probe_every=None):
    """Reproduce recipe_blend_curriculum's ramp, probing at a grid of ``a``."""
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ramp_end = max(1, int(ramp_frac * steps))
    probe_every = probe_every or max(1, steps // 12)

    xt, yt = xte[:512], yte[:512]
    rows = []
    print(f"\n=== RAMP probe (d9 S{S}, steps={steps}, ramp_frac={ramp_frac}) ===")
    print(f"{'step':>5} {'a':>5} {'gen_acc':>8} {'stair_acc':>10} "
          f"{'g_stair':>8} {'g_gen':>8}")
    for i, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        a = min(1.0, i / ramp_end)
        if i % probe_every == 0 or i == steps - 1:
            with torch.no_grad():
                gacc = genuine_acc(flow, xt, yt, S)
                sl = staircase_logits(flow, xt)
                sacc = float((sl.argmax(-1) == yt).double().mean())
            gn_s, gn_g = branch_grad_norms(flow, x, y, S, a, base)
            gstats = _layer_stats(genuine_layer_values(flow, xt[:128], S))
            sstats = _layer_stats(staircase_layer_values(flow, xt[:128]))
            rows.append(dict(step=i, a=a, gen_acc=gacc, stair_acc=sacc,
                             g_stair=gn_s, g_gen=gn_g, gstats=gstats, sstats=sstats))
            print(f"{i:>5} {a:>5.2f} {gacc:>8.3f} {sacc:>10.3f} "
                  f"{gn_s:>8.2f} {gn_g:>8.2f}")
        blend = (1 - a) * staircase_logits(flow, x) + a * genuine_logits(flow, x, S)
        loss = kd_ce_loss(blend, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    with torch.no_grad():
        gacc = genuine_acc(flow, xt, yt, S)
    rows.append(dict(step=steps, a=1.0, gen_acc=gacc, stair_acc=None,
                     g_stair=None, g_gen=None,
                     gstats=_layer_stats(genuine_layer_values(flow, xt[:128], S)),
                     sstats=None))
    print(f"{steps:>5} {1.0:>5.2f} {gacc:>8.3f}  (final deployed)")
    return rows


def report_collapse(rows, cont):
    print("\n--- COLLAPSE LOCALISATION ---")
    prev = None
    collapse_a = None
    for r in rows:
        if r["gen_acc"] is None:
            continue
        if prev is not None and prev - r["gen_acc"] > 0.15 and collapse_a is None:
            collapse_a = r["a"]
            print(f"  *** genuine_acc DROP {prev:.3f}->{r['gen_acc']:.3f} "
                  f"at a={r['a']:.2f} (step {r['step']}) ***")
        prev = r["gen_acc"]
    print("\n--- per-layer decoded distributions (mean/dead%) over the ramp ---")
    for r in rows:
        if r.get("gstats"):
            print(f"a={r['a']:.2f} GEN   {_fmt_stats(r['gstats'])}")
        if r.get("sstats"):
            print(f"a={r['a']:.2f} STAIR {_fmt_stats(r['sstats'])}")
    return collapse_a


# ----------------------------------------------------------------------------- #
# Diagnostic 2: proposed fix — staircase-init then short genuine FT             #
# ----------------------------------------------------------------------------- #
def staircase_then_genuine(flow, xtr, ytr, xte, yte, S, base, teacher, *,
                           stair_steps, gen_steps, seed, lr=2e-3, bs=256):
    """Train weights on the LOSSLESS staircase forward to convergence, THEN short
    genuine FT. Does it avoid the collapse?"""
    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=stair_steps)
    xt, yt = xte[:512], yte[:512]
    print(f"\n=== STAIRCASE-INIT ({stair_steps} stair) then GENUINE-FT ({gen_steps}) ===")
    for x, y in batches(xtr, ytr, bs, stair_steps, seed):
        logits = staircase_logits(flow, x)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        sl = staircase_logits(flow, xt)
        sacc = float((sl.argmax(-1) == yt).double().mean())
    gacc0 = genuine_acc(flow, xt, yt, S)
    print(f"  after staircase init: staircase_acc={sacc:.3f}  cold-genuine_acc={gacc0:.3f}")
    print(f"  genuine layer stats: {_fmt_stats(_layer_stats(genuine_layer_values(flow, xt[:128], S)))}")

    opt = torch.optim.Adam(trainable_params(flow), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, gen_steps))
    for j, (x, y) in enumerate(batches(xtr, ytr, bs, gen_steps, seed + 1)):
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(flow), 1.0)
        opt.step(); sched.step()
    gacc = genuine_acc(flow, xt, yt, S)
    print(f"  after {gen_steps} genuine FT: genuine_acc={gacc:.3f}")
    return sacc, gacc0, gacc


# ----------------------------------------------------------------------------- #
def main():
    DEPTH, S = 9, 16
    SEED = 0
    t0 = time.time()
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(DEPTH, S, seed=SEED)
    print(f"built d={DEPTH} S={S} in {time.time()-t0:.1f}s  cont(LOSSLESS target)={cont:.4f}")

    import copy
    flow_ramp = copy.deepcopy(flow)
    t1 = time.time()
    rows = diag_ramp(flow_ramp, xtr, ytr, xte, yte, S, base, teacher,
                     steps=600, seed=SEED, ramp_frac=0.7)
    print(f"[ramp diag wall {time.time()-t1:.1f}s]")
    collapse_a = report_collapse(rows, cont)

    flow_fix = copy.deepcopy(flow)
    t2 = time.time()
    sacc, gacc0, gacc = staircase_then_genuine(
        flow_fix, xtr, ytr, xte, yte, S, base, teacher,
        stair_steps=400, gen_steps=200, seed=SEED)
    fix_wall = time.time() - t2

    print("\n========================= SUMMARY =========================")
    print(f"d={DEPTH} S={S}  cont(LOSSLESS)={cont:.4f}  combo-baseline(d9s32)=0.911")
    final_blend = next(r["gen_acc"] for r in reversed(rows) if r["gen_acc"] is not None)
    print(f"BLEND ramp final deployed genuine_acc = {final_blend:.3f}  "
          f"(collapse at a={collapse_a})")
    print(f"STAIRCASE-init -> genuine-FT deployed  = {gacc:.3f}  "
          f"(staircase_acc={sacc:.3f}, cold-genuine after init={gacc0:.3f}) "
          f"[wall {fix_wall:.1f}s]")
    print(f"gap to cont: blend={cont-final_blend:+.3f}  stair-then-gen={cont-gacc:+.3f}")


if __name__ == "__main__":
    main()
