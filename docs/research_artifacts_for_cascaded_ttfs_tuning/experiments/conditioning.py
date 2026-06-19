"""PILLAR C2 -- NUMERICAL CONDITIONING of the greedy fire-once cascade.

Probe membrane/ramp/decoded-value scale by depth in the GENUINE single-spike
cascade and prototype a depth-invariant CONDITIONING that keeps the effective
drive (decoded rate) in a healthy band at every depth WITHOUT changing the greedy
fire-once rule (theta/scale/bias only -- all exactly foldable / parity-preserving).

The cold cascade decays/explodes by depth (death cascade): deep layers either go
dead (under-fire) or over-fire (premature greedy crossing on mixed-sign fan-in),
so the per-depth HEALTH collapses monotonically. Conditioning re-conditions the
band so HEALTH stays high at every depth, giving PILLAR-2 FT a well-conditioned
init (alive, ANN-rate-matched, ordered) instead of a chance-level death cascade.

Run: python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/conditioning.py
"""

from __future__ import annotations

import copy
import os
import sys
import time

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from recipe_harness import build, genuine_acc  # noqa: E402
from revive import _per_neuron, calibrate_revive  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales  # noqa: E402
from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases  # noqa: E402

DEAD_RATE = 0.02


def _teacher_rate(p, t):
    """ANN target firing-rate (decoded/scale, in [0,1]) for perceptron p."""
    sc = p.activation_scale.detach().double()
    tt = t.clamp(min=0).double()
    if sc.dim() > 0:
        n = min(sc.numel(), tt.numel())
        return (tt[:n] / sc[:n].clamp(min=1e-9)).clamp(0, 1)
    return (tt / float(sc)).clamp(0, 1)


def _decoded_records(flow, x, S):
    """Per-perceptron-INDEX decoded value tensors over the batch (full, not mean).

    A {perceptron_index: decoded[batch, channels]} map for the genuine cascade,
    suitable as the ``cascade_means_fn`` source for DFQ bias correction.
    """
    rec = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by_p = {}
    for node, val in rec.items():
        p = perceptron_of(node)
        if p is not None:
            by_p[id(p)] = val.reshape(-1, val.shape[-1]).double()
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by_p.get(id(p))
        if v is not None:
            out[k] = v
    return out


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / d)


def probe(flow, x, S, teacher):
    """Per-depth conditioning + health metrics on the genuine cascade.

    Returns a list of per-perceptron dicts (skips encoding layer) with:
      rate        cascade decoded/scale channel-mean (effective drive, [0,1])
      rate_t      ANN target rate
      decoded_mag mean |decoded| (membrane/ramp magnitude proxy)
      alive_frac  1 - fraction of ANN-active channels that are dead
      decode_corr per-channel Pearson(decoded_cascade mean, teacher channel-mean)
      rate_err    mean |rate - rate_t| over ANN-active channels
      health      (1-rate_err)*alive_frac*max(decode_corr,0)  in [0,1]
    plus model-level mean health.
    """
    perceptrons = list(flow.get_perceptrons())
    pn = _per_neuron(flow, x, S)
    decoded_full = _decoded_records(flow, x, S)
    rows = []
    for k, p in enumerate(perceptrons):
        if getattr(p, "is_encoding_layer", False):
            continue
        decoded, rate = pn[k]
        t = teacher.get(k)
        if rate is None or t is None:
            continue
        rate = rate.clamp(0, 1)
        rate_t = _teacher_rate(p, t)
        n = min(rate.numel(), rate_t.numel())
        rate, rate_t = rate[:n], rate_t[:n]
        active = rate_t > (DEAD_RATE)
        if not bool(active.any()):
            continue
        dead = (rate < DEAD_RATE) & active
        alive_frac = 1.0 - float(dead.sum()) / float(active.sum())
        rate_err = float((rate[active] - rate_t[active]).abs().mean())
        # decode_corr: per-channel correlation of decoded[batch] vs teacher mean
        # (teacher full samples unavailable here -> use channel-ordering corr of
        # decoded channel-mean vs teacher channel-mean across active channels).
        dec_mu = decoded_full[k].mean(0)[:n] if k in decoded_full else decoded[:n]
        tt = t.clamp(min=0).double()[:n]
        decode_corr = _pearson(dec_mu[active], tt[active]) if int(active.sum()) > 1 else 0.0
        health = (1.0 - min(rate_err, 1.0)) * alive_frac * max(decode_corr, 0.0)
        rows.append(dict(
            layer=k, rate=float(rate[active].mean()), rate_t=float(rate_t[active].mean()),
            decoded_mag=float(decoded.abs().mean()), alive_frac=alive_frac,
            decode_corr=decode_corr, rate_err=rate_err, health=health,
            over=float((rate[active] - rate_t[active]).clamp(min=0).mean()),
            under=float((rate_t[active] - rate[active]).clamp(min=0).mean()),
        ))
    model_health = sum(r["health"] for r in rows) / max(1, len(rows))
    return rows, model_health


def _ann_mean_dict(flow, teacher):
    """Teacher channel-mean keyed by perceptron index (non-encoding only)."""
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        if getattr(p, "is_encoding_layer", False):
            continue
        t = teacher.get(k)
        if t is not None:
            out[k] = t.clamp(min=0).double()
    return out


def condition_cascade(flow, x, S, teacher, *, revive_iters=40, bias_iters=8,
                      eta=0.5, per_channel=True):
    """Depth-invariant conditioning (theta/scale/bias only, foldable).

    Two stages, both parity-preserving:
      1. REVIVE: lower per-channel theta for dead-but-should-fire neurons so every
         layer's alive_frac -> ~1 (kills the under-fire half of the death cascade).
      2. DFQ BIAS: match each perceptron's decoded channel-mean to the ANN's by
         nudging layer.bias. This shifts the membrane trajectory so the greedy
         crossing lands at the ANN level -- it pulls the premature OVER-fire band
         (decoded mean running high at depth) back into the healthy band, keeping
         the effective drive depth-invariant.
    Re-run iteratively so downstream re-conditions as upstream is fixed.
    """
    calibrate_revive(flow, x, S, teacher, iters=revive_iters,
                     dead_rate=DEAD_RATE * 2.5, step=0.6, per_channel=per_channel)
    ann_mean = _ann_mean_dict(flow, teacher)

    def cascade_means_fn():
        return _decoded_records(flow, x, S)

    stats = dfq_correct_biases(flow, ann_mean, cascade_means_fn,
                               bias_iters=bias_iters, eta=eta)
    propagate_boundary_input_scales(flow)
    # second revive pass: bias shifts may re-kill / re-awaken channels
    calibrate_revive(flow, x, S, teacher, iters=revive_iters // 2,
                     dead_rate=DEAD_RATE * 2.5, step=0.7, per_channel=per_channel)
    propagate_boundary_input_scales(flow)
    return flow, stats


def _fmt_layers(rows, key):
    return "[" + ", ".join(f"{r[key]:.2f}" for r in rows) + "]"


def ft_speed(flow, xtr, ytr, xte, yte, S, base, teacher, cont, *,
             checkpoints=(0, 25, 50, 100, 200, 400, 800), seed=0,
             lossless_tol=0.005, tag=""):
    """FT (staircase-STE) the given init; report genuine acc + wall at each
    cumulative step checkpoint and the (steps, seconds) to reach cont-lossless."""
    import copy

    from recipe_staircase_ste import train

    f = copy.deepcopy(flow)
    prev = 0
    hit_steps = hit_secs = None
    elapsed = 0.0
    print(f"  [{tag}] FT steps -> genuine acc (cont={cont:.3f})")
    for ck in checkpoints:
        if ck > prev:
            t0 = time.time()
            train(f, xtr, ytr, xte, yte, S, base, teacher, steps=ck - prev, seed=seed)
            elapsed += time.time() - t0
            prev = ck
        acc = genuine_acc(f, xte, yte, S)
        flag = ""
        if hit_steps is None and acc >= cont - lossless_tol:
            hit_steps, hit_secs = ck, elapsed
            flag = "  <-- lossless"
        print(f"      {ck:>4} steps  {acc:.3f}  ({elapsed:.0f}s){flag}")
    return hit_steps, hit_secs


def run_grid(depths=(6, 9, 12), Ss=(8, 16, 32), seed=0):
    print("=== PILLAR C2: numerical conditioning of greedy fire-once cascade ===\n")
    for depth in depths:
        for S in Ss:
            t0 = time.time()
            flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
            cold_acc = genuine_acc(flow, xte, yte, S)
            rows0, h0 = probe(flow, xte, S, teacher)
            cflow = copy.deepcopy(flow)
            cflow, stats = condition_cascade(cflow, xte, S, teacher)
            cond_acc = genuine_acc(cflow, xte, yte, S)
            rows1, h1 = probe(cflow, xte, S, teacher)
            dt = time.time() - t0
            print(f"--- depth={depth} S={S}  continuous={cont:.3f}  ({dt:.0f}s) ---")
            print(f"  genuine acc      cold={cold_acc:.3f}  conditioned={cond_acc:.3f}")
            print(f"  model HEALTH     cold={h0:.3f}  conditioned={h1:.3f}")
            print(f"  per-depth health cold {_fmt_layers(rows0,'health')}")
            print(f"                   cond {_fmt_layers(rows1,'health')}")
            print(f"  per-depth alive  cold {_fmt_layers(rows0,'alive_frac')}")
            print(f"                   cond {_fmt_layers(rows1,'alive_frac')}")
            print(f"  per-depth corr   cold {_fmt_layers(rows0,'decode_corr')}")
            print(f"                   cond {_fmt_layers(rows1,'decode_corr')}")
            print(f"  per-depth rate   cold {_fmt_layers(rows0,'rate')} (tgt "
                  f"{_fmt_layers(rows0,'rate_t')})")
            print(f"                   cond {_fmt_layers(rows1,'rate')}")
            print(f"  decoded mag      cold {_fmt_layers(rows0,'decoded_mag')}")
            print(f"                   cond {_fmt_layers(rows1,'decoded_mag')}")
            print(f"  DFQ mean-gap     {stats['mean_gap_before']:.4f} -> "
                  f"{stats['mean_gap_after']:.4f}\n")


if __name__ == "__main__":
    run_grid()
