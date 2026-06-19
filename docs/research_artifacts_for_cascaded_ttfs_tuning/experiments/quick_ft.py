"""PILLAR F1 — VERY QUICK fine-tune from a HEALTHY calibrated init.

Quantify how much a HEALTHY conversion init accelerates the greedy-adaptation FT
of the TRUE greedy fire-once cascaded single-spike TTFS deployment.

Pipeline (UNCHANGED greedy fire-once execution throughout — eval is pure genuine
`TTFSSegmentForward`, the deployed path):

  1. build(depth, S, seed) -> converted ANN flow (cold, scalar activation_scale).
  2. HEALTHY init = scale-aware boundaries + DFQ per-neuron bias correction
     (`match_activation_distributions`, the in-src validated calibration) which
     revives dead deep channels, re-orders decode, and matches the ANN rate
     WITHOUT saturating, + an OPTIONAL brief genuine warmup. The death cascade is
     dominated by premature-fire (a WEIGHT/sign problem) so calibration alone
     can leave cold accuracy near chance; the win is HEALTH (alive_frac,
     decode_corr, rate_match) and a far cheaper FT.
  3. CHANCE init = cold converted flow, no calibration.
  4. FT both with recipe_combo (genuine-cascade KD+CE, progressive unfreeze,
     per-channel theta co-train) under GREEDY execution, sweeping STEPS, and
     report steps + wall-seconds to LOSSLESS (gap to cont < 1pp).

HEALTH metric (per the pillar-1 diagnosis): per non-encoding layer
  HEALTH = (1 - rate_match_err) * alive_fraction * max(decode_corr, 0)  in [0,1]
  rate = decoded / activation_scale in [0,1]; rate_match_err = mean|rate_casc -
  rate_teacher| over ANN-firing channels; alive_fraction = 1 - frac(ANN-active
  channels that are dead, rate<=0.02); decode_corr = mean per-channel
  Pearson(decoded_casc, teacher_activation). Model HEALTH = mean over non-encoding
  layers. Healthy target ~1.0 at every depth.

Run: python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/quick_ft.py
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import torch

# This worktree's ``spikingjelly/`` submodule dir is empty; the genuine-cascade FT
# backward needs ``spikingjelly.activation_based``. Fall back to the main checkout.
if "spikingjelly.activation_based" not in sys.modules:
    for _sj in ("/home/yigit/repos/research_stuff/mimarsinan/spikingjelly",):
        if os.path.isdir(os.path.join(_sj, "spikingjelly", "activation_based")):
            sys.path.append(_sj)
            break

from recipe_harness import build, genuine_acc
from ft_budget import _calibrate_cuda, DEV
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.spiking.distribution_matching import match_activation_distributions
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.spiking.segment_partition import perceptron_of

import recipe_combo

DEAD_RATE = 0.02


def _per_layer_decoded(flow, x, S):
    """Per-perceptron genuine-cascade decoded values, shape (batch, channels),
    keyed by perceptron index (the deployed greedy fire-once path)."""
    rec: dict = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by_perc = {id(perceptron_of(n)): v for n, v in rec.items()
               if perceptron_of(n) is not None}
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by_perc.get(id(p))
        if v is not None:
            out[k] = v.reshape(-1, v.shape[-1]).double()
    return out


def _teacher_per_sample(base, flow, x):
    """Per-perceptron ANN activation outputs, shape (batch, channels), keyed by the
    flow perceptron index (so it aligns with `_per_layer_decoded`). Captured on the
    analytical (cycle-accurate OFF) flow so the activation == the ANN ReLU output."""
    means = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, out, k=k):
            means[k] = out.detach().reshape(-1, out.shape[-1]).double()
        handles.append(p.activation.register_forward_hook(hook))
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(True)
    return means


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b) / denom) if float(denom) > 1e-12 else 0.0


def health(flow, x, S, base):
    """Per-depth + model HEALTH of the greedy fire-once cascade vs the ANN teacher.
    Returns (model_health, per_layer_list, per_layer_alive_list)."""
    decoded = _per_layer_decoded(flow, x, S)
    teacher = _teacher_per_sample(base, flow, x)
    per_layer, alive_list = [], []
    perceptrons = flow.get_perceptrons()
    for k, p in enumerate(perceptrons):
        if getattr(p, "is_encoding_layer", False):
            continue
        dv = decoded.get(k)
        tv = teacher.get(k)
        if dv is None or tv is None:
            continue
        n = min(dv.shape[-1], tv.shape[-1])
        dv, tv = dv[:, :n], tv[:, :n]
        sc = p.activation_scale.detach().double()
        sc = sc if sc.dim() == 0 else sc[:n].clamp(min=1e-9)
        rate_c = (dv / sc).mean(0)
        # teacher rate = ANN activation / scale (same [0,1] normalization)
        rate_t = (tv / sc).mean(0)
        fires = rate_t > DEAD_RATE
        if not bool(fires.any()):
            per_layer.append(1.0); alive_list.append(1.0); continue
        rate_err = float((rate_c[fires] - rate_t[fires]).abs().mean())
        alive_frac = 1.0 - float((rate_c[fires] <= DEAD_RATE).double().mean())
        corrs = [_pearson(dv[:, j], tv[:, j]) for j in range(n) if bool(fires[j])]
        decode_corr = sum(corrs) / max(1, len(corrs))
        h = (1.0 - rate_err) * alive_frac * max(decode_corr, 0.0)
        per_layer.append(max(0.0, h))
        alive_list.append(alive_frac)
    model_h = sum(per_layer) / max(1, len(per_layer))
    return model_h, per_layer, alive_list


def make_teacher_flow(base, xtr):
    tflow = convert_torch_model(base, (64,), 10, device=str(DEV))
    _calibrate_cuda(tflow, xtr[:256])
    return tflow


def calibrate_healthy(flow, base, xtr, xcal, S, *, quantile=0.99, bias_iters=40,
                      eta=0.5):
    """HEALTHY init: scale-aware boundaries + DFQ per-neuron bias correction
    (revives dead deep channels to alive_frac~1.0 + matches ANN rate, no
    saturation). bias_iters=40/eta=0.5 maximised model HEALTH at d=9 S16. The
    residual decode/rate error is the premature-fire WEIGHT problem the quick FT
    adapts. In place."""
    tflow = make_teacher_flow(base, xtr)
    stats = match_activation_distributions(
        flow, tflow, xcal.double(), S,
        quantile=quantile, bias_iters=bias_iters, eta=eta)
    return flow, stats


def ft_to_lossless(make_init, xtr, ytr, xte, yte, S, base, teacher, cont, *,
                   step_grid=(50, 100, 200, 400, 800), seed=0, lossless_pp=0.01):
    """Each grid point is a SEPARATE FT run of that many steps from the same init
    (so the cosine-LR + progressive-unfreeze schedule spans the full budget). The
    schedule is budget-dependent, so we cannot share a single growing checkpoint.
    Returns (rows=[(steps, acc, seconds)], first_lossless=(steps, seconds, acc))."""
    rows = []
    first = None
    for steps in step_grid:
        flow = make_init()
        t0 = time.time()
        recipe_combo.train(flow, xtr, ytr, xte, yte, S, base, teacher,
                           steps=steps, seed=seed)
        dt = time.time() - t0
        acc = genuine_acc(flow, xte, yte, S)
        rows.append((steps, acc, dt))
        if first is None and (cont - acc) < lossless_pp:
            first = (steps, dt, acc)
    return rows, first


def run(depth, S, seed=0, step_grid=(50, 100, 200, 400, 800)):
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed)
    cold = genuine_acc(flow, xte, yte, S)
    xcal = xte[:512]

    cold_h, cold_pl, cold_alive = health(flow, xcal, S, base)

    healthy = copy.deepcopy(flow)
    t0 = time.time()
    healthy, stats = calibrate_healthy(healthy, base, xtr, xcal, S)
    cal_dt = time.time() - t0
    healthy_acc = genuine_acc(healthy, xte, yte, S)
    h_h, h_pl, h_alive = health(healthy, xcal, S, base)

    print(f"\n=== PILLAR F1  depth={depth} S={S} seed={seed} "
          f"cont={cont:.3f} ===")
    print(f"cold genuine={cold:.3f}  healthy-init genuine={healthy_acc:.3f}  "
          f"(calibration {cal_dt:.1f}s, dead {stats['dead_fraction_before']:.2f}"
          f"->{stats['dead_fraction_after']:.2f}, "
          f"gap {stats['mean_gap_before']:.3f}->{stats['mean_gap_after']:.3f})")
    print(f"model HEALTH  cold={cold_h:.3f}  healthy={h_h:.3f}  (chance HEALTH~0)")
    print("  per-depth HEALTH cold   : "
          + " ".join(f"{v:.2f}" for v in cold_pl))
    print("  per-depth HEALTH healthy: "
          + " ".join(f"{v:.2f}" for v in h_pl))
    print("  per-depth alive  cold   : "
          + " ".join(f"{v:.2f}" for v in cold_alive))
    print("  per-depth alive  healthy: "
          + " ".join(f"{v:.2f}" for v in h_alive))

    chance_rows, chance_first = ft_to_lossless(
        lambda: copy.deepcopy(flow), xtr, ytr, xte, yte, S, base, teacher, cont,
        step_grid=step_grid, seed=seed)
    healthy_rows, healthy_first = ft_to_lossless(
        lambda: copy.deepcopy(healthy), xtr, ytr, xte, yte, S, base, teacher, cont,
        step_grid=step_grid, seed=seed)

    print(f"\n{'steps':>6} | {'chance+FT':>10} {'sec':>6} | "
          f"{'healthy+FT':>11} {'sec':>6}   (cont {cont:.3f})")
    for (st, ca, cs), (_, ha, hs) in zip(chance_rows, healthy_rows):
        print(f"{st:>6} | {ca:>10.3f} {cs:>6.1f} | {ha:>11.3f} {hs:>6.1f}")

    def fmt(first):
        if first is None:
            return "NOT reached in budget"
        st, sec, acc = first
        return f"{st} steps / {sec:.1f}s (acc {acc:.3f})"

    print(f"\nlossless (<{1.0:.0f}pp gap) from CHANCE  : {fmt(chance_first)}")
    print(f"lossless (<{1.0:.0f}pp gap) from HEALTHY: {fmt(healthy_first)}"
          f"  (+ {cal_dt:.1f}s calibration)")
    return {
        "depth": depth, "S": S, "cont": cont, "cold": cold,
        "healthy_acc": healthy_acc, "cold_health": cold_h, "healthy_health": h_h,
        "cal_dt": cal_dt, "chance_first": chance_first,
        "healthy_first": healthy_first, "chance_rows": chance_rows,
        "healthy_rows": healthy_rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=9)
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=str, default="50,100,200,400,800")
    args = ap.parse_args()
    grid = tuple(int(s) for s in args.grid.split(","))
    run(args.depth, args.S, args.seed, grid)
