"""Is the cascaded-TTFS collapse a CORRECTABLE per-layer GAIN, or fundamental
causal-firing (partial-sum) timing noise?

The cascaded single-spike core fires on a GREEDY threshold crossing of its
RUNNING partial sum (ramp), while the staircase/LIF use the COMPLETE weighted
sum. With mixed-sign ReLU weights, arrival order corrupts the crossing. But the
per-layer gen-vs-stair correlation is high (~0.81 at L1), suggesting a mostly
MONOTONIC (gain) distortion. This decides it:

  (1) GAIN per layer: median(gen/stair) over firing neurons -- a single number
      per layer. If genuine is stair*g_L, the collapse is a compounding gain.
  (2) RESIDUAL after the per-neuron oracle gain: corr and how much non-gain
      scatter remains (the irreducible timing noise).
  (3) NON-NEGATIVE control: redo with |W| interior weights (no cancellation).
      If the collapse vanishes, mixed-sign partial-sum cancellation is the cause.
  (4) ORACLE per-layer theta-scale ceiling: best cascaded accuracy a static
      per-layer threshold trim can buy (the deployable gain-correction ceiling).

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/mechanism.py
"""

from __future__ import annotations

import itertools
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from cascade_fixtures import _SingleSegmentMLP, _calibrate_scales, install_ttfs_nodes  # noqa: E402
from cascade_lab import digits_task, train_continuous, _accuracy  # noqa: E402
from decode_curve import _genuine_per_layer, _staircase_per_layer  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


def _build(depth, S, seed=0, abs_weights=False):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, 64, 64, 10)
    train_continuous(base, xtr, ytr, epochs=120)
    if abs_weights:
        with torch.no_grad():
            lins = [m for m in base.modules() if isinstance(m, torch.nn.Linear)]
            for m in lins[:-1]:           # interior layers only; keep classifier signed
                m.weight.abs_()
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (64,), 10, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    flow.double()
    return flow, xte, yte, cont


def _genuine_acc(flow, x, y, S):
    with torch.no_grad():
        return _accuracy(TTFSSegmentForward(flow.get_mapper_repr(), S)(x.double()), y)


def per_layer_gain(depth=3, S=8, seed=0, abs_weights=False):
    flow, xte, yte, cont = _build(depth, S, seed, abs_weights)
    gen = _genuine_per_layer(flow, xte, S)
    stair = _staircase_per_layer(flow, xte)
    stair_acc = None
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    with torch.no_grad():
        stair_acc = _accuracy(flow.double()(xte.double()), yte)
    gen_acc = _genuine_acc(flow, xte, yte, S)
    tag = "ABS" if abs_weights else "signed"
    print(f"\n--- per-layer GAIN d={depth} S={S} [{tag}]  "
          f"cont={cont:.3f} stair={stair_acc:.3f} genuine={gen_acc:.3f} ---")
    print(f"{'layer':>5} {'gen_mean':>9} {'stair_mean':>11} {'median(g/s)':>12} "
          f"{'corr':>6} {'resid_after_gain':>17}")
    for k in sorted(stair):
        g = gen.get(k)
        s = stair[k]
        if g is None:
            continue
        n = min(g.numel(), s.numel())
        gf, sf = g.reshape(-1)[:n], s.reshape(-1)[:n]
        fire = sf.abs() > 1e-9
        if fire.sum() < 2:
            print(f"{k:>5}  (too few firing)")
            continue
        ratio = (gf[fire] / sf[fire])
        med = float(ratio.median())
        gc, sc = gf - gf.mean(), sf - sf.mean()
        corr = float((gc @ sc) / (gc.norm() * sc.norm() + 1e-12))
        # residual std after applying the single per-layer gain median to stair
        resid = (gf - med * sf)
        resid_rel = float(resid[fire].std() / (gf[fire].std() + 1e-12))
        print(f"{k:>5} {float(gf.mean()):>9.4f} {float(sf.mean()):>11.4f} "
              f"{med:>12.4f} {corr:>6.3f} {resid_rel:>17.3f}")


def oracle_theta_ceiling(depth=3, S=8, seed=0, grid=(1.0, 0.7, 0.5, 0.35, 0.25, 0.15)):
    flow, xte, yte, cont = _build(depth, S, seed)
    percs = flow.get_perceptrons()
    base_scales = [p.activation_scale.clone() for p in percs]

    def acc_for(gammas):
        for k, p in enumerate(percs):
            g = gammas[k] if k < len(gammas) else 1.0
            p.set_activation_scale(base_scales[k] * g)
        install_ttfs_nodes(flow, S)
        return _genuine_acc(flow, xte, yte, S)

    n = len(percs)
    base_acc = acc_for([1.0] * n)
    best = ([1.0] * n, base_acc)
    for gtup in itertools.product(grid, repeat=n):
        a = acc_for(list(gtup))
        if a > best[1]:
            best = (list(gtup), a)
    print(f"d={depth} S={S} seed={seed}: cont={cont:.3f} baseline_genuine={base_acc:.3f} "
          f"-> oracle_theta={best[1]:.3f}  gamma={best[0]}")


if __name__ == "__main__":
    print("=" * 78)
    print("(1)(2) per-layer GAIN + residual: is the collapse correctable gain?")
    print("=" * 78)
    per_layer_gain(depth=3, S=8)
    per_layer_gain(depth=6, S=8)

    print("\n" + "=" * 78)
    print("(3) NON-NEGATIVE interior-weight control (no cancellation)")
    print("=" * 78)
    per_layer_gain(depth=3, S=8, abs_weights=True)
    per_layer_gain(depth=6, S=8, abs_weights=True)

    print("\n" + "=" * 78)
    print("(4) ORACLE per-layer theta-scale ceiling (deployable static gain)")
    print("=" * 78)
    for d in (3, 4):
        oracle_theta_ceiling(depth=d, S=8)
