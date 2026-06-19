"""Localize the genuine-cascade collapse (genuine=chance cold even at d=3).

Two probes on the IDENTICAL converted flow:

(A) DECODE CURVE: a 2-layer identity chain (W=1,b=0,scale=1). Sweep the input
    value v in [0,1]; record the genuine single-spike cascade's decoded output of
    the consumer layer vs the analytical staircase decode. This reveals whether
    the deployed ramp decode is LINEAR (T-tau)/T (== staircase) or QUADRATIC
    (T-tau)^2 (the capacity.py claim), per neuron, with NO weight effect.

(B) PER-LAYER COLLAPSE: on the trained d=3 flow, capture the per-perceptron mean
    decoded value for the genuine cascade vs the analytical staircase. Find the
    layer where genuine diverges from staircase -- the collapse point.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/decode_curve.py
"""

from __future__ import annotations

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
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


def _genuine_per_layer(flow, x, S):
    rec: dict = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by = {id(perceptron_of(n)): v for n, v in rec.items() if perceptron_of(n) is not None}
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by.get(id(p))
        if v is not None:
            out[k] = v.reshape(-1, v.shape[-1]).double()
    return out


def _staircase_per_layer(flow, x):
    """Capture each perceptron's analytical-staircase output value."""
    caps: dict = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, out, k=k):
            caps[k] = out.detach().double().reshape(-1, out.shape[-1])
        handles.append(p.register_forward_hook(hook))
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return caps


def probe_decode_curve(S=8):
    """2-layer identity: input v -> layer0 emits a single spike at tau(v) -> layer1
    (consumer) reconstructs. Compare genuine decoded(layer0 output) to v itself."""
    torch.manual_seed(0)
    base = _SingleSegmentMLP(2, 1, 1, 1)
    for m in base.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
    flow = convert_torch_model(base, (1,), 1, device="cpu")
    _calibrate_scales(flow, torch.linspace(0, 1, 17, dtype=torch.float64).reshape(-1, 1))
    install_ttfs_nodes(flow, S)
    flow.double()
    vs = torch.linspace(0, 1, 17, dtype=torch.float64).reshape(-1, 1)
    gen = _genuine_per_layer(flow, vs, S)
    print(f"--- (A) DECODE CURVE, identity 2-layer, S={S} ---")
    print(f"{'v_in':>6} {'L0_decoded':>11} {'L1_decoded':>11} {'linear=v':>9} {'quad=v^2':>9}")
    for i, v in enumerate(vs.reshape(-1).tolist()):
        l0 = float(gen[0].reshape(-1)[i]) if 0 in gen else float("nan")
        l1 = float(gen[1].reshape(-1)[i]) if 1 in gen else float("nan")
        print(f"{v:>6.3f} {l0:>11.4f} {l1:>11.4f} {v:>9.3f} {v*v:>9.3f}")


def probe_layer_collapse(depth=3, S=8):
    torch.manual_seed(0)
    xtr, ytr, xte, yte = digits_task(seed=1)
    base = _SingleSegmentMLP(depth, 64, 64, 10)
    train_continuous(base, xtr, ytr, epochs=120)
    flow = convert_torch_model(base, (64,), 10, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    flow.double()
    gen = _genuine_per_layer(flow, xte, S)
    stair = _staircase_per_layer(flow, xte)
    print(f"\n--- (B) PER-LAYER COLLAPSE, trained d={depth} S={S} ---")
    print(f"{'layer':>5} {'gen_mean':>9} {'stair_mean':>11} {'gen_nonzero%':>13} "
          f"{'stair_nonzero%':>15} {'corr(gen,stair)':>16}")
    for k in sorted(stair):
        g = gen.get(k)
        s = stair[k]
        if g is None:
            print(f"{k:>5}  (no genuine record)")
            continue
        n = min(g.numel(), s.numel())
        gf, sf = g.reshape(-1)[:n], s.reshape(-1)[:n]
        nz_g = float((gf.abs() > 1e-9).double().mean()) * 100
        nz_s = float((sf.abs() > 1e-9).double().mean()) * 100
        gc, sc = gf - gf.mean(), sf - sf.mean()
        corr = float((gc @ sc) / (gc.norm() * sc.norm() + 1e-12))
        print(f"{k:>5} {float(gf.mean()):>9.4f} {float(sf.mean()):>11.4f} "
              f"{nz_g:>12.1f}% {nz_s:>14.1f}% {corr:>16.4f}")


if __name__ == "__main__":
    probe_decode_curve(S=8)
    probe_layer_collapse(depth=3, S=8)
