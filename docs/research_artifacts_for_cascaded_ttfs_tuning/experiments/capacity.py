"""DIRECTION F (H6) — information-capacity / optimal-code analysis of the
single-spike TTFS timing code vs the multi-spike LIF rate code.

This bounds the whole cascaded-TTFS effort. It answers, with numbers:

  (1) bits/neuron of each code at resolution S and at cascade depth d (accounting
      for the deployed latency windows and the integer spike times);
  (2) is LIF-level accuracy reachable IN PRINCIPLE by single-spike timing at
      practical S, or is there a hard representational ceiling?
  (3) what is the theoretically optimal single-spike encode/decode under the
      deployed constraints, and how far is the current ``round()``-TTFS+ramp from
      it?

Everything runs on the isolated ``cascade_lab`` harness (CPU, float64, seconds).
We do NOT edit ``cascade_lab.py``; we import it. Run directly:

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/capacity.py

Findings write-up: ``../15_capacity.md``.
"""

from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ART = os.path.dirname(_HERE)                 # research_artifacts_for_cascaded_ttfs_tuning
_REPO = os.path.abspath(os.path.join(_ART, "..", ".."))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "tests"), _ART, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_fixtures import (  # noqa: E402
    _SingleSegmentMLP,
    _calibrate_scales,
    cascade_forward,
    install_ttfs_nodes,
)
from cascade_lab import _accuracy, digits_task, train_continuous  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import (  # noqa: E402
    TTFSSegmentForward,
)
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402

PRIMARY = dict(depth=3, width=64, in_dim=64, n_classes=10)


# ---------------------------------------------------------------------------
# (1) Closed-form per-neuron capacity (counting + uniform-quantizer bound)
# ---------------------------------------------------------------------------
def per_neuron_levels_and_bits():
    """Distinguishable codewords / bits per neuron for both codes at window T.

    Single-spike TIMING: spike time tau in {0,...,T-1} plus the no-spike state
    (tau=T, decodes 0) => T+1 distinguishable levels.
    Multi-spike RATE (LIF): spike count in {0,...,T} => T+1 distinguishable levels.
    They are IDENTICAL per neuron at the same window. bits/neuron = log2(T+1).
    """
    rows = []
    for T in (4, 8, 16, 32):
        levels = T + 1
        rows.append({"T": T, "levels_timing": levels, "levels_rate": levels,
                     "bits_per_neuron": round(math.log2(levels), 4)})
    return rows


def single_layer_quantization_error():
    """Single-layer encode->decode error of round()-TTFS over uniform v in [0,1].

    tau = round(T(1-v)); ideal linear decode v_hat = (T-tau)/T. This is a uniform
    mid-tread quantizer: unbiased, std ~ (1/T)/sqrt(12). Identical to the rate code's
    count=round(Tv) quantizer. Confirms the per-layer codes are equivalent.
    """
    rows = []
    v = np.linspace(0.0, 1.0, 100001)
    for T in (4, 8, 16, 32):
        tau = np.round(T * (1.0 - v))
        vhat = (T - tau) / T
        err = vhat - v
        rows.append({"T": T, "mean_err": round(float(err.mean()), 6),
                     "std_err": round(float(err.std()), 6),
                     "max_abs_err": round(float(np.abs(err).max()), 5),
                     "uniform_bound_1over2T": round(0.5 / T, 5)})
    return rows


def ramp_effective_weight_distortion():
    """The deployed ramp decode's effective weight per upstream spike vs the ideal.

    A single upstream spike at local cycle tau (weight 1) drives the consumer's
    ramp_current=1 for all c>=tau, so membrane(T-1) = sum_{c=tau}^{T-1}(c-tau+1) =
    triangular ~ (T-tau)(T-tau+1)/2 ~ (T-tau)^2/2. The continuous teacher wants the
    upstream value (T-tau)/T (LINEAR in remaining window). The ramp therefore applies
    a QUADRATIC-in-(remaining-window) emphasis: R(tau) ~ (T-tau)^2 vs ideal L(tau) ~
    (T-tau). The ratio R/L collapses for late spikes (small upstream values) -- THIS
    is the cascade's systematic distortion that compounds into the death cascade.
    """
    rows = []
    for T in (8, 16):
        R = np.array([sum(range(1, (T - 1 - tau) + 2)) if (T - 1 - tau) >= 0 else 0
                      for tau in range(T)], dtype=float)
        Rn = R / R.max()
        per_tau = []
        for tau in range(T):
            L = (T - tau) / T
            per_tau.append({"tau": tau, "L_linear": round(L, 4),
                            "R_ramp_norm": round(float(Rn[tau]), 4),
                            "R_over_L": round(float(Rn[tau] / L) if L > 0 else 0.0, 3)})
        rows.append({"T": T, "per_tau": per_tau})
    return rows


# ---------------------------------------------------------------------------
# (1b) Empirical: do per-neuron levels survive cascade depth?
# ---------------------------------------------------------------------------
def _identity_chain(depth, S, seed=0):
    """A pure pass-through chain (W=1, b=0, scale=1) to isolate the code transfer
    per depth from any weight effect."""
    torch.manual_seed(seed)
    base = _SingleSegmentMLP(depth, 1, 1, 1)
    for m in base.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.constant_(m.weight, 1.0)
            torch.nn.init.constant_(m.bias, 0.0)
    flow = convert_torch_model(base, (1,), 1, device="cpu")
    _calibrate_scales(flow, torch.linspace(0, 1, 17, dtype=torch.float64).reshape(-1, 1))
    install_ttfs_nodes(flow, S)
    return flow.double()


def _decoded_per_depth(flow, x, S):
    rec: dict = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by = {id(perceptron_of(n)): v for n, v in rec.items() if perceptron_of(n) is not None}
    out: dict = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by.get(id(p))
        if v is not None:
            out[k] = v.reshape(-1, v.shape[-1]).double()
    return out


def identity_chain_levels(depth=4, S=8, seed=0):
    """With weights=identity, the timing code is LOSSLESS through depth: every
    depth carries the full T+1 levels and decodes the quantized input exactly. So
    the death cascade is NOT a per-neuron capacity loss with depth."""
    flow = _identity_chain(depth, S, seed)
    xs = torch.linspace(0, 1, 17, dtype=torch.float64).reshape(-1, 1)
    res = _decoded_per_depth(flow, xs, S)
    rows = []
    for d in range(depth):
        if d not in res:
            continue
        vals = res[d].reshape(-1)
        levels = sorted({round(float(v), 4) for v in vals})
        rows.append({"depth": d, "n_levels": len(levels),
                     "max_abs_dev_from_input_quant": round(
                         float((vals - torch.round(xs.reshape(-1) * S) / S).abs().max()), 5)})
    return rows


# ---------------------------------------------------------------------------
# (3) How far is genuine round()-TTFS+ramp from the OPTIMAL single-spike decode?
# ---------------------------------------------------------------------------
def _staircase_acc(flow, x, y):
    """Per-layer analytical TTFS staircase (cycle_accurate OFF): the best a
    single-spike timing code with a LINEAR (T-tau)/T decode can do -- no cascade
    ramp distortion. The optimal-code upper bound for single-spike timing."""
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        return _accuracy(flow(x.double()), y)


def optimal_vs_genuine(depth, width, in_dim, n_classes, S, seed):
    """Train continuous, then report continuous / ideal-staircase / genuine accuracy.

    ideal-staircase == optimal single-spike timing (linear decode, T+1 levels/layer).
    genuine == deployed cascade (quadratic ramp + latency windows). The gap
    staircase-genuine is the cost of the SUBOPTIMAL deployed decode (recoverable in
    principle); the gap continuous-staircase is the irreducible T+1-level quantization
    cost (the true capacity floor of a single-spike timing code at this S)."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    train_continuous(base, xtr, ytr, epochs=120)
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    stair = _staircase_acc(flow, xte, yte)
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    gen = _accuracy(cascade_forward(flow, xte, S), yte)
    return {"cont": float(cont), "ideal_staircase": float(stair), "genuine": float(gen)}


# ---------------------------------------------------------------------------
# (2) Is the gap a capacity ceiling or a correctable distortion?
#     Oracle per-depth theta-scale (a DEPLOYABLE per-layer correction).
# ---------------------------------------------------------------------------
def oracle_theta_scale(depth, width, in_dim, n_classes, S, seed, grid=(1.0, 0.7, 0.5, 0.35, 0.25)):
    """Search a per-depth threshold scale gamma_d (smaller theta -> earlier fire ->
    higher decode) that maximizes genuine accuracy. This is the DEPLOYABLE category:
    a per-layer scale/threshold trim, NOT a decode change. If a static per-depth
    gamma lifts genuine toward the ideal staircase, the death cascade is a correctable
    GAIN distortion, not a capacity ceiling."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    train_continuous(base, xtr, ytr, epochs=120)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    percs = flow.get_perceptrons()
    base_scales = [p.activation_scale.clone() for p in percs]

    def acc_for(gammas):
        for k, p in enumerate(percs):
            g = gammas[k] if k < len(gammas) else 1.0
            p.set_activation_scale(base_scales[k] * g)
        install_ttfs_nodes(flow, S)
        return _accuracy(cascade_forward(flow, xte, S), yte)

    base_acc = float(acc_for([1.0] * depth))
    best = ([1.0] * depth, base_acc)
    for g in itertools.product(grid, repeat=depth):
        a = float(acc_for(list(g)))
        if a > best[1]:
            best = (list(g), a)
    return {"baseline": base_acc, "best_gamma": best[0], "best_acc": best[1]}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _mean_over_seeds(fn, seeds=(0, 1, 2), **kw):
    rows = [fn(seed=s, **kw) for s in seeds]
    keys = rows[0].keys()
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}


if __name__ == "__main__":
    print("=" * 78)
    print("(1) PER-NEURON CAPACITY (closed form) — timing vs rate")
    print("=" * 78)
    for r in per_neuron_levels_and_bits():
        print(r)

    print("\n--- single-layer round()-TTFS quantization error (== rate code) ---")
    for r in single_layer_quantization_error():
        print(r)

    print("\n--- ramp decode effective-weight distortion R(tau)~(T-tau)^2 vs L~(T-tau) ---")
    for blk in ramp_effective_weight_distortion():
        print(f"T={blk['T']}:")
        for row in blk["per_tau"]:
            print("   ", row)

    print("\n" + "=" * 78)
    print("(1b) DO PER-NEURON LEVELS SURVIVE DEPTH? (identity chain, S=8)")
    print("=" * 78)
    for r in identity_chain_levels(depth=4, S=8):
        print(r)

    print("\n" + "=" * 78)
    print("(3) OPTIMAL single-spike decode (ideal staircase) vs GENUINE cascade")
    print("=" * 78)
    for S in (4, 8, 16, 32):
        m = _mean_over_seeds(optimal_vs_genuine, **PRIMARY, S=S)
        print(f"S={S:>2}: cont={m['cont']:.3f} ideal_staircase={m['ideal_staircase']:.3f} "
              f"genuine={m['genuine']:.3f}  | quant_cost={m['cont']-m['ideal_staircase']:+.3f} "
              f"decode_cost(staircase-genuine)={m['ideal_staircase']-m['genuine']:+.3f}")

    print("\n" + "=" * 78)
    print("(2) CAPACITY CEILING vs CORRECTABLE DISTORTION — oracle per-depth theta scale")
    print("=" * 78)
    for S in (8, 16):
        for seed in (0, 1, 2):
            r = oracle_theta_scale(**PRIMARY, S=S, seed=seed)
            print(f"S={S:>2} seed={seed}: baseline={r['baseline']:.4f} -> "
                  f"oracle theta-scale={r['best_acc']:.4f} (gamma={r['best_gamma']})")
    print("\nDONE")
