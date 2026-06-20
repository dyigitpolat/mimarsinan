"""DIRECTION A — Analytical decode model + depth-budget law (H1).

Derives the closed-form decoded value of the single-spike ramp-integrate cascade
and validates it against ``cascade_lab``. Run from the repo root with the venv
active:  ``python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/char_decode_law.py``

This file is READ-ONLY w.r.t. cascade_lab; it only imports it.

THE MECHANISM (verified, see __main__):
  * Encode:  a value v in [0,1] (post-theta-normalisation) fires a single spike at
    LOCAL cycle  tau = round(S*(1-v))  -> latched ramp of length (S - tau).
  * A consumer double-integrates the arriving latched ramps. With one W=1 input
    latched from local cycle a, membrane(t) = W * sum_{j=1}^{t-a+1} j (quadratic),
    so it crosses 1 quickly; decode = (S - t_fire)/S.
  * Latency:  layer d cannot fire before global cycle d (1 cycle/hop), and only
    integrates inside its OWN window [d, d+S). The producer fired one hop earlier,
    so relative arrival inside the consumer window = (producer tau) - 1.

THE LAW:
  * Per-layer retention rho(S) = 1 - c/S    (encode/quantization+threshold loss).
  * Window-shortening: the LOCAL fire-cycle drifts later with depth because each
    layer's inputs arrive progressively later (compressed range), and a quadratic
    ramp that starts later/weaker needs more cycles to cross 1. When the required
    crossing cycle exceeds the window length S, the neuron never fires -> death.
  * Compounding: atten[d] ~= prod_{k<=d} rho_k, super-geometric once a layer's
    fire-cycle nears S (the drift accelerates).
  * Depth budget: d_max(S) = depth at which the typical local fire-cycle reaches S.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(__file__)
_LAB = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "tests"), _LAB, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_lab import (  # noqa: E402
    _capture_activation_means,
    _cascade_decoded_means,
    attenuation_profile,
    conversion_gap,
    digits_task,
    train_continuous,
)
from cascade_fixtures import (  # noqa: E402
    _SingleSegmentMLP,
    _calibrate_scales,
    install_ttfs_nodes,
)
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import (  # noqa: E402
    TTFSSegmentForward,
)
from mimarsinan.spiking.segment_partition import (  # noqa: E402
    is_encoding_perceptron,
    perceptron_of,
)
from mimarsinan.spiking.segment_policy_ttfs import TtfsSegmentPolicy  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


# ----------------------------------------------------------------------------- #
# 0. Closed-form reference decoder (the analytical model under test).
# ----------------------------------------------------------------------------- #
def encode_tau(v: float, S: int) -> int:
    """Local spike cycle of value v in [0,1]:  tau = round(S(1-v))."""
    return int(round(S * (1.0 - min(max(v, 0.0), 1.0))))


def analytic_decode(arrivals_weights, S: int, consumer_lat: int = 0):
    """Closed-form genuine decode of one neuron from its arriving latched ramps.

    ``arrivals_weights`` = list of (a_local, w): each input's latched-ramp start
    cycle (local to the consumer window) and its effective weight (in_scale/theta
    folded). Returns decoded value (S - t_fire)/S, or 0.0 if it never crosses in S.

    Membrane is the double integral of the summed latched ramps:
        ramp(t)   = sum_i w_i * [t >= a_i]
        membrane(t) = sum_{s=0}^{t} ramp(s)
    Fire at first t with membrane >= 1. This reproduces TTFSActivation exactly.
    """
    ramp = 0.0
    membrane = 0.0
    for t in range(S):
        ramp += sum(w for a, w in arrivals_weights if t >= a)
        membrane += ramp
        if membrane >= 1.0:
            return (S - t) / S
    return 0.0


def closed_form_matches_node(S: int = 8, seed: int = 0, n: int = 500) -> float:
    """Max abs diff between analytic_decode and the genuine TTFSActivation node."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        N = int(rng.integers(1, 12))
        arr = [(int(rng.integers(0, S)), float(rng.uniform(-0.6, 0.6))) for _ in range(N)]
        # genuine node:
        m = TTFSActivation(T=S, activation_scale=1.0, input_scale=1.0, bias=None,
                           thresholding_mode="<=", encoding=False)
        m.set_cycle_accurate(True)
        m.reset_state()
        latch = None
        accum = 0.0
        for t in range(S):
            xt = sum(w * (1.0 if t >= a else 0.0) for a, w in arr)
            s = m.forward(torch.tensor([xt], dtype=torch.float64))
            latch = s if latch is None else torch.maximum(latch, s)
            accum += float(latch)
        genuine = accum / S
        worst = max(worst, abs(genuine - analytic_decode(arr, S)))
    return worst


# ----------------------------------------------------------------------------- #
# 1. Per-layer retention rho(S) = 1 - c/S   (fit on the random-init depth-0 ratio)
# ----------------------------------------------------------------------------- #
def fit_rho_law(Ss=(4, 8, 16, 32, 64, 128), seeds=(0, 1, 2)):
    out = {}
    for seed in seeds:
        rho0 = [attenuation_profile(depth=2, S=S, seed=seed)[0]["ratio"] for S in Ss]
        cs = [(1.0 - r) * S for r, S in zip(rho0, Ss)]
        out[seed] = {"S": list(Ss), "rho0": [round(r, 4) for r in rho0],
                     "c_per_S": [round(c, 3) for c in cs], "c_mean": float(np.mean(cs))}
    return out


# ----------------------------------------------------------------------------- #
# 2. Spike-fire-time distribution by depth (the window-shortening mechanism).
# ----------------------------------------------------------------------------- #
def fire_time_by_depth(depth=4, width=64, S=8, seed=0, epochs=120):
    """Mean LOCAL fire-cycle, no-fire fraction, and decode ratio per depth in the
    TRAINED cascade. The local-fire-cycle drift toward S is the death mechanism."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, width, 64, 10)
    train_continuous(base, xtr, ytr, epochs=epochs)
    flow = convert_torch_model(base, (64,), 10, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    teacher = _capture_activation_means(flow, xte)
    install_ttfs_nodes(flow, S)

    fire_cycle: dict = {}
    orig = TtfsSegmentPolicy.run_segment

    def patched(self, driver, seg_nodes, values, xx):
        T = driver.T
        seg_set = set(seg_nodes)
        ext = driver.external_consumed(seg_nodes)
        dep = self.segment_depths(driver, seg_nodes)
        zeros = self._segment_output_zeros(driver, seg_nodes, values, xx)
        n_cycles = T + max(dep.values(), default=0)
        boundary_trains: dict = {}

        def boundary_spikes(src, t):
            tr = boundary_trains.get(src)
            if tr is None:
                tr = self._boundary_single_spike_train(values[src], T, n_cycles)
                boundary_trains[src] = tr
            return tr[t]

        def read(src, out, perc_prev, t, consumer):
            if src not in seg_set:
                if is_encoding_perceptron(consumer):
                    return values[src]
                return boundary_spikes(src, t)
            if perceptron_of(src) is not None:
                return perc_prev.get(src, zeros[src])
            return out[src]

        perc_prev: dict = {}
        ff = {n: None for n in seg_nodes if perceptron_of(n) is not None}
        for t in range(n_cycles):
            out: dict = {}
            for n in seg_nodes:
                if t < dep[n] or t >= dep[n] + T:
                    out[n] = zeros[n]
                    continue
                d = driver._deps.get(n, [])
                if len(d) == 1:
                    inp = read(d[0], out, perc_prev, t, n)
                elif len(d) == 0:
                    inp = xx
                else:
                    inp = tuple(read(dep_, out, perc_prev, t, n) for dep_ in d)
                out[n] = n.forward(inp)
            for n in seg_nodes:
                if perceptron_of(n) is not None:
                    perc_prev[n] = out[n]
            for n in ff:
                if n in out:
                    sp = out[n]
                    if ff[n] is None:
                        ff[n] = torch.full(sp.shape[1:], -1.0, dtype=torch.float64)
                    fired_now = (sp > 0).any(dim=0) if sp.dim() > 1 else (sp > 0)
                    newly = fired_now & (ff[n] < 0)
                    ff[n][newly] = t - dep[n]
        for n in ff:
            fire_cycle[n] = ff[n]
        return orig(self, driver, seg_nodes, values, xx)

    TtfsSegmentPolicy.run_segment = patched
    try:
        drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
        with torch.no_grad():
            drv(xte.double())
    finally:
        TtfsSegmentPolicy.run_segment = orig

    cascade = _cascade_decoded_means(flow, xte, S)
    by_perc = {id(perceptron_of(n)): fc for n, fc in fire_cycle.items()
               if perceptron_of(n) is not None}
    rows = []
    for k, p in enumerate(flow.get_perceptrons()):
        fc = by_perc.get(id(p))
        if fc is None:
            continue
        fired = fc[fc >= 0]
        t = teacher[k][: cascade[k].numel()].clamp(min=0)
        tm = float(t.mean())
        cm = float(cascade[k].mean())
        rows.append({
            "depth": k,
            "mean_local_fire": float(fired.double().mean()) if fired.numel() else None,
            "frac_no_fire": float((fc < 0).double().mean()),
            "ratio": cm / tm if tm > 1e-9 else 0.0,
            "S": S,
        })
    return rows


# ----------------------------------------------------------------------------- #
# 3. Compounding test: atten[d] vs product of per-layer factors.
# ----------------------------------------------------------------------------- #
def compounding_test(depth=4, S=8, seeds=(0, 1, 2)):
    """Is atten[d] ~= prod_{k<=d} (atten[k]/atten[k-1])?  (trivially true) — the
    real test: is the per-layer factor STABLE (geometric) or does it accelerate
    (super-geometric collapse)? Report per-layer factor f[d]=atten[d]/atten[d-1]."""
    out = {}
    for seed in seeds:
        r = conversion_gap(depth=depth, S=S, seed=seed, width=64, in_dim=64, n_classes=10)
        at = [a if a is not None else 0.0 for a in r["atten_ratio_by_depth"]]
        factors = [at[0]] + [
            (at[d] / at[d - 1] if at[d - 1] > 1e-6 else 0.0) for d in range(1, len(at))
        ]
        out[seed] = {"atten": [round(a, 3) for a in at],
                     "per_layer_factor": [round(f, 3) for f in factors],
                     "gen_acc": r["gen_acc"], "cont_acc": r["cont_acc"]}
    return out


# ----------------------------------------------------------------------------- #
# 4. Depth budget d_max(S): depth at which genuine acc collapses to chance.
# ----------------------------------------------------------------------------- #
def fit_dmax(Ss=(4, 8, 16, 32, 64), depths=(2, 3, 4, 6, 8), seeds=(0, 1, 2),
             chance=0.12):
    """For each S, the largest depth whose mean genuine acc still beats chance+margin."""
    table = {}
    dmax = {}
    for S in Ss:
        row = {}
        last_good = 0
        for d in depths:
            accs = [conversion_gap(depth=d, S=S, seed=s, width=64, in_dim=64,
                                   n_classes=10)["gen_acc"] for s in seeds]
            mean_acc = float(np.mean(accs))
            row[d] = round(mean_acc, 3)
            if mean_acc > chance + 0.10:
                last_good = d
        table[S] = row
        dmax[S] = last_good
    return table, dmax


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    print("=" * 78)
    print("0. CLOSED-FORM DECODER == GENUINE NODE (validation of the model)")
    print("=" * 78)
    for S in (4, 8, 16, 32):
        w = closed_form_matches_node(S=S, seed=0, n=400)
        print(f"  S={S:>2}: max |analytic_decode - TTFSActivation| = {w:.2e}")

    print("\n" + "=" * 78)
    print("1. PER-LAYER RETENTION LAW  rho(S) = 1 - c/S  (random-init depth-0)")
    print("=" * 78)
    fit = fit_rho_law()
    cs = []
    for seed, d in fit.items():
        print(f"  seed={seed}: S={d['S']} rho0={d['rho0']}  c/S={d['c_per_S']}  c~={d['c_mean']:.3f}")
        cs.append(d["c_mean"])
    print(f"  >>> fitted c = {np.mean(cs):.3f} +- {np.std(cs):.3f}  =>  rho(S) ~= 1 - {np.mean(cs):.2f}/S")

    print("\n" + "=" * 78)
    print("2. WINDOW-SHORTENING: mean LOCAL fire-cycle drifts toward S with depth")
    print("=" * 78)
    for S in (8, 16, 32):
        print(f"  --- S={S} (window length {S}) ---")
        for seed in (0, 1, 2):
            rows = fire_time_by_depth(depth=4, S=S, seed=seed)
            mlf = [r["mean_local_fire"] for r in rows]
            nf = [round(r["frac_no_fire"], 2) for r in rows]
            rt = [round(r["ratio"], 3) for r in rows]
            print(f"    seed={seed}: mean_local_fire={[round(x,2) if x is not None else None for x in mlf]} "
                  f"no_fire={nf} ratio={rt}")

    print("\n" + "=" * 78)
    print("3. COMPOUNDING: per-layer factor f[d]=atten[d]/atten[d-1] (depth=4)")
    print("=" * 78)
    for S in (8, 16, 32):
        print(f"  --- S={S} ---")
        cc = compounding_test(depth=4, S=S)
        for seed, d in cc.items():
            print(f"    seed={seed}: atten={d['atten']} factor={d['per_layer_factor']} "
                  f"gen_acc={d['gen_acc']}")

    print("\n" + "=" * 78)
    print("4. DEPTH BUDGET d_max(S)  (mean genuine acc over seeds; chance~0.12)")
    print("=" * 78)
    table, dmax = fit_dmax()
    depths = sorted({d for row in table.values() for d in row})
    print("    S \\ depth |" + "".join(f"{d:>7}" for d in depths) + "  | d_max")
    for S, row in table.items():
        cells = "".join(f"{row.get(d, '-'):>7}" for d in depths)
        print(f"    S={S:>3}     |{cells}  |  {dmax[S]}")
    print("\n  prediction d_max ~ T = S (depth budget law). Fitted d_max(S):")
    for S, dm in dmax.items():
        print(f"    S={S:>3}: d_max={dm}  (d_max/S = {dm / S:.2f})")
