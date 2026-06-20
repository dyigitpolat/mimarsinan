"""PHASE 3 — push the STATIC calibration-only correction to its per-neuron limit.

The death cascade is a per-DEPTH gain distortion that G (a per-layer theta trim)
largely handles. The RESIDUAL this phase targets: after G, the genuine cascade
still caps below the ideal staircase / LIF. ROOT-CAUSE HYPOTHESIS: the ramp decode
applies a PER-SAMPLE nonlinear gain ``g_eff(tau) = (T - tau + 1)/(T + 1)`` to a
spike arriving at local cycle ``tau``; each input drives a neuron to a different
fire time ``tau``, so the effective gain VARIES PER SAMPLE (and the fire-time
distribution differs PER NEURON). A single static threshold per neuron can only
invert a fire-time-INDEPENDENT gain; it cannot invert a per-sample-varying gain.

This file builds progressively finer CALIBRATION-ONLY corrections and measures the
residual each leaves vs the ideal staircase (``capacity._staircase_acc``, the
optimal linear single-spike decode):

  (1) PER-DEPTH        G (``closed_form.g_relative``)   — one scalar / layer  [baseline]
  (2) PER-NEURON mean  theta_k from each channel's MEAN calib fire-cycle      [calib-only]
  (3) PER-NEURON L2    theta_k that MINIMIZES the per-sample decode L2 to the
                       staircase target (trades mean-bias vs the fire-time SPREAD)
                       — the best a STATIC per-neuron threshold can do on calib   [calib-only]
  (4) ORACLE/NEURON    per-neuron theta L2-fit on the EVAL set — UPPER BOUND on
                       ANY static per-neuron threshold calibration              [oracle]

Plus the per-neuron RESIDUAL DECOMPOSITION (the load-bearing measurement):
  * bias_resid   = |mean_genuine - mean_staircase| per neuron  (closable by a
                   static per-neuron SCALE)
  * spread_resid = std over samples of the residual AFTER the best per-neuron AFFINE
                   (slope+offset) fit to the staircase  (NOT closable by ANY static
                   per-neuron map: this is the irreducible per-sample-tau spread).
And a RE-ENCODE reference (the ENCODE-CHANGE lever, NOT a static calibration): a
per-layer decode->re-encode host boundary resets the ramp compounding, showing the
encode side can move the residual that the threshold side cannot.

DEPLOYABILITY (every theta lever keeps the deployed ramp decode bit-exact ->
NF<->SCM parity holds; only the per-neuron THRESHOLD is used):
  * per-neuron theta == per-OUTPUT-channel ``activation_scale`` == a per-neuron
    threshold trim == folding a per-neuron gain into that neuron's row of W,b. A
    first-class chip parameter (the genuine node already broadcasts a tensor theta;
    verified bit-exact when uniform). DEPLOYABLE as CALIBRATION.
  * oracle per-neuron theta — DEPLOYABLE in form, but L2-fit on the eval target, so
    used ONLY as the static-threshold upper bound.
  * re-encode boundary — an ENCODE/decode change (host op per hop), NOT a static
    calibration; reported to bound the OTHER lever, not as deployable calibration.

Run:  source env/bin/activate
      python docs/.../experiments/per_sample.py            # full (seeds 0-2, S 8/16, depth 3/4)
      python docs/.../experiments/per_sample.py quick       # 2 seeds, S=8, depth 3
Findings: ../33_per_sample_correction.md
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch

torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ART = os.path.dirname(_HERE)
sys.path.insert(0, _ART)
sys.path.insert(0, _HERE)

import cascade_lab  # noqa: F401  (sets sys.path for src/tests/repo)
from cascade_lab import (  # noqa: E402
    _SingleSegmentMLP,
    _accuracy,
    _calibrate_scales,
    cascade_forward,
    digits_task,
    train_continuous,
)
from closed_form import (  # noqa: E402
    C_RHO,
    _restore_params,
    build_flow,
    g_eff,
    g_relative,
    mean_tau0,
)
from cascade_fixtures import _HostOpMLP, install_ttfs_nodes  # noqa: E402
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

PRIMARY = dict(depth=3, width=64, in_dim=64, n_classes=10)
FLOOR = 0.02  # min theta gain (a layer past d_max asks an impossible boost; cap sane).


# =========================================================================== #
# Per-NEURON observers (per-channel, not channel-means).
# =========================================================================== #
def per_neuron_decoded(flow, x, S):
    """Genuine-cascade decoded value per perceptron, shape (B, F), keyed by depth.

    Reads the segment forward's ``node_value_recorder`` side channel — the exact
    deployed decode (already x activation_scale), the same mechanism DFQ uses.
    """
    rec: dict = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by = {id(perceptron_of(n)): v for n, v in rec.items()
          if perceptron_of(n) is not None}
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by.get(id(p))
        if v is not None:
            out[k] = v.reshape(-1, v.shape[-1]).double()
    return out


def staircase_decoded(flow, x):
    """Ideal-staircase decoded value per perceptron, shape (B, F), keyed by depth.

    ``set_cycle_accurate(False)`` is the per-layer analytical TTFS (linear (T-tau)/T
    decode, NO cascade ramp distortion) — the optimal single-spike timing decode and
    the per-neuron TARGET the static correction tries to reproduce.
    """
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    out: dict = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, o, k=k):
            out[k] = o.detach().reshape(-1, o.shape[-1]).double()
        handles.append(p.activation.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return out


def per_neuron_fire_stats(flow, x, S):
    """Per-perceptron per-CHANNEL fire-cycle mean & std on ``x`` (calib-only).

    Returns {depth: (mean_tau[F], std_tau[F], no_fire_frac[F])} over the FIRED
    population per channel. The std is the per-NEURON fire-time SPREAD that drives the
    per-sample g_eff nonlinearity. Reuses char's run_segment fire-cycle instrumentation
    (kept per-channel rather than population-pooled).
    """
    fire_cycle: dict = {}
    orig = TtfsSegmentPolicy.run_segment

    def patched(self, driver, seg_nodes, values, xx):
        T = driver.T
        seg_set = set(seg_nodes)
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
                        ff[n] = torch.full(sp.shape, -1.0, dtype=torch.float64)
                    newly = (sp > 0) & (ff[n] < 0)
                    ff[n][newly] = float(t) - dep[n]
        for n in ff:
            fire_cycle[n] = ff[n]
        return orig(self, driver, seg_nodes, values, xx)

    TtfsSegmentPolicy.run_segment = patched
    try:
        drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
        with torch.no_grad():
            drv(x.double())
    finally:
        TtfsSegmentPolicy.run_segment = orig

    by = {id(perceptron_of(n)): fc for n, fc in fire_cycle.items()
          if perceptron_of(n) is not None}
    out: dict = {}
    for k, p in enumerate(flow.get_perceptrons()):
        fc = by.get(id(p))
        if fc is None:
            continue
        fc = fc.reshape(-1, fc.shape[-1])
        F = fc.shape[-1]
        means = torch.full((F,), float(S), dtype=torch.float64)
        stds = torch.zeros(F, dtype=torch.float64)
        nofire = torch.zeros(F, dtype=torch.float64)
        for j in range(F):
            col = fc[:, j]
            fired = col[col >= 0]
            nofire[j] = float((col < 0).double().mean())
            if fired.numel() >= 1:
                means[j] = float(fired.double().mean())
            if fired.numel() >= 2:
                stds[j] = float(fired.double().std(unbiased=False))
        out[k] = (means, stds, nofire)
    return out


# =========================================================================== #
# Static per-neuron theta correctors.  Every one sets per-OUTPUT-channel theta,
# SEQUENTIALLY (re-measure each layer after correcting upstream so it sees the
# revived input), and re-installs the genuine node (bit-exact decode preserved).
# =========================================================================== #
def _set_layer_theta(p, base_scale, gain_vec, S, flow):
    F = p.layer.weight.shape[0]
    g = torch.ones(F, dtype=torch.float64)
    n = min(F, gain_vec.numel())
    g[:n] = gain_vec[:n]
    with torch.no_grad():
        p.set_activation_scale(base_scale * g)
    install_ttfs_nodes(flow, S)


def apply_per_depth(flow, base_scales, S, c=C_RHO):
    """(1) per-DEPTH G (baseline): one scalar gain per layer (closed_form.g_relative)."""
    for d, p in enumerate(flow.get_perceptrons()):
        _set_layer_theta(p, base_scales[d], torch.full(
            (p.layer.weight.shape[0],), g_relative(S, d, c)), S, flow)


def apply_per_neuron_mean(flow, base_scales, S, calib, c=C_RHO):
    """(2) per-NEURON theta from each channel's MEAN calib fire-cycle.

    g_k = g_eff(tau_k) / g_eff(tau0), RELATIVE to the (healthy) encode layer, capped
    <=1, floor FLOOR — the per-channel generalization of closed_form.calib_fire.
    """
    percs = flow.get_perceptrons()
    install_ttfs_nodes(flow, S)
    tau0 = float(per_neuron_fire_stats(flow, calib, S)[0][0].mean())
    g0 = max(g_eff(tau0, S), 1e-6)
    for d, p in enumerate(percs):
        stats = per_neuron_fire_stats(flow, calib, S)
        tau = stats[d][0] if d in stats else torch.full(
            (p.layer.weight.shape[0],), mean_tau0(S, c) + d * math.sqrt(S),
            dtype=torch.float64)
        g = (g_eff(tau, S) / g0).clamp(FLOOR, 1.0)
        _set_layer_theta(p, base_scales[d], g, S, flow)


def _l2_optimal_gain(gen_col, targ_col):
    """The static per-neuron threshold gain that minimizes the per-sample decode L2.

    Genuine decode scales ~monotonically with 1/theta (smaller theta -> earlier fire
    -> larger value). Approximate the local response as decode ~ alpha * gen (alpha =
    theta_old/theta_new). The L2-optimal alpha for target t given current g is the
    least-squares projection  alpha* = <g, t> / <g, g>  ->  theta_new = theta_old /
    alpha*. Trading the mean-bias against the spread (a high-variance neuron gets a
    smaller |alpha-1| pull), unlike the pure mean-match (2).
    """
    num = (gen_col * targ_col.clamp(min=0)).sum(0)
    den = (gen_col ** 2).sum(0).clamp(min=1e-9)
    alpha = (num / den)
    # alpha = value multiplier we want; theta must shrink by alpha -> gain = 1/alpha.
    return (1.0 / alpha.clamp(min=1e-6)).clamp(FLOOR, 50.0)


def _apply_l2_per_neuron(flow, base_scales, S, x, target_decoded, refine=False):
    """Per-layer (sequential) per-neuron L2-optimal theta fit to ``target_decoded``.

    ``refine`` adds a per-neuron multiplicative grid search (re-running the genuine
    cascade) around the projection estimate, minimizing the TRUE per-sample decode L2
    under the nonlinear cascade — a tighter UPPER BOUND on any static per-neuron theta.
    """
    percs = flow.get_perceptrons()
    grid = torch.tensor([0.6, 0.75, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0], dtype=torch.float64)
    for d, p in enumerate(percs):
        install_ttfs_nodes(flow, S)
        gen = per_neuron_decoded(flow, x, S).get(d)
        targ = target_decoded.get(d)
        if gen is None or targ is None:
            continue
        n = min(gen.shape[-1], targ.shape[-1], p.layer.weight.shape[0])
        gain = _l2_optimal_gain(gen[:, :n], targ[:, :n])
        gain[gen[:, :n].abs().mean(0) < 1e-6] = FLOOR  # dead: theta cannot revive -> floor
        _set_layer_theta(p, base_scales[d], gain, S, flow)
        if not refine:
            continue
        t = targ[:, :n].clamp(min=0)
        best_gain = gain.clone()
        cur = per_neuron_decoded(flow, x, S).get(d)[:, :n]
        best_err = ((cur - t) ** 2).mean(0)
        for f in grid:
            cand = (best_gain * f).clamp(FLOOR, 50.0)
            _set_layer_theta(p, base_scales[d], cand, S, flow)
            dec = per_neuron_decoded(flow, x, S).get(d)[:, :n]
            err = ((dec - t) ** 2).mean(0)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_gain[:n] = torch.where(better, cand[:n], best_gain[:n])
        _set_layer_theta(p, base_scales[d], best_gain, S, flow)


def apply_per_neuron_l2(flow, base_scales, S, calib, c=C_RHO):
    """(3) per-NEURON L2-optimal theta on CALIB (best static threshold, calib-only).

    Warm-start from the mean-match (2) so each layer's L2 fit sees a revived cascade,
    then one L2 refine pass per layer against the calib STAIRCASE target.
    """
    apply_per_neuron_mean(flow, base_scales, S, calib, c=c)
    cur = [p.activation_scale.clone() for p in flow.get_perceptrons()]
    install_ttfs_nodes(flow, S)
    target = staircase_decoded(flow, calib)
    _apply_l2_per_neuron(flow, cur, S, calib, target)


def _per_layer_acc_grid(flow, base_scales, S, xte, yte,
                        grid=(1.0, 0.7, 0.5, 0.35, 0.25)):
    """Per-DEPTH theta-scale coordinate ascent maximizing eval accuracy (the
    capacity.oracle_theta_scale lever; one global multiplier per layer)."""
    percs = flow.get_perceptrons()
    g = [1.0] * len(percs)

    def acc():
        for k, p in enumerate(percs):
            _set_layer_theta(p, base_scales[k], torch.full(
                (p.layer.weight.shape[0],), g[k]), S, flow)
        return float(_accuracy(cascade_forward(flow, xte, S), yte))

    best = acc()
    improved = True
    while improved:
        improved = False
        for k in range(len(percs)):
            g0 = g[k]
            for cand in grid:
                g[k] = cand
                a = acc()
                if a > best + 1e-9:
                    best, g0, improved = a, cand, True
            g[k] = g0
    for k, p in enumerate(percs):
        _set_layer_theta(p, base_scales[k], torch.full(
            (p.layer.weight.shape[0],), g[k]), S, flow)
    return best


def apply_oracle_per_neuron(flow, base_scales, S, calib, xte, yte):
    """(4) ORACLE upper bound on ANY static per-neuron THRESHOLD: best EVAL accuracy
    over several static-theta strategies, so it can never underperform a simpler one.

    Candidates (all static per-neuron theta, decode bit-exact): per-neuron mean-match,
    per-neuron L2 fit to the EVAL staircase (+grid refine), and a per-depth accuracy
    coordinate ascent. The accuracy-best is kept. Reads the eval metric => an UPPER
    BOUND reference, NOT deployable as calibration.
    """
    candidates = []

    apply_per_neuron_mean(flow, base_scales, S, calib, c=C_RHO)
    candidates.append(([p.activation_scale.clone() for p in flow.get_perceptrons()],
                       float(_accuracy(cascade_forward(flow, xte, S), yte))))

    apply_per_neuron_mean(flow, base_scales, S, calib, c=C_RHO)
    cur = [p.activation_scale.clone() for p in flow.get_perceptrons()]
    install_ttfs_nodes(flow, S)
    _apply_l2_per_neuron(flow, cur, S, xte, staircase_decoded(flow, xte), refine=True)
    candidates.append(([p.activation_scale.clone() for p in flow.get_perceptrons()],
                       float(_accuracy(cascade_forward(flow, xte, S), yte))))

    _per_layer_acc_grid(flow, base_scales, S, xte, yte)
    candidates.append(([p.activation_scale.clone() for p in flow.get_perceptrons()],
                       float(_accuracy(cascade_forward(flow, xte, S), yte))))

    best_scales, _ = max(candidates, key=lambda c: c[1])
    for p, s in zip(flow.get_perceptrons(), best_scales):
        with torch.no_grad():
            p.set_activation_scale(s.clone())
    install_ttfs_nodes(flow, S)


# =========================================================================== #
# RE-ENCODE reference: the ENCODE-CHANGE lever (per-hop decode->re-encode host op).
# =========================================================================== #
def reencode_accuracy(*, depth, width, in_dim, n_classes, S, seed):
    """Genuine accuracy of a per-LAYER-segmented cascade (host op between every
    layer => decode->re-encode each hop, resetting the ramp compounding). Bounds the
    ENCODE-side lever (NOT a static threshold calibration). The HostOp's value scale
    (0.7,0.05) is folded into the trained weights so it is not a confound for acc."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _HostOpMLP(depth, width, in_dim, n_classes)
    train_continuous(base, xtr, ytr, epochs=120)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    gen = float(_accuracy(cascade_forward(flow, xte, S), yte))
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        stair = float(_accuracy(flow(xte.double()), yte))
    return {"reencode_genuine": gen, "reencode_staircase": stair}


# =========================================================================== #
# Per-neuron residual decomposition (bias vs irreducible spread).
# =========================================================================== #
def per_neuron_residual(flow, x, S):
    """Over ACTIVE neurons (staircase mean > 1e-3):
       bias_resid   = mean |mean_genuine - mean_staircase|   (closable by static scale)
       spread_resid = mean over neurons of std_samples( genuine - (a*stair+b) )
                      using the per-neuron OLS affine fit  (NOT closable by ANY static
                      per-neuron map: the irreducible per-sample tau spread).
    """
    install_ttfs_nodes(flow, S)
    gen = per_neuron_decoded(flow, x, S)
    stair = staircase_decoded(flow, x)
    install_ttfs_nodes(flow, S)
    bias_t, spread_t = [], []
    for d in gen:
        g = gen[d]
        s = stair[d][:, :g.shape[-1]].clamp(min=0)
        active = s.mean(0) > 1e-3
        if active.sum() == 0:
            continue
        g, s = g[:, active], s[:, active]
        bias_t.append(float((g.mean(0) - s.mean(0)).abs().mean()))
        sm, gm = s.mean(0), g.mean(0)
        cov = ((s - sm) * (g - gm)).mean(0)
        var = ((s - sm) ** 2).mean(0).clamp(min=1e-9)
        a = cov / var
        b = gm - a * sm
        spread_t.append(float((g - (a * s + b)).std(0).mean()))
    return {"bias_resid": float(np.mean(bias_t)) if bias_t else 0.0,
            "spread_resid": float(np.mean(spread_t)) if spread_t else 0.0}


# =========================================================================== #
# Evaluation.
# =========================================================================== #
def _staircase_accuracy(flow, x, y):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        return float(_accuracy(flow(x.double()), y))


_LEVELS = ("per_depth", "per_neuron_mean", "per_neuron_l2", "oracle_neuron")


def evaluate(*, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0):
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)

    def restore():
        _restore_params(flow, base_scales, base_weights, base_biases)
        install_ttfs_nodes(flow, S)

    restore()
    base_gen = float(_accuracy(cascade_forward(flow, xte, S), yte))
    stair = _staircase_accuracy(flow, xte, yte)
    br = per_neuron_residual(flow, xte, S)
    out = {"cont": round(cont_acc, 4), "staircase": round(stair, 4),
           "baseline": round(base_gen, 4),
           "base_bias_resid": round(br["bias_resid"], 4),
           "base_spread_resid": round(br["spread_resid"], 4)}

    appliers = {
        "per_depth": lambda: apply_per_depth(flow, base_scales, S),
        "per_neuron_mean": lambda: apply_per_neuron_mean(flow, base_scales, S, calib),
        "per_neuron_l2": lambda: apply_per_neuron_l2(flow, base_scales, S, calib),
        "oracle_neuron": lambda: apply_oracle_per_neuron(
            flow, base_scales, S, calib, xte, yte),
    }
    for lvl in _LEVELS:
        restore()
        appliers[lvl]()
        out[lvl] = round(float(_accuracy(cascade_forward(flow, xte, S), yte)), 4)
        r = per_neuron_residual(flow, xte, S)
        out[lvl + "_bias_resid"] = round(r["bias_resid"], 4)
        out[lvl + "_spread_resid"] = round(r["spread_resid"], 4)
    restore()
    return out


# =========================================================================== #
# Reports.
# =========================================================================== #
def _agg(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def report(seeds=(0, 1, 2), Ss=(8, 16), depths=(3, 4)):
    print("=" * 96)
    print("PROGRESSIVE STATIC PER-NEURON THRESHOLD CALIBRATION vs the IDEAL STAIRCASE")
    print("  closed% = fraction of the genuine->staircase gap recovered by each STATIC level")
    print("=" * 96)
    for depth in depths:
        for S in Ss:
            rows = [evaluate(depth=depth, S=S, seed=s) for s in seeds]
            a = _agg(rows)
            gap = a["staircase"] - a["baseline"]
            print(f"\n--- depth={depth} S={S} (mean seeds {seeds}) | "
                  f"cont={a['cont']:.3f} staircase={a['staircase']:.3f} "
                  f"baseline_genuine={a['baseline']:.3f}  gap={gap:+.3f} ---")
            print(f"  {'level':<18}{'acc':>7}{'closed%':>9}"
                  f"{'bias_resid':>12}{'spread_resid':>14}")
            print(f"  {'baseline':<18}{a['baseline']:>7.3f}{'-':>9}"
                  f"{a['base_bias_resid']:>12.4f}{a['base_spread_resid']:>14.4f}")
            for lvl in _LEVELS:
                closed = (a[lvl] - a["baseline"]) / gap if abs(gap) > 1e-6 else 0.0
                print(f"  {lvl:<18}{a[lvl]:>7.3f}{closed:>8.0%}"
                      f"{a[lvl + '_bias_resid']:>12.4f}{a[lvl + '_spread_resid']:>14.4f}")
            re = _agg([reencode_accuracy(depth=depth, width=64, in_dim=64,
                                         n_classes=10, S=S, seed=s) for s in seeds])
            print(f"  [encode-change ref] per-hop re-encode genuine="
                  f"{re['reencode_genuine']:.3f} (its own staircase="
                  f"{re['reencode_staircase']:.3f})")


def report_quick():
    report(seeds=(0, 1), Ss=(8,), depths=(3,))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "full"
    if which == "quick":
        report_quick()
    else:
        report()
    print("\nDONE")
