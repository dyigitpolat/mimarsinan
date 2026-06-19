"""DIRECTION B — depth-aware analytical PRE-COMPENSATION (H2).

The single-spike TTFS cascade attenuates deep layers to ~0 (the death cascade):
the calibrated ``activation_scale`` (theta = max|drive|) pushes the normalized
drive ``v_norm = relu(drive)/theta`` low (mean ~0.2), so neurons fire LATE
(``tau = round(S(1 - v_norm))`` near ``S``), leaving a short ramp window and a
tiny downstream membrane integral. Attenuation compounds with depth.

The deployable lever is ``theta`` itself: lowering it raises ``v_norm`` -> earlier
firing -> longer ramp -> the layer survives. We want a per-layer correction
DERIVED FROM CALIBRATION STATISTICS (not searched on the eval metric — that was
tried and over-fits), baked into ``activation_scale`` (which the hardware already
supports — it is the per-neuron threshold theta).

Methods compared (all COLD: no genuine fine-tuning):
  * ``baseline``        — theta = max|drive| (the shipped calibration).
  * ``mean_target``     — theta_d = mean(relu(act_d)) / target. Closed form, per
                          layer, from calib activation stats. ``target`` is the
                          desired mean normalized drive (a healthy firing time
                          ``tau ~ S(1-target)``). theta-independent (ReLU activation
                          does not depend on theta) so no forward ordering needed.
  * ``percentile``      — theta_d = quantile_q(relu(act_d)) (DFQ-style robust max);
                          a parameter-free analytic that clips the heavy tail.
  * ``global_const``    — one shared multiplier on every theta (sanity reference;
                          this is a 1-D metric search, shown only to bound the win).

A method WINS if, at the PRIMARY benchmark (depth=3 digits, S=8), it lifts genuine
accuracy from ~0.074 toward the continuous ~0.944, measured cold on the held-out
test set, robust across seeds {0,1,2}, and trends right across S and depth.

Run:  source env/bin/activate; python docs/.../experiments/precomp.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cascade_lab  # noqa: F401  (sets sys.path to src/tests/repo)
import torch

from cascade_lab import (
    _SingleSegmentMLP,
    _accuracy,
    _calibrate_scales,
    _capture_activation_means,
    _cascade_decoded_means,
    cascade_forward,
    digits_task,
    train_continuous,
)
from cascade_fixtures import install_ttfs_nodes
from mimarsinan.torch_mapping.converter import convert_torch_model


# ---------------------------------------------------------------------------
# Build a trained, converted, calibrated cascade flow (the conversion_gap setup).
# Cached per (depth, seed): retraining a 120-epoch MLP per method/S is the cost,
# and theta is a mutable per-node knob we reset between methods on the SAME flow.
# ---------------------------------------------------------------------------
_FLOW_CACHE: dict = {}


def build_flow(*, depth, width, in_dim, n_classes, seed, epochs=120):
    key = (depth, width, in_dim, n_classes, seed, epochs)
    if key in _FLOW_CACHE:
        return _FLOW_CACHE[key]
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    train_continuous(base, xtr, ytr, epochs=epochs)
    with torch.no_grad():
        cont_acc = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    calib = xtr[:256]
    _calibrate_scales(flow, calib)
    base_scales = [p.activation_scale.clone() for p in flow.get_perceptrons()]
    teacher = _capture_activation_means(flow, xte)
    # Capture per-channel ReLU activation stats on CALIB *before* any TTFS install
    # replaces ``p.activation`` (install_ttfs_nodes swaps the module, so hooking it
    # later would read the spike node, not the ReLU). The ReLU activation is
    # theta-independent, so this single pre-install snapshot is the SSOT for every
    # analytical correction (no cache-order dependence).
    calib_stats = _per_channel_relu_stats(flow, calib)
    base_biases = [p.layer.bias.detach().clone() if p.layer.bias is not None else None
                   for p in flow.get_perceptrons()]
    teacher_calib = _capture_activation_means(flow, calib)
    _FLOW_CACHE[key] = (flow, calib, xte, yte, cont_acc, base_scales, teacher,
                        calib_stats, base_biases, teacher_calib)
    return _FLOW_CACHE[key]


def _restore_biases(flow, base_biases):
    for p, b in zip(flow.get_perceptrons(), base_biases):
        if b is not None and p.layer.bias is not None:
            with torch.no_grad():
                p.layer.bias.copy_(b)


def _per_channel_relu_stats(flow, x):
    """Per-perceptron channel-flattened ReLU-activation tensor on ``x`` (calib).

    Must be called BEFORE ``install_ttfs_nodes`` (which replaces ``p.activation``).
    The ReLU activation is theta-independent, so these stats are a fixed,
    calibration-only description of each layer's drive distribution — exactly the
    kind of statistic an analytical (non-metric-searched) correction may use.
    """
    acts: dict = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, out, k=k):
            acts[k] = out.detach().reshape(-1, out.shape[-1]).clamp(min=0).double()
        handles.append(p.activation.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return acts


def _set_thetas(flow, base_scales, thetas):
    for i, p in enumerate(flow.get_perceptrons()):
        p.set_activation_scale(thetas[i] if isinstance(thetas[i], torch.Tensor)
                               else base_scales[i] * float(thetas[i]))


# ---------------------------------------------------------------------------
# Pre-compensation methods. Each takes the pre-install calib ReLU stats and
# returns a list of theta tensors (per perceptron). All are pure functions of
# calibration statistics — never of the eval metric.
# ---------------------------------------------------------------------------
def thetas_baseline(base_scales, stats):
    return [s.clone() for s in base_scales]


def thetas_mean_target(base_scales, stats, *, target=0.5):
    """theta_d = mean(relu(act_d)) / target  — scalar closed form per layer.

    Picks theta so the calib MEAN positive drive maps to normalized value
    ``target`` -> mean firing time ``tau ~ S(1-target)`` (mid-window for 0.5),
    inverting the death cascade's late-firing.
    """
    out = []
    for k in range(len(base_scales)):
        m = float(stats[k].mean())
        out.append(torch.tensor(max(m / target, 1e-6), dtype=torch.float64))
    return out


def thetas_mean_target_perchannel(base_scales, stats, *, target=0.5):
    """Per-CHANNEL theta_d[c] = mean_c(relu(act_d)) / target (vector threshold).

    The hardware threshold is per-neuron, so a per-channel theta is just as
    deployable as a scalar one; this matches each channel's own drive.
    """
    out = []
    for k in range(len(base_scales)):
        m = stats[k].mean(0).clamp(min=1e-6) / target  # (C,)
        out.append(m.to(torch.float64))
    return out


def thetas_posmean_target(base_scales, stats, *, target=0.5):
    """Like mean_target but over the *fired* (>0) population mean per channel.

    Dead/silent channels (always 0) keep theta tiny -> clamped; live channels are
    centred at ``target``. Closer to the "typical active firing time" the ramp sees.
    """
    out = []
    for k in range(len(base_scales)):
        a = stats[k]
        mask = (a > 1e-9).double()
        denom = mask.sum(0).clamp(min=1.0)
        pm = (a.sum(0) / denom).clamp(min=1e-6) / target  # (C,)
        out.append(pm.to(torch.float64))
    return out


def thetas_percentile(base_scales, stats, *, q=0.5):
    """theta_d = per-channel quantile_q(relu(act_d)) — DFQ-style robust scale."""
    out = []
    for k in range(len(base_scales)):
        qv = torch.quantile(stats[k], q, dim=0).clamp(min=1e-6)
        out.append(qv.to(torch.float64))
    return out


def thetas_global_const(base_scales, stats, *, m=0.45):
    return [s.clone() * m for s in base_scales]


def apply_bias_correction(flow, calib, teacher_calib, S):
    """DFQ-style cold bias correction ON TOP of a theta correction (deployable).

    After theta is set, the cascade's per-channel decoded mean on calib differs
    from the teacher's by an additive residual ``r_d = teacher_mean - cascade_mean``.
    Add ``r_d`` to the layer's bias so the *next* drive is centred where the teacher
    expects it. Done forward (layer-by-layer) so each correction sees the corrected
    upstream. Pure calibration statistics; the bias is a parameter the chip supports.
    """
    percs = flow.get_perceptrons()
    for d in range(len(percs)):
        install_ttfs_nodes(flow, S)
        cas = _cascade_decoded_means(flow, calib, S)
        c = cas.get(d)
        t = teacher_calib.get(d)
        if c is None or t is None:
            continue
        n = min(c.numel(), t.numel())
        # residual in OUTPUT (post-activation, value) units; bias enters pre-activation
        # additively and (within ReLU's active band) shifts the output ~1:1, so add it
        # directly to layer.bias for the active channels.
        resid = (t[:n].clamp(min=0) - c[:n])
        p = percs[d]
        if p.layer.bias is not None and p.layer.bias.numel() >= n:
            with torch.no_grad():
                p.layer.bias[:n] += resid.to(p.layer.bias.dtype)
    install_ttfs_nodes(flow, S)


METHODS = {
    "baseline": thetas_baseline,
    "mean_target": thetas_mean_target,
    "mean_target_pc": thetas_mean_target_perchannel,
    "posmean_pc": thetas_posmean_target,
    "percentile": thetas_percentile,
    "global_const": thetas_global_const,
}


# ---------------------------------------------------------------------------
# Evaluate one method on one (depth, S, seed). Returns gen_acc + attenuation.
# ---------------------------------------------------------------------------
def evaluate(method, *, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0,
             method_kw=None, return_atten=False, bias_correct=False):
    flow, calib, xte, yte, cont_acc, base_scales, teacher, stats, base_biases, \
        teacher_calib = build_flow(
            depth=depth, width=width, in_dim=in_dim, n_classes=n_classes, seed=seed)
    _restore_biases(flow, base_biases)
    thetas = METHODS[method](base_scales, stats, **(method_kw or {}))
    _set_thetas(flow, base_scales, thetas)
    install_ttfs_nodes(flow, S)
    if bias_correct:
        apply_bias_correction(flow, calib, teacher_calib, S)
    gen_acc = _accuracy(cascade_forward(flow, xte, S), yte)
    out = {"cont_acc": round(float(cont_acc), 4), "gen_acc": round(float(gen_acc), 4),
           "gap_closed": None}
    gap0 = float(cont_acc) - 0.0  # reported relative to chance-collapsed baseline below
    out["cont_acc"] = round(float(cont_acc), 4)
    if return_atten:
        cas = _cascade_decoded_means(flow, xte, S)
        prof = []
        for k in range(depth):
            t, c = teacher.get(k), cas.get(k)
            if t is None or c is None:
                continue
            n = min(t.numel(), c.numel())
            tm = float(t[:n].clamp(min=0).mean())
            cm = float(c[:n].mean())
            prof.append(round(cm / tm, 3) if tm > 1e-9 else None)
        out["atten"] = prof
    return out


# ---------------------------------------------------------------------------
# Experiment drivers.
# ---------------------------------------------------------------------------
def primary_table(seeds=(0, 1, 2)):
    print("=== PRIMARY (depth=3 digits, S=8): gen_acc by method x seed ===")
    print(f"{'method':<16} " + " ".join(f"seed{s}" for s in seeds) + "   mean   cont   atten(seed0)")
    base_gen = []
    for method in METHODS:
        accs, conts = [], []
        for s in seeds:
            r = evaluate(method, S=8, seed=s, return_atten=(s == seeds[0]))
            accs.append(r["gen_acc"])
            conts.append(r["cont_acc"])
            if s == seeds[0]:
                atten = r.get("atten")
        m = sum(accs) / len(accs)
        if method == "baseline":
            base_gen = accs
        print(f"{method:<16} " + " ".join(f"{a:.3f}" for a in accs)
              + f"   {m:.4f}  {sum(conts)/len(conts):.3f}  {atten}")
    return base_gen


def sweep_S(method, seeds=(0, 1, 2)):
    print(f"\n=== {method}: gen_acc vs S (depth=3, mean over seeds {seeds}) ===")
    print(f"{'S':>4}  baseline  {method}")
    for S in (4, 8, 16, 32):
        b = sum(evaluate("baseline", S=S, seed=s)["gen_acc"] for s in seeds) / len(seeds)
        m = sum(evaluate(method, S=S, seed=s)["gen_acc"] for s in seeds) / len(seeds)
        print(f"{S:>4}  {b:.4f}    {m:.4f}")


def sweep_depth(method, seeds=(0, 1, 2)):
    print(f"\n=== {method}: gen_acc vs depth (S=8, mean over seeds {seeds}) ===")
    print(f"{'depth':>6}  cont   baseline  {method}")
    for depth in (2, 3, 4):
        confs = [evaluate("baseline", depth=depth, S=8, seed=s) for s in seeds]
        ms = [evaluate(method, depth=depth, S=8, seed=s)["gen_acc"] for s in seeds]
        cont = sum(c["cont_acc"] for c in confs) / len(confs)
        b = sum(c["gen_acc"] for c in confs) / len(confs)
        m = sum(ms) / len(ms)
        print(f"{depth:>6}  {cont:.3f}  {b:.4f}    {m:.4f}")


def target_sweep(seeds=(0, 1, 2)):
    print("\n=== mean_target: target sweep (depth=3 S=8) ===")
    print(f"{'target':>7}  gen_acc(mean)")
    for target in (0.3, 0.4, 0.5, 0.6, 0.7):
        accs = [evaluate("mean_target", S=8, seed=s, method_kw={"target": target})["gen_acc"]
                for s in seeds]
        print(f"{target:>7}  {sum(accs)/len(accs):.4f}   {[round(a,3) for a in accs]}")


def gap_closed_table(seeds=(0, 1, 2)):
    """Fraction of the (cont - baseline_gen) gap that each method closes, PRIMARY."""
    print("\n=== gap-closed fraction (PRIMARY depth=3 S=8) ===")
    print(f"{'method':<18} gen   cont  baseline  gap_closed%")
    bl = [evaluate("baseline", S=8, seed=s) for s in seeds]
    base_gen = sum(r["gen_acc"] for r in bl) / len(bl)
    cont = sum(r["cont_acc"] for r in bl) / len(bl)
    denom = max(cont - base_gen, 1e-6)
    rows = [("mean_target(0.5)", "mean_target", {"target": 0.5}, False),
            ("mean_target(0.6)", "mean_target", {"target": 0.6}, False),
            ("mean_target+bias", "mean_target", {"target": 0.6}, True),
            ("global_const(metric)", "global_const", {"m": 0.45}, False)]
    for label, method, kw, bc in rows:
        g = sum(evaluate(method, S=8, seed=s, method_kw=kw, bias_correct=bc)["gen_acc"]
                for s in seeds) / len(seeds)
        print(f"{label:<18} {g:.3f} {cont:.3f} {base_gen:.3f}    "
              f"{100*(g-base_gen)/denom:5.1f}%")


if __name__ == "__main__":
    primary_table()
    target_sweep()
    sweep_S("mean_target")
    sweep_depth("mean_target")
    gap_closed_table()
    print("\n--- bias-correction ablation (atten->1 but kills accuracy) on PRIMARY ---")
    for s in (0, 1, 2):
        r0 = evaluate("mean_target", S=8, seed=s, return_atten=True)
        r1 = evaluate("mean_target", S=8, seed=s, return_atten=True, bias_correct=True)
        print(f"seed{s}: theta-only gen={r0['gen_acc']} atten={r0['atten']} | "
              f"+bias gen={r1['gen_acc']} atten={r1['atten']}")
