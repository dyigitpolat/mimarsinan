"""Phase 3 — N2 (bias-only phase-advance FT) vs full FT, and a decomposition of the
genuine -> ideal-staircase residual.

Two questions (artifact 32_n2_residual_diagnosis.md):

  Q1  Does bias-only FT (N2: freeze weight DIRECTIONS, train ONLY layer.bias through
      the genuine single-spike cascade) reach HIGHER deployed accuracy than full
      genuine FT (the teacher-blend-equivalent: train ALL weights+biases through the
      genuine forward)?  Controlled {full-FT, N2, G+full-FT, G+N2} x depth{3,4} x
      S{8,16}, multi-seed, all from the SAME continuous base + calibration, compared
      to the ideal staircase.  Isolate WHY (does the bias-only constraint find the
      phase-advance solution full FT misses / overshoots?).

  Q2  Decompose the residual genuine-FT -> ideal-staircase into:
       (a) SURROGATE-GRADIENT NOISE: wider / annealed ATan surrogate, more steps.
       (b) the PER-SAMPLE (T-tau)^2 GAIN: each input -> a different fire-time tau per
           neuron -> a per-sample nonlinear gain a static per-depth theta cannot
           invert.  Does a richer per-NEURON (vs per-depth) static correction help?
           And what is the per-SAMPLE oracle upper bound (the irreducible part of (b))?
       (c) GENERALIZATION: train- vs test-accuracy on the genuine cascade per method.

Everything runs on the isolated ``cascade_lab`` harness (CPU, float64, seconds).
We do NOT edit cascade_lab.py / other agents' files; we import them.

Run:  source env/bin/activate
      python docs/.../experiments/n2_deep.py q1     # controlled comparison (Q1)
      python docs/.../experiments/n2_deep.py q2a    # surrogate noise (Q2a)
      python docs/.../experiments/n2_deep.py q2b    # per-sample gain (Q2b)
      python docs/.../experiments/n2_deep.py q2c    # generalization (Q2c)
      python docs/.../experiments/n2_deep.py why     # N2-vs-full mechanism probe
      python docs/.../experiments/n2_deep.py         # all of the above
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ART = os.path.dirname(_HERE)
_REPO = os.path.abspath(os.path.join(_ART, "..", ".."))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "tests"), _ART, _REPO, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cascade_lab  # noqa: F401,E402  (sets sys.path to src/tests)
from cascade_lab import (  # noqa: E402
    _SingleSegmentMLP,
    _accuracy,
    _calibrate_scales,
    cascade_forward,
    digits_task,
    train_continuous,
)
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402

C_RHO = 1.91  # per-layer retention constant (char law), see closed_form.py


# --------------------------------------------------------------------------- #
# Shared infra: train continuous ONCE per (depth, seed); cheap fresh flows.
# --------------------------------------------------------------------------- #
_BASE_CACHE: dict = {}
EP = 120


def _trained_base(depth, seed, *, width=64, in_dim=64, n_classes=10, epochs=EP):
    key = (depth, seed, width, in_dim, n_classes, epochs)
    if key not in _BASE_CACHE:
        torch.manual_seed(seed)
        xtr, ytr, xte, yte = digits_task(seed=seed + 1)
        base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
        train_continuous(base, xtr, ytr, epochs=epochs)
        with torch.no_grad():
            cont = float(_accuracy(base(xte.float()), yte))
        _BASE_CACHE[key] = (base.state_dict(), xtr, ytr, xte, yte, cont,
                            (depth, width, in_dim, n_classes))
    return _BASE_CACHE[key]


def _fresh_flow(depth, seed, **kw):
    sd, xtr, ytr, xte, yte, cont, dims = _trained_base(depth, seed, **kw)
    d, width, in_dim, n_classes = dims
    base = _SingleSegmentMLP(d, width, in_dim, n_classes)
    base.load_state_dict(sd)
    base.eval()
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    return flow, xtr, ytr, xte, yte, cont


def _install_and_double(flow, S):
    install_ttfs_nodes(flow, S)
    flow.double()
    return flow


def _gen_acc(flow, x, y, S):
    _install_and_double(flow, S)
    with torch.no_grad():
        return float(_accuracy(cascade_forward(flow, x, S), y))


def _staircase_acc(flow, x, y, S):
    """Ideal staircase: linear per-layer (T-tau)/T decode, no cascade ramp."""
    _install_and_double(flow, S)
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        acc = float(_accuracy(flow(x.double()), y))
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(True)
    return acc


# --------------------------------------------------------------------------- #
# G — gain correction (the deployed ttfs_gain_correction, geometric rule).
# --------------------------------------------------------------------------- #
def apply_gain_correction(flow, S, *, c=C_RHO):
    """theta_d <- theta_d * rho_0 * gamma^d  (char geometric law, derived base)."""
    rho0 = 1.0 - c / S
    gamma = 1.0 - math.sqrt(S) / (S + 1.0)
    percs = flow.get_perceptrons()
    n = len(percs)
    for d, p in enumerate(percs):
        if d == n - 1:  # readout untouched (argmax invariant to a positive scale)
            continue
        g = max(rho0 * (gamma ** d), 0.02)
        p.set_activation_scale(p.activation_scale * g)
    return flow


# --------------------------------------------------------------------------- #
# Generic genuine-cascade fine-tune (one code path; the FT MODE is a flag set).
#   mode="full"  -> train weights + biases
#   mode="bias"  -> train ONLY biases (N2: freeze weight directions)
# --------------------------------------------------------------------------- #
def genuine_ft(flow, xtr, ytr, S, *, mode="bias", epochs=40, lr=2e-2,
               surrogate_alpha=2.0, surrogate_temp=0.5, bs=256,
               alpha_anneal=None, record_bias_delta=False):
    """Fine-tune through the genuine single-spike cascade.

    ``alpha_anneal`` (a, b): linearly ramp surrogate_alpha a->b over epochs (wider
    early surrogate -> sharper late).  Returns the flow (and, if requested, the
    per-layer mean learned bias_norm delta = bias/theta change, for the WHY probe)."""
    _install_and_double(flow, S)
    percs = flow.get_perceptrons()

    bias0 = None
    if record_bias_delta:
        bias0 = [p.layer.bias.detach().clone() if p.layer.bias is not None else None
                 for p in percs]
        theta = [float(p.activation_scale) for p in percs]

    train_params = []
    for p in percs:
        train_w = (mode == "full")
        p.layer.weight.requires_grad_(train_w)
        if train_w:
            train_params.append(p.layer.weight)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(True)
            train_params.append(p.layer.bias)

    opt = torch.optim.Adam(train_params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    for ep in range(epochs):
        if alpha_anneal is not None:
            a, b = alpha_anneal
            alpha = a + (b - a) * (ep / max(epochs - 1, 1))
        else:
            alpha = surrogate_alpha
        for m in flow.modules():
            if isinstance(m, TTFSActivation):
                m.set_surrogate_alpha(alpha)
        perm = torch.randperm(xtr.shape[0])
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = cascade_forward(flow, xtr[idx], S, grad=True,
                                     surrogate_temp=surrogate_temp)
            lossf(logits, ytr[idx]).backward()
            opt.step()

    if record_bias_delta:
        deltas = []
        for k, p in enumerate(percs):
            if bias0[k] is None:
                deltas.append(None)
                continue
            d = (p.layer.bias.detach() - bias0[k]) / max(theta[k], 1e-9)
            deltas.append(float(d.mean()))
        return flow, deltas
    return flow


# =========================================================================== #
# Q1 — controlled {full-FT, N2, G+full-FT, G+N2} vs ideal staircase.
# =========================================================================== #
def eval_q1_cell(depth, S, seed, *, epochs=40, lr=2e-2):
    """One (depth,S,seed) cell: from the SAME continuous base + calibration, run the
    four FT variants and report test accuracy of each + the ideal staircase + cont."""
    out = {}
    # references (no FT)
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    out["cont"] = cont
    out["staircase"] = _staircase_acc(flow, xte, yte, S)
    out["base_gen"] = _gen_acc(flow, xte, yte, S)

    def run(mode, with_g):
        f, xtr_, ytr_, xte_, yte_, _ = _fresh_flow(depth, seed)
        if with_g:
            _install_and_double(f, S)
            apply_gain_correction(f, S)
        genuine_ft(f, xtr_, ytr_, S, mode=mode, epochs=epochs, lr=lr)
        with torch.no_grad():
            return float(_accuracy(cascade_forward(f, xte_, S), yte_))

    out["full_ft"] = run("full", False)
    out["n2"] = run("bias", False)
    out["g_full_ft"] = run("full", True)
    out["g_n2"] = run("bias", True)
    return out


def report_q1(seeds=(0, 1, 2), cells=((3, 8), (3, 16), (4, 8), (4, 16))):
    print("=" * 92)
    print("Q1  CONTROLLED FT COMPARISON (test acc; same continuous base + calib per cell)")
    print("    full-FT = all weights+biases ; N2 = bias-only ; G = geom gain correction")
    print("    staircase = IDEAL single-spike decode (the recoverable ceiling)")
    print("=" * 92)
    hdr = (f"{'d':>2}{'S':>4}{'cont':>7}{'stair':>7}{'base':>7}"
           f"{'fullFT':>8}{'N2':>7}{'G+full':>8}{'G+N2':>7}"
           f"{'N2-full':>9}{'%stair(best)':>13}")
    print(hdr)
    agg = {}
    for depth, S in cells:
        rows = [eval_q1_cell(depth, S, s) for s in seeds]
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        best = max(m["full_ft"], m["n2"], m["g_full_ft"], m["g_n2"])
        frac = (best / m["staircase"] * 100) if m["staircase"] > 1e-9 else 0.0
        agg[(depth, S)] = m
        print(f"{depth:>2}{S:>4}{m['cont']:>7.3f}{m['staircase']:>7.3f}{m['base_gen']:>7.3f}"
              f"{m['full_ft']:>8.3f}{m['n2']:>7.3f}{m['g_full_ft']:>8.3f}{m['g_n2']:>7.3f}"
              f"{m['n2']-m['full_ft']:>+9.3f}{frac:>12.1f}%")
    return agg


# =========================================================================== #
# WHY probe — does the bias-only constraint find the phase-advance full-FT misses?
# =========================================================================== #
def report_why(seeds=(0, 1, 2), cell=(4, 8), epochs=40):
    depth, S = cell
    print("\n" + "=" * 92)
    print(f"WHY  N2 vs full-FT mechanism probe  (depth={depth} S={S})")
    print("    learned mean bias_norm delta per depth (positive == phase-advance into")
    print("    the faithful early window); train/test gap; weight-norm drift for full-FT")
    print("=" * 92)
    for mode in ("bias", "full"):
        bd_acc, tr_acc, deltas_all, wdrift_all = [], [], [], []
        for s in seeds:
            flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, s)
            _install_and_double(flow, S)
            w0 = [p.layer.weight.detach().clone() for p in flow.get_perceptrons()]
            flow, deltas = genuine_ft(flow, xtr, ytr, S, mode=mode, epochs=epochs,
                                      record_bias_delta=True)
            with torch.no_grad():
                te = float(_accuracy(cascade_forward(flow, xte, S), yte))
                tr = float(_accuracy(cascade_forward(flow, xtr, S), ytr))
            wd = [float((p.layer.weight.detach() - w0[k]).norm() / (w0[k].norm() + 1e-9))
                  for k, p in enumerate(flow.get_perceptrons())]
            bd_acc.append(te); tr_acc.append(tr)
            deltas_all.append([d for d in deltas if d is not None])
            wdrift_all.append(wd)
        md = np.mean(np.array(deltas_all), axis=0)
        mwd = np.mean(np.array(wdrift_all), axis=0)
        print(f"  mode={mode:>5}: test={np.mean(bd_acc):.3f} train={np.mean(tr_acc):.3f} "
              f"(gen-gap={np.mean(tr_acc)-np.mean(bd_acc):+.3f})")
        print(f"            mean bias_norm delta/depth = {[round(float(x),3) for x in md]}")
        if mode == "full":
            print(f"            rel weight drift /depth    = {[round(float(x),3) for x in mwd]}")


# =========================================================================== #
# Q2a — SURROGATE-GRADIENT NOISE: wider / annealed surrogate, more steps.
# =========================================================================== #
def report_q2a(seeds=(0, 1, 2), cell=(3, 8)):
    depth, S = cell
    print("\n" + "=" * 92)
    print(f"Q2a  SURROGATE-GRADIENT NOISE  (depth={depth} S={S}; N2 bias-only FT)")
    print("    Does a wider / annealed ATan surrogate or more steps close the")
    print("    genuine-FT -> staircase residual?  alpha LOW == WIDE (softer) surrogate.")
    print("=" * 92)
    stairs = []
    for s in seeds:
        flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, s)
        stairs.append(_staircase_acc(flow, xte, yte, S))
    stair = float(np.mean(stairs))
    print(f"  ideal staircase (target) = {stair:.3f}")

    def run(label, **ftkw):
        accs = []
        for s in seeds:
            flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, s)
            genuine_ft(flow, xtr, ytr, S, mode="bias", **ftkw)
            with torch.no_grad():
                accs.append(float(_accuracy(cascade_forward(flow, xte, S), yte)))
        m = float(np.mean(accs))
        print(f"  {label:<34} acc={m:.3f}  resid->stair={stair-m:+.3f}")
        return m

    base = run("alpha=2.0 epochs=40 (default)", epochs=40, surrogate_alpha=2.0)
    run("alpha=1.0 (wider)            ", epochs=40, surrogate_alpha=1.0)
    run("alpha=0.5 (much wider)       ", epochs=40, surrogate_alpha=0.5)
    run("alpha=4.0 (sharper)          ", epochs=40, surrogate_alpha=4.0)
    run("alpha anneal 0.5->4.0        ", epochs=40, alpha_anneal=(0.5, 4.0))
    run("alpha=2.0 epochs=120 (more)  ", epochs=120, surrogate_alpha=2.0)
    run("alpha=2.0 ep=120 lr=5e-3     ", epochs=120, surrogate_alpha=2.0, lr=5e-3)
    run("anneal 0.5->4.0 epochs=120   ", epochs=120, alpha_anneal=(0.5, 4.0))


# =========================================================================== #
# Q2b — the PER-SAMPLE (T-tau)^2 GAIN: per-depth vs per-neuron vs per-sample.
# =========================================================================== #
def _genuine_decoded_per_perc(flow, x, S):
    """Per-perceptron genuine-cascade decoded values (B, F) keyed by depth index."""
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
    from mimarsinan.spiking.segment_partition import perceptron_of
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


def _teacher_per_perc(flow, x):
    """Per-perceptron continuous (clamped-ReLU, theta-normalized) activation (B,F)."""
    means: dict = {}
    handles = []
    percs = flow.get_perceptrons()
    for k, p in enumerate(percs):
        def hook(_m, _i, out, k=k, p=p):
            v = torch.relu(out.detach()).reshape(-1, out.shape[-1]).double()
            means[k] = (v / float(p.activation_scale)).clamp(0, 1)
        handles.append(p.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return means


def report_q2b(seeds=(0, 1, 2), cell=(3, 8)):
    depth, S = cell
    print("\n" + "=" * 92)
    print(f"Q2b  THE PER-SAMPLE (T-tau)^2 GAIN  (depth={depth} S={S})")
    print("    Decompose the genuine-vs-teacher per-neuron decoded-value error into")
    print("    a per-DEPTH scalar gain, a per-NEURON gain, and the per-SAMPLE residual.")
    print("    'explained' = variance of the value error removed by that correction.")
    print("    Then: ORACLE static corrections applied to the cascade -> accuracy gained.")
    print("=" * 92)

    for s in seeds:
        flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, s)
        _install_and_double(flow, S)
        gen = _genuine_decoded_per_perc(flow, xte, S)
        teach = _teacher_per_perc(flow, xte)
        # per-neuron value-error variance decomposition on the NON-readout layers
        tot_var = depth_var = neuron_var = 0.0
        for k in range(depth - 1):  # exclude readout (argmax invariant)
            if k not in gen or k not in teach:
                continue
            g = gen[k]; t = teach[k]
            n = min(g.shape[0], t.shape[0])
            g = g[:n]; t = t[:n]
            err = g - t                        # (B,F) raw decode error
            tot_var += float((err ** 2).mean())
            # per-depth scalar gain alpha_d minimizing ||alpha_d*t - g||? Equivalent
            # framing: residual of g after the best per-depth scale of t.
            ad = float((g * t).sum() / (t * t).sum().clamp(min=1e-12))
            depth_var += float(((g - ad * t) ** 2).mean())
            # per-neuron scale a_f
            af = (g * t).sum(0) / (t * t).sum(0).clamp(min=1e-12)
            neuron_var += float(((g - af * t) ** 2).mean())
        if tot_var <= 0:
            continue
        exp_depth = 1 - depth_var / tot_var
        exp_neuron = 1 - neuron_var / tot_var
        print(f"  seed={s}: value-err var explained by  per-DEPTH gain={exp_depth*100:5.1f}%  "
              f"per-NEURON gain={exp_neuron*100:5.1f}%  per-SAMPLE residual={neuron_var/tot_var*100:5.1f}%")

    # ORACLE static corrections -> accuracy delta (what each tier of correction buys)
    print("  --- oracle static encode corrections -> genuine test accuracy ---")
    print(f"  {'tier':<26}{'gen_acc':>9}{'vs base':>9}")
    for tier in ("base", "per_depth", "per_neuron", "per_sample"):
        accs = []
        for s in seeds:
            accs.append(_oracle_corrected_acc(depth, S, s, tier))
        print(f"  {tier:<26}{np.mean(accs):>9.3f}"
              f"{np.mean(accs) - (np.mean([_oracle_corrected_acc(depth, S, s, 'base') for s in seeds]) if tier!='base' else np.mean(accs)):>+9.3f}")


def _oracle_corrected_acc(depth, S, seed, tier):
    """Apply an ORACLE static correction to each non-readout layer's decoded value,
    matched to the teacher, then run the readout on the corrected features.

    tier:
      base       -> no correction (genuine decode straight to readout)
      per_depth  -> multiply each layer's decoded vector by the oracle per-depth scalar
      per_neuron -> oracle per-neuron scalar
      per_sample -> oracle per-(neuron,sample) scalar == clamp the genuine to teacher
                    (the UPPER BOUND of any static encode correction)

    The correction is applied in the VALUE domain between segments — i.e. it models
    the best a per-depth / per-neuron / per-sample encode pre-scale could do.  Only
    the FINAL (readout) layer's logits decide accuracy, so we recompute the readout
    on the corrected last-hidden features.  This bounds Q2b's contribution.
    """
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    _install_and_double(flow, S)
    gen = _genuine_decoded_per_perc(flow, xte, S)
    teach = _teacher_per_perc(flow, xte)
    last_hidden = depth - 2  # feature feeding the readout
    if last_hidden < 0:
        return _gen_acc(flow, xte, yte, S)
    g = gen[last_hidden]
    t = teach[last_hidden]
    n = min(g.shape[0], t.shape[0])
    g = g[:n]; t = t[:n]
    if tier == "base":
        feat = g
    elif tier == "per_depth":
        # one scalar per layer minimizing ||a*g - t||: the best per-depth gain.
        a = (g * t).sum() / (g * g).sum().clamp(min=1e-12)
        feat = g * a
    elif tier == "per_neuron":
        # one scalar per neuron minimizing ||a_f*g_f - t_f||.
        a = (g * t).sum(0) / (g * g).sum(0).clamp(min=1e-12)
        feat = g * a
    elif tier == "per_sample":
        feat = t  # oracle: genuine perfectly corrected to teacher == staircase feature
    else:
        raise ValueError(tier)
    # run readout (last perceptron) on corrected features, then argmax.
    readout = flow.get_perceptrons()[-1]
    w = readout.layer.weight.detach().double()
    b = readout.layer.bias.detach().double() if readout.layer.bias is not None else 0.0
    # features are theta-normalized [0,1]; undo normalization of the last hidden layer
    theta_h = float(flow.get_perceptrons()[last_hidden].activation_scale)
    logits = feat.double() * theta_h @ w.t() + b
    return float(_accuracy(logits, yte[:n]))


# =========================================================================== #
# Q2c — GENERALIZATION: train vs test on the genuine cascade per FT method.
# =========================================================================== #
def report_q2c(seeds=(0, 1, 2), cells=((3, 8), (4, 8))):
    print("\n" + "=" * 92)
    print("Q2c  GENERALIZATION  (genuine-cascade train vs test acc per FT method)")
    print("=" * 92)
    print(f"{'d':>2}{'S':>4}{'method':>10}{'train':>8}{'test':>8}{'gen-gap':>9}")
    for depth, S in cells:
        for mode in ("full", "bias"):
            trs, tes = [], []
            for s in seeds:
                flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, s)
                genuine_ft(flow, xtr, ytr, S, mode=mode, epochs=40)
                with torch.no_grad():
                    trs.append(float(_accuracy(cascade_forward(flow, xtr, S), ytr)))
                    tes.append(float(_accuracy(cascade_forward(flow, xte, S), yte)))
            tr, te = float(np.mean(trs)), float(np.mean(tes))
            print(f"{depth:>2}{S:>4}{('N2' if mode=='bias' else 'full'):>10}"
                  f"{tr:>8.3f}{te:>8.3f}{tr-te:>+9.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("q1", "all"):
        report_q1()
    if which in ("why", "all"):
        report_why()
    if which in ("q2a", "all"):
        report_q2a()
    if which in ("q2b", "all"):
        report_q2b()
    if which in ("q2c", "all"):
        report_q2c()
