"""PHASE 3 KEY PATH — LINEARIZE the effective decode by warping the ENCODE.

THE PROBLEM (post death-cascade residual). Phase 1/2 showed the cascaded
single-spike TTFS "death cascade" is a correctable per-DEPTH gain distortion (the
gain trim G, ``ttfs_gain_correction``, revives it cold; the teacher-blend genuine
ramp revives it on the real mmixcore to ~0.93). AFTER that, the genuine cascade
still caps ~3.5pp below LIF (~0.94 vs ~0.975). This file attacks that RESIDUAL.

THE MECHANISM (bit-exact, see char_decode_law / capacity). A consumer
double-integrates its arriving latched ramps. A SINGLE upstream spike at
consumer-local cycle ``tau`` (unit weight) drives the consumer membrane at the end
of the window to a TRIANGULAR number:

    R(tau) = sum_{j=1}^{k} j = k(k+1)/2,    k = S - tau      (remaining window)

so the per-spike membrane DRIVE is QUADRATIC in the remaining window k, whereas the
faithful (teacher / ideal-staircase) decode wants the value to enter LINEARLY:
the spike was emitted to carry  v = (S - tau)/S = k/S  (uniform TTFS encode
``tau = round(S(1-v))``). Normalising both by their value at tau=0 (k=S):

    R_norm(tau) = R(tau)/R(0) = k(k+1) / (S(S+1))           (deployed, quadratic)
    L_norm(tau) = v           = k / S                        (intended, linear)
    g_eff(tau)  = R_norm/L_norm = (S - tau + 1)/(S + 1)      (the per-spike gain)

A consumer summing weighted inputs computes  sum_i w_i * R_norm(tau_i)  but the
teacher wants  sum_i w_i * v_i. A STATIC per-neuron theta can only rescale the SUM;
it cannot invert a per-input (per-sample) nonlinearity in tau. That residual
nonlinearity — NOT capacity (timing carries log2(S+1) bits, same as rate; the ideal
staircase already hits LIF level) — is what caps the genuine cascade.

THE FIX (this file). The ENCODE is free: we choose the fire time tau = phi(v) as a
function of the value v. DERIVE phi so the deployed quadratic DRIVE is LINEAR in v
(matches the staircase): set  R_norm(phi(v)) = v, i.e.

    k(k+1) / (S(S+1)) = v   =>   k = phi_k(v) = (-1 + sqrt(1 + 4 v S (S+1))) / 2
    tau = phi(v) = round(S - phi_k(v))                          (LINEARIZING ENCODE)

Equivalently a value PRE-WARP  v -> w(v) = phi_k(v)/S  fed to the STANDARD encode
``tau = round(S(1 - w))`` lands the same fire time. So the linearizing encode is a
per-neuron MONOTONE value map (or, in the standard-encode view, a value pre-warp).

DEPLOYABILITY (the load-bearing distinction this file establishes):
  * The INPUT / encoding layer (segment value-domain entry, ``encoding=True`` node:
    charge V/theta, fire at round(S(1 - V/theta))) admits an ARBITRARY monotone
    value map baked into calibration -> phi is FULLY deployable there (it is host
    preprocessing; free).
  * INTERNAL cascade neurons emit their spike via the ramp+threshold; the ONLY
    per-neuron knobs are theta (activation_scale) and bias. theta gives an AFFINE
    value warp  v -> v/g  before the standard encode — exactly the gain trim G — NOT
    an arbitrary phi. So a single-spike NONLINEAR phi is NOT internally deployable on
    the cascade; the best deployable single-spike encode warp is affine (== G).
  * The exact per-neuron linearization that IS deployable internally is the
    DUAL-SPIKE code: two timed spikes per neuron whose two triangular ramp drives
    SUM to a (near-)linear decode. Cost: 2 spikes/neuron (2x traffic) and a node
    that can fire twice.

Run:  source env/bin/activate
      python docs/.../experiments/linearize.py          # math + cold cascade
      python docs/.../experiments/linearize.py ft        # + fine-tuning
Findings: ../31_decode_linearization.md
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cascade_lab  # noqa: F401  (sets sys.path to src/tests/repo)
from cascade_lab import _accuracy, cascade_forward  # noqa: E402
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from closed_form import (  # noqa: E402  reuse the validated G machinery + flow cache
    C_RHO,
    apply_gain_correction,
    build_flow,
    g_relative,
    _restore_params,
)
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402

CHANCE = 0.10


# =========================================================================== #
# 0. The closed-form linearizing encode  phi(v)  and its inverse.
# =========================================================================== #
def ramp_drive_norm(tau, S):
    """Normalised quadratic membrane drive of a single spike at local cycle ``tau``.

    R_norm(tau) = k(k+1)/(S(S+1)), k = S - tau (clamped >=0). Bit-exact to
    capacity.ramp_effective_weight_distortion's R_ramp_norm column. This is what the
    deployed ramp decode ACTUALLY contributes per upstream spike.
    """
    k = np.clip(S - np.asarray(tau, dtype=np.float64), 0.0, None)
    return k * (k + 1.0) / (S * (S + 1.0))


def phi_k(v, S):
    """Linearizing remaining-window k = phi_k(v): solve k(k+1)/(S(S+1)) = v.

    k = (-1 + sqrt(1 + 4 v S (S+1)))/2. Monotone increasing in v over [0,1] -> [0,S].
    """
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    return (-1.0 + np.sqrt(1.0 + 4.0 * v * S * (S + 1.0))) / 2.0


def linearizing_encode_tau(v, S):
    """tau = phi(v) = round(S - phi_k(v)) — the per-neuron LINEARIZING fire time."""
    tau = np.round(S - phi_k(v, S)).astype(int)
    return np.clip(tau, 0, S)


def value_prewarp(v, S):
    """v -> w(v) = phi_k(v)/S: the value pre-warp that, fed to the STANDARD encode
    tau=round(S(1-w)), reproduces the linearizing fire time. This is the deployable
    form for the INPUT/encoding layer (a monotone value map baked into calibration).
    """
    return phi_k(v, S) / S


# =========================================================================== #
# 1. Single-spike: linearization vs the uniform encode (the encode/decode law).
# =========================================================================== #
def single_spike_law(Ss=(4, 8, 16, 32)):
    """Encode->ramp-decode round-trip error of the linearizing vs uniform encode.

    UNIFORM:  tau=round(S(1-v)); deployed drive R_norm(tau) is QUADRATIC -> the
              decoded drive is NOT v (g_eff(tau) gain). std/max of (R_norm - v).
    LINEARIZED: tau=phi(v); R_norm(phi(v)) ~= v by construction, BUT the integer-tau
              grid is now NON-UNIFORM (dense near k=S, sparse near k=0), so it trades
              decode-linearity for COARSER quantization where R is flat. We report
              BOTH errors so the tradeoff is explicit: linearization removes the
              SYSTEMATIC (bias-like, shared-across-neurons) gain distortion at the
              cost of EXTRA (zero-mean) quantization noise.
    """
    rows = []
    v = np.linspace(0.0, 1.0, 100001)
    for S in Ss:
        tau_u = np.round(S * (1.0 - v)).astype(int)
        drive_u = ramp_drive_norm(tau_u, S)          # what the consumer actually gets
        err_u = drive_u - v                          # vs intended linear v
        tau_l = linearizing_encode_tau(v, S)
        drive_l = ramp_drive_norm(tau_l, S)
        err_l = drive_l - v
        rows.append({
            "S": S,
            "uniform_drive_bias": round(float(err_u.mean()), 5),
            "uniform_drive_std": round(float(err_u.std()), 5),
            "uniform_drive_max": round(float(np.abs(err_u).max()), 5),
            "linz_drive_bias": round(float(err_l.mean()), 5),
            "linz_drive_std": round(float(err_l.std()), 5),
            "linz_drive_max": round(float(np.abs(err_l).max()), 5),
        })
    return rows


# =========================================================================== #
# 2. Dual-spike: two timed spikes whose summed triangular drives are LINEAR.
# =========================================================================== #
def _dual_codebook(S):
    """All achievable (normalised) summed drives of TWO spikes at integer cycles.

    Two spikes at local cycles -> remaining windows k1,k2 in {0..S}; summed drive
    Drive(k1,k2) = T(k1)+T(k2), T(k)=k(k+1)/2; normalised by the max 2*T(S). Returns
    sorted unique (drive_norm, k1, k2). Both triangulars are convex/positive, so the
    sum is still convex in each k — but the JOINT codebook has ~ (S+1)^2/2 levels and
    a far denser, more uniform value coverage than the single-spike S+1 levels.
    """
    Tk = np.array([k * (k + 1) / 2.0 for k in range(S + 1)], dtype=np.float64)
    dmax = 2.0 * Tk[S]
    entries = {}
    for k1 in range(S + 1):
        for k2 in range(k1, S + 1):           # k1<=k2 (spikes unordered)
            d = (Tk[k1] + Tk[k2]) / dmax
            key = round(d, 9)
            if key not in entries:
                entries[key] = (d, k1, k2)
    out = sorted(entries.values())
    return out


def dual_spike_encode(v, S, codebook=None):
    """Pick the (k1,k2) two-spike codeword whose summed drive is nearest to v.

    Returns the achieved normalised drive (the dual-spike decoded value). This is the
    exact per-neuron linearization the cascade CANNOT do with one spike; the cost is
    2 spikes/neuron and a twice-firing node.
    """
    cb = codebook if codebook is not None else _dual_codebook(S)
    drives = np.array([c[0] for c in cb])
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    idx = np.abs(drives[None, :] - v[..., None]).argmin(axis=-1)
    return drives[idx]


def dual_spike_law(Ss=(4, 8, 16, 32)):
    """Best linear-decode error of the dual-spike code vs the single-spike codes."""
    rows = []
    v = np.linspace(0.0, 1.0, 4001)
    for S in Ss:
        cb = _dual_codebook(S)
        d_dual = dual_spike_encode(v, S, cb)
        err_d = d_dual - v
        # single-spike best-achievable (linearizing) for reference
        d_lin = ramp_drive_norm(linearizing_encode_tau(v, S), S)
        err_l = d_lin - v
        rows.append({
            "S": S,
            "single_levels": S + 1,
            "dual_levels": len(cb),
            "single_linz_max": round(float(np.abs(err_l).max()), 5),
            "single_linz_std": round(float(err_l.std()), 5),
            "dual_max": round(float(np.abs(err_d).max()), 5),
            "dual_std": round(float(err_d.std()), 5),
        })
    return rows


# =========================================================================== #
# 3. Cascade prototype — the INPUT/encoding-layer value pre-warp (deployable).
# =========================================================================== #
# The encoding node fires at round(S(1 - V/theta)). Pre-warping the model INPUT
# x -> warp(x) (a monotone value map) before it enters the encoder reshapes the
# encode-layer fire times to the linearizing grid for layer-0's downstream drive.
# This is the only place a NONLINEAR single-spike phi is deployable (host value
# domain). We measure how much of the genuine->staircase residual it recovers.
def _staircase_acc(flow, x, y):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        return float(_accuracy(flow(x.double()), y))


def evaluate_layer_scope(*, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0):
    """How much does a NONLINEAR phi recover when applied at a RESTRICTED set of
    layers (which is the deployability question for single-spike)?

    Reuses the layerwise-snap (faithful per-neuron-code proxy on a single segment).
    Compares:
      * ``uniform``     — genuine: every layer's per-spike value is R_norm(round(.)).
      * ``phi_enc``     — DEPLOYABLE single-spike: nonlinear phi ONLY at the ENCODE
                          layer (depth 0, an arbitrary monotone value map at the
                          value-domain entry); internal layers stay uniform (their
                          only deployable knob is affine theta = G, applied here).
      * ``phi_enc_G``   — phi at encode layer + the affine G trim on internal layers.
      * ``phi_all``     — ORACLE: nonlinear phi at EVERY layer (NOT internally
                          deployable for single-spike — bounds what a per-neuron phi
                          could do; == the ideal staircase).
    The phi_all - phi_enc_G gap = the part of the residual a single-spike encode
    CANNOT reach deployably (because internal neurons can only warp affinely).
    """
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)
    _restore_params(flow, base_scales, base_weights, base_biases)
    install_ttfs_nodes(flow, S)
    stair = _staircase_acc(flow, xte, yte)

    n_layers = len(flow.get_perceptrons())

    def snap_uniform(v):
        return ramp_drive_norm(np.round(S * (1.0 - v)).astype(int), S)

    def snap_linz(v):
        return ramp_drive_norm(linearizing_encode_tau(v, S), S)

    # per-layer snap selectors
    def scoped(linz_layers, g_layers):
        # g_layers: apply the affine G value-warp v -> clamp(v / g_relative) before
        # the uniform snap (the deployable internal affine encode-warp).
        per_layer = []
        for d in range(n_layers):
            if d in linz_layers:
                per_layer.append(("linz", None))
            elif d in g_layers:
                per_layer.append(("g", g_relative(S, d, C_RHO)))
            else:
                per_layer.append(("uniform", None))
        return per_layer

    def run(per_layer):
        _restore_params(flow, base_scales, base_weights, base_biases)
        return round(_layerwise_decode_acc_scoped(flow, xte, yte, S, per_layer,
                                                  snap_uniform, snap_linz), 4)

    enc_only = scoped({0}, set())
    enc_plus_g = scoped({0}, set(range(1, n_layers)))
    all_linz = scoped(set(range(n_layers)), set())

    out = {"cont": round(cont_acc, 4), "ideal_staircase": round(stair, 4),
           "uniform": run([("uniform", None)] * n_layers),
           "phi_enc": run(enc_only),
           "phi_enc_G": run(enc_plus_g),
           "phi_all": run(all_linz)}
    _restore_params(flow, base_scales, base_weights, base_biases)
    return out


def _layerwise_decode_acc_scoped(flow, x, y, S, per_layer, snap_uniform, snap_linz):
    """Layerwise value-snap with a PER-LAYER decode policy.

    per_layer[d] in {("uniform",None), ("linz",None), ("g", g)}:
      uniform -> R_norm(round(S(1-v)))           (genuine per-spike value)
      linz    -> R_norm(phi(v)) (nonlinear encode value map at this layer)
      g       -> R_norm(round(S(1-clamp(v/g))))  (affine theta=G encode warp)
    """
    percs = flow.get_perceptrons()
    handles = []
    for d, p in enumerate(percs):
        scale = float(p.activation_scale)
        kind, param = per_layer[d] if d < len(per_layer) else ("uniform", None)

        def hook(_m, _i, out, scale=scale, kind=kind, param=param):
            r = (torch.relu(out) / max(scale, 1e-12)).clamp(0.0, 1.0).detach().cpu().numpy()
            if kind == "linz":
                dec = snap_linz(r)
            elif kind == "g":
                warped = np.clip(r / max(param, 1e-6), 0.0, 1.0)
                dec = snap_uniform(warped)
            else:
                dec = snap_uniform(r)
            snapped = torch.as_tensor(dec, dtype=out.dtype, device=out.device)
            return snapped * scale
        handles.append(p.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        logits = flow(x.double())
    for h in handles:
        h.remove()
    return float(_accuracy(logits, y))


# =========================================================================== #
# 4. The IDEAL nonlinear single-spike encode applied at EVERY layer (oracle).
# =========================================================================== #
# Not internally deployable (internal neurons only have affine theta/bias), but it
# bounds what a per-neuron NONLINEAR phi could recover if the hardware allowed it.
# We emulate it by replacing the genuine per-neuron drive R_norm(tau) with v
# directly = the IDEAL STAIRCASE (capacity.optimal_vs_genuine), so the oracle for a
# perfectly-linearized single-spike cascade IS the ideal staircase number.
def ideal_nonlinear_oracle(*, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0):
    """The staircase == a per-neuron-NONLINEAR-phi cascade with perfect linearization
    (each neuron's decode is v, no ramp distortion). The deployable ceiling for the
    linearize-the-encode idea (single-spike) on EVERY layer."""
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)
    _restore_params(flow, base_scales, base_weights, base_biases)
    install_ttfs_nodes(flow, S)
    stair = _staircase_acc(flow, xte, yte)
    return {"cont": round(cont_acc, 4), "ideal_staircase": round(stair, 4)}


# =========================================================================== #
# 5. Dual-spike cascade emulation — per-neuron exact linearization end to end.
# =========================================================================== #
# The genuine cascade is single-spike and bit-exact; we cannot install a 2-spike
# node into TTFSSegmentForward without changing the deployed forward. To measure the
# CEILING of the dual-spike code on the cascade we emulate a layer transfer where
# each neuron's decoded value is snapped to the nearest DUAL-spike codeword (linear
# decode), then propagated. This isolates "what does per-neuron exact linearization
# buy end-to-end" from the single-spike grid coarseness.
def proxy_faithfulness(*, depth=3, width=64, in_dim=64, n_classes=10, seed=0):
    """The uniform layerwise-snap proxy isolates the per-neuron DECODE-LINEARITY
    effect; it deliberately OMITS the cross-layer latency/window-shortening of the
    death cascade (the SEPARATE effect G handles). Validation: at high S (no death
    cascade) the proxy ~= the genuine cascade; at low S the genuine is lower (the
    extra death-cascade gap). Confirms the proxy's scope."""
    from closed_form import build_flow as _bf, _restore_params as _rp
    rows = []
    for S in (8, 16, 32):
        flow, calib, xtr, ytr, xte, yte, cont, bs, bw, bb = _bf(
            depth=depth, width=width, in_dim=in_dim, n_classes=n_classes, seed=seed)
        _rp(flow, bs, bw, bb)
        install_ttfs_nodes(flow, S)
        gen = float(_accuracy(cascade_forward(flow, xte, S), yte))
        _rp(flow, bs, bw, bb)
        proxy = _layerwise_decode_acc(
            flow, xte, yte, S,
            lambda v: ramp_drive_norm(np.round(S * (1 - v)).astype(int), S))
        rows.append({"S": S, "genuine_cascade": round(gen, 4),
                     "uniform_snap_proxy": round(proxy, 4)})
    return rows


def _layerwise_decode_acc(flow, x, y, S, snap_fn):
    """Forward the converted flow in the value domain, but after each ReLU
    perceptron snap the activation to ``snap_fn(v_norm)`` (the code's decoded value)
    and re-scale. snap_fn maps normalised [0,1] -> achieved decoded [0,1]."""
    percs = flow.get_perceptrons()
    handles = []
    for p in percs:
        scale = float(p.activation_scale)

        def hook(_m, _i, out, scale=scale):
            r = (torch.relu(out) / max(scale, 1e-12)).clamp(0.0, 1.0)
            snapped = torch.as_tensor(snap_fn(r.detach().cpu().numpy()),
                                      dtype=out.dtype, device=out.device)
            return snapped * scale
        handles.append(p.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        logits = flow(x.double())
    for h in handles:
        h.remove()
    return float(_accuracy(logits, y))


def evaluate_dual_spike(*, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0):
    """End-to-end accuracy of {uniform single-spike decode, linearizing single-spike,
    dual-spike} as LAYERWISE value snaps vs the staircase. The snaps decode:
      uniform  -> R_norm(round(S(1-v)))     (the genuine cascade's per-spike value)
      linz     -> R_norm(phi(v)) ~= v       (single-spike linearized; grid-limited)
      dual     -> nearest dual codeword     (two-spike exact-ish linear)
    All single-segment, so layerwise-snap is a faithful proxy for the per-neuron code
    transfer (the cascade's cross-layer latency is the SEPARATE death-cascade effect,
    handled by G; this isolates the DECODE-LINEARITY residual)."""
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)
    _restore_params(flow, base_scales, base_weights, base_biases)
    install_ttfs_nodes(flow, S)
    stair = _staircase_acc(flow, xte, yte)
    cb = _dual_codebook(S)

    def snap_uniform(v):
        return ramp_drive_norm(np.round(S * (1.0 - v)).astype(int), S)

    def snap_linz(v):
        return ramp_drive_norm(linearizing_encode_tau(v, S), S)

    def snap_dual(v):
        return dual_spike_encode(v, S, cb)

    out = {"cont": round(cont_acc, 4), "ideal_staircase": round(stair, 4)}
    for name, fn in (("uniform", snap_uniform), ("linz_single", snap_linz),
                     ("dual_spike", snap_dual)):
        _restore_params(flow, base_scales, base_weights, base_biases)
        out[name] = round(_layerwise_decode_acc(flow, xte, yte, S, fn), 4)
    _restore_params(flow, base_scales, base_weights, base_biases)
    return out


# =========================================================================== #
# 6. WITH genuine fine-tuning: how close does the BEST DEPLOYABLE recipe
#    (G init + genuine STE fine-tune) get to the linearized-decode ceilings?
# =========================================================================== #
def _genuine_ft(flow, xtr, ytr, S, *, ft_epochs=40, ft_lr=2e-3, surrogate_temp=0.5):
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=ft_lr)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    for _ in range(ft_epochs):
        opt.zero_grad()
        logits = cascade_forward(flow, xtr, S, grad=True, surrogate_temp=surrogate_temp)
        lossf(logits, ytr).backward()
        opt.step()


def evaluate_ft_vs_ceilings(*, depth=3, width=64, in_dim=64, n_classes=10, S=8,
                            seed=0, ft_epochs=40, ft_lr=2e-3):
    """Where does the deployable recipe land relative to the linearization ceilings?

    Reports, all on the SAME trained flow:
      * ``plainG_ft``  — G init + genuine STE fine-tune (the deployed single-spike
                         recipe; FT is the network's only way to *approximate* the
                         per-sample tau nonlinearity the static encode cannot invert).
      * ``staircase``  — ideal linear single-spike decode (== nonlinear-phi-everywhere
                         oracle); the linearization target.
      * ``dual_emul``  — dual-spike cascade ceiling (per-neuron exact linearization
                         via 2 timed spikes); the deployable-with-2-spike ceiling.
    The gaps say how much of the genuine->staircase residual genuine FT ALREADY
    closes, and what a code change (dual-spike) would add beyond it.
    """
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)
    _restore_params(flow, base_scales, base_weights, base_biases)
    install_ttfs_nodes(flow, S)
    stair = _staircase_acc(flow, xte, yte)

    cb = _dual_codebook(S)
    _restore_params(flow, base_scales, base_weights, base_biases)
    dual = round(_layerwise_decode_acc(flow, xte, yte, S,
                 lambda v: dual_spike_encode(v, S, cb)), 4)

    _restore_params(flow, base_scales, base_weights, base_biases)
    install_ttfs_nodes(flow, S)
    gen_cold = float(_accuracy(cascade_forward(flow, xte, S), yte))

    _restore_params(flow, base_scales, base_weights, base_biases)
    apply_gain_correction(flow, base_scales, S, rule="relative", c=C_RHO)
    _genuine_ft(flow, xtr, ytr, S, ft_epochs=ft_epochs, ft_lr=ft_lr)
    plain_ft = float(_accuracy(cascade_forward(flow, xte, S), yte))
    _restore_params(flow, base_scales, base_weights, base_biases)

    return {"cont": round(cont_acc, 4), "gen_cold": round(gen_cold, 4),
            "plainG_ft": round(plain_ft, 4), "staircase": round(stair, 4),
            "dual_emul": dual}


# =========================================================================== #
# Reports
# =========================================================================== #
def report_law():
    print("=" * 80)
    print("0. THE LINEARIZING ENCODE  phi(v): R_norm(phi(v)) = v")
    print("=" * 80)
    print("   R_norm(tau)=k(k+1)/(S(S+1)), k=S-tau   [deployed quadratic drive]")
    print("   phi_k(v)=(-1+sqrt(1+4 v S(S+1)))/2     [linearizing remaining-window]")
    print("   value pre-warp  w(v)=phi_k(v)/S  fed to standard tau=round(S(1-w))")
    print(f"\n   {'S':>3} | uniform tau for v=.25/.5/.75 | phi(v) tau | prewarp w(v) -> "
          "round(S(1-w)) tau  (must equal phi tau)")
    for S in (4, 8, 16, 32):
        vs = np.array([0.25, 0.5, 0.75])
        tu = np.round(S * (1 - vs)).astype(int)
        tl = linearizing_encode_tau(vs, S)
        # deployable form: warp the value, feed the STANDARD encoder -> same fire time.
        tw = np.round(S * (1.0 - value_prewarp(vs, S))).astype(int)
        ok = "OK" if np.array_equal(tl, tw) else "MISMATCH"
        print(f"   {S:>3} | {list(tu)} | {list(tl)} | {list(tw)} [{ok}]")


def report_single_dual():
    print("\n" + "=" * 80)
    print("1. SINGLE-SPIKE: linearizing vs uniform encode (decode-drive error)")
    print("=" * 80)
    print(f"   {'S':>3} | {'unif bias':>9} {'unif std':>9} {'unif max':>9} | "
          f"{'linz bias':>9} {'linz std':>9} {'linz max':>9}")
    for r in single_spike_law():
        print(f"   {r['S']:>3} | {r['uniform_drive_bias']:>9} {r['uniform_drive_std']:>9} "
              f"{r['uniform_drive_max']:>9} | {r['linz_drive_bias']:>9} "
              f"{r['linz_drive_std']:>9} {r['linz_drive_max']:>9}")
    print("\n   (uniform has a NEGATIVE systematic drive bias = the death-cascade gain;")
    print("    linz removes the bias but the non-uniform grid raises zero-mean noise)")

    print("\n" + "=" * 80)
    print("2. DUAL-SPIKE: two triangular drives summed -> near-linear decode")
    print("=" * 80)
    print(f"   {'S':>3} | {'#lvl 1sp':>8} {'#lvl 2sp':>8} | {'1sp linz max':>12} "
          f"{'1sp std':>8} | {'2sp max':>8} {'2sp std':>8}")
    for r in dual_spike_law():
        print(f"   {r['S']:>3} | {r['single_levels']:>8} {r['dual_levels']:>8} | "
              f"{r['single_linz_max']:>12} {r['single_linz_std']:>8} | "
              f"{r['dual_max']:>8} {r['dual_std']:>8}")


def _mean(fn, seeds, **kw):
    rs = [fn(seed=s, **kw) for s in seeds]
    return {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]}


def report_dual_cascade(seeds=(0, 1, 2)):
    print("\n" + "=" * 80)
    print("3a. PROXY FAITHFULNESS — uniform snap isolates DECODE-LINEARITY (no latency)")
    print("    high S (no death cascade): proxy ~= genuine; low S: genuine lower (the")
    print("    SEPARATE death-cascade latency gap, handled by G).")
    print("=" * 80)
    for r in proxy_faithfulness(seed=0):
        print(f"    S={r['S']:>3}: genuine_cascade={r['genuine_cascade']:.3f}  "
              f"uniform_snap_proxy={r['uniform_snap_proxy']:.3f}")
    print("\n" + "=" * 80)
    print("3b. END-TO-END layerwise decode snap (isolates DECODE-LINEARITY residual)")
    print("    uniform=genuine per-spike value | linz_single | dual_spike vs staircase")
    print("=" * 80)
    for depth in (3, 4):
        print(f"\n--- depth={depth} (mean over seeds {seeds}) ---")
        print(f"   {'S':>3} {'cont':>7} {'stair':>7} {'uniform':>8} {'linz_1sp':>9} "
              f"{'dual_2sp':>9} | {'dual/stair gap':>14}")
        for S in (4, 8, 16, 32):
            m = _mean(evaluate_dual_spike, seeds, depth=depth, S=S)
            gap = m["ideal_staircase"] - m["dual_spike"]
            print(f"   {S:>3} {m['cont']:>7.3f} {m['ideal_staircase']:>7.3f} "
                  f"{m['uniform']:>8.3f} {m['linz_single']:>9.3f} {m['dual_spike']:>9.3f} "
                  f"| {gap:>+13.3f}")


def report_layer_scope(seeds=(0, 1, 2)):
    print("\n" + "=" * 80)
    print("4. DEPLOYABILITY SCOPE — nonlinear phi at ENCODE layer only (deployable)")
    print("   vs phi everywhere (oracle: not internally deployable for single-spike).")
    print("   phi_enc_G = deployable single-spike best; phi_all = the staircase.")
    print("=" * 80)
    for depth in (3, 4):
        print(f"\n--- depth={depth} (mean over seeds {seeds}) ---")
        print(f"   {'S':>3} {'cont':>7} {'stair':>7} {'uniform':>8} {'phi_enc':>8} "
              f"{'phi_enc_G':>10} {'phi_all':>8}")
        for S in (4, 8, 16, 32):
            m = _mean(evaluate_layer_scope, seeds, depth=depth, S=S)
            print(f"   {S:>3} {m['cont']:>7.3f} {m['ideal_staircase']:>7.3f} "
                  f"{m['uniform']:>8.3f} {m['phi_enc']:>8.3f} {m['phi_enc_G']:>10.3f} "
                  f"{m['phi_all']:>8.3f}")


def report_ft(seeds=(0, 1, 2)):
    print("\n" + "=" * 80)
    print("5. GENUINE FINE-TUNE vs the LINEARIZATION CEILINGS (real deployable path)")
    print("   plainG_ft = G init + genuine STE FT; staircase = linearize target;")
    print("   dual_emul = dual-spike (2 spikes/neuron) ceiling.")
    print("=" * 80)
    for depth in (3, 4):
        for S in (8, 16):
            m = _mean(evaluate_ft_vs_ceilings, seeds, depth=depth, S=S)
            print(f"   depth={depth} S={S:>2}: cont={m['cont']:.3f} "
                  f"gen_cold={m['gen_cold']:.3f} plainG_ft={m['plainG_ft']:.3f} | "
                  f"staircase={m['staircase']:.3f} dual={m['dual_emul']:.3f} | "
                  f"FT->stair gap={m['staircase']-m['plainG_ft']:+.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cold"
    report_law()
    report_single_dual()
    report_dual_cascade()
    report_layer_scope()
    if which in ("ft", "all"):
        report_ft()
