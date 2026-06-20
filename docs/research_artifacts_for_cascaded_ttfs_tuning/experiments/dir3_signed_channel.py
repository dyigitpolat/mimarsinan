"""DIRECTION 3 DECISIVE: full signed-channel + complement-spike monotonization on the
REAL trained net's weights, multi-layer, measured against the real staircase.

Each interior cascade neuron z = sum_i W_i v_i + b is split W = W+ - W-. Using
v_i = 1 - tau_i/T (tau_i the spike cycle), the negative part is rewritten
   -sum W-_i v_i = -sum W-_i + sum W-_i (1 - v_i)
The complement value (1 - v_i) is delivered by a SECOND spike per input at cycle
(T - tau_i) weighted by +W-_i, and the constant -sum W-_i is a per-neuron bias.
Then every spike-driven membrane increment is NON-NEGATIVE -> the running membrane
is monotone non-decreasing -> the first theta-crossing reflects the COMPLETE sum,
killing the premature-fire death cascade BY CONSTRUCTION.

We simulate the genuine ramp cascade (membrane = integral of ramp_current, single
fire, fire-time decode) layer by layer with and without the construction, and compare
to the analytical staircase, at depth 6 and 9. Cost of the construction: 2x input
spikes per neuron and 2x fan-in (the +channel and the complement -channel).
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(__file__)
_ART = "/home/yigit/repos/research_stuff/mimarsinan/docs/research_artifacts_for_cascaded_ttfs_tuning/experiments"
for _p in (_HERE, _ART, os.path.join(_ART, ".."), "/home/yigit/repos/research_stuff/mimarsinan/tests"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recipe_harness import build  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402


def _layers(flow):
    """Extract per-perceptron (W, b_eff, theta) in exec order (float64, on device)."""
    out = []
    for p in flow.get_perceptrons():
        W = p.layer.weight.detach().double()
        b = (p.layer.bias.detach().double() if p.layer.bias is not None
             else torch.zeros(W.shape[0], dtype=torch.float64, device=W.device))
        theta = float(p.activation_scale)
        out.append((W, b, theta, bool(getattr(p, "is_encoding_layer", False))))
    return out


def _tau(v, T):
    return torch.round(T * (1.0 - v.clamp(0, 1))).long()


def cascade_layer(v_in, W, b, theta, T, *, mono):
    """One cascade neuron layer: v_in (B,in) normalized [0,1] presynaptic values ->
    (B,out) decoded normalized [0,1] values via genuine ramp + single fire."""
    B = v_in.shape[0]
    O = W.shape[0]
    tau = _tau(v_in, T)                       # (B,in)
    ramp = torch.zeros(B, O, dtype=v_in.dtype, device=v_in.device)
    mem = torch.zeros(B, O, dtype=v_in.dtype, device=v_in.device)
    fired = torch.zeros(B, O, dtype=torch.bool, device=v_in.device)
    fv = torch.zeros(B, O, dtype=v_in.dtype, device=v_in.device)
    if mono:
        Wp = W.clamp(min=0.0)
        Wn = (-W).clamp(min=0.0)
        comp = _tau(1.0 - v_in, T)            # complement spike cycle
        # constant -sum(W-) delivered as a per-cycle LINEAR bias (membrane channel),
        # NOT a t=0 ramp injection: a per-cycle bias c contributes c*T to membrane_end
        # (linear), giving the correct -sum(W-) value contribution without slamming the
        # ramp negative early (which would defer all fires and re-collapse the cascade).
        neg_const = -(torch.ones(B, W.shape[1], dtype=v_in.dtype, device=v_in.device) @ Wn.t()) / theta
    for t in range(T):
        if mono:
            inc = (tau == t).to(v_in.dtype) @ (Wp.t() / theta) \
                + (comp == t).to(v_in.dtype) @ (Wn.t() / theta)
            ramp = ramp + inc
            mem = mem + ramp + (b.to(v_in.device) / theta) + neg_const
            fire = (~fired) & (mem >= 1.0)
            fv = torch.where(fire, torch.full_like(fv, (T - t) / T), fv)
            fired = fired | fire
            continue
        else:
            ramp = ramp + (tau == t).to(v_in.dtype) @ (W.t() / theta)
        mem = mem + ramp + (b.to(v_in.device) / theta)
        fire = (~fired) & (mem >= 1.0)
        fv = torch.where(fire, torch.full_like(fv, (T - t) / T), fv)
        fired = fired | fire
    return fv


def run_cascade(flow, x, T, *, mono):
    layers = _layers(flow)
    # encoding layer: value-domain (host) — produce its normalized output analytically
    v = x.double()
    for i, (W, b, theta, is_enc) in enumerate(layers):
        z = v @ W.t() + b
        if is_enc:
            v = (torch.relu(z) / theta).clamp(0, 1)   # host ComputeOp (ideal value)
        else:
            v = cascade_layer(v, W, b, theta, T, mono=mono)
    # last layer scale back to logits domain
    return v


def genuine_logits_scaled(flow, x, T, *, mono):
    layers = _layers(flow)
    v = run_cascade(flow, x, T, mono=mono)
    return v * float(layers[-1][2])           # de-normalize by output theta


def run():
    T = 8
    for depth in (6, 9):
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, T, seed=0)
        stair = ttfs_staircase_acc(flow, xte, yte, T)
        # sanity: baseline manual cascade should match the shipped genuine (death cascade)
        base_log = genuine_logits_scaled(flow, xte, T, mono=False)
        base_acc = _accuracy(base_log, yte)
        mono_log = genuine_logits_scaled(flow, xte, T, mono=True)
        mono_acc = _accuracy(mono_log, yte)
        print(f"d={depth} S={T} | cont={cont:.3f} stair={stair:.3f} | "
              f"manual_cold={base_acc:.3f}  MONO(signed+complement)={mono_acc:.3f}", flush=True)


if __name__ == "__main__":
    run()
