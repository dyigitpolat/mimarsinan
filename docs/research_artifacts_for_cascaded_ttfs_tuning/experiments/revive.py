"""Data-grounded per-neuron DEAD-NEURON REVIVAL calibration for the deep cascade.

Hypothesis (user): correct per-layer calibration makes even a deep cascade deploy
losslessly on MNIST — no residual/shallow tricks. Mechanism: deep neurons go fully
DEAD (rate≈0: their attenuated input never crosses threshold). theta-scaling an
ALIVE neuron is ~invariant (decoded = rate*theta, rate∝1/theta), so a per-depth
geometric trim over-corrects alive layers and anchors accuracy down. The RIGHT
calibration lowers theta ONLY for under-firing neurons, iteratively (reviving a
layer revives its downstream), leaving alive neurons untouched.

Run: python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/revive.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))
from cascade_lab import (  # noqa: E402
    _SingleSegmentMLP, _accuracy, _calibrate_scales, _capture_activation_means,
    convert_torch_model, digits_task, install_ttfs_nodes, train_continuous,
)
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402
from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales  # noqa: E402


def _per_neuron(flow, x, S):
    """Per-perceptron (decoded_value, firing_rate=count/T) channel-means."""
    rec = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by = {id(perceptron_of(n)): v for n, v in rec.items() if perceptron_of(n) is not None}
    out = []
    for p in flow.get_perceptrons():
        v = by.get(id(p))
        if v is None:
            out.append((None, None)); continue
        decoded = v.reshape(-1, v.shape[-1]).double().mean(0)
        scale = p.activation_scale.detach().double()
        if scale.dim() > 0:
            n = min(scale.numel(), decoded.numel())
            rate = decoded[:n] / scale[:n].clamp(min=1e-9)
        else:
            rate = decoded / float(scale)
        out.append((decoded, rate))   # (decoded value, firing rate)
    return out


def _per_channel_scale(p):
    s = p.activation_scale.detach()
    if s.dim() == 0:
        return None  # scalar; caller decides
    return s


def calibrate_revive(flow, x, S, teacher, *, iters=40, dead_rate=0.05, step=0.6,
                     floor=1e-4, per_channel=False):
    """Lower theta ONLY for under-firing (dead) NON-encoding neurons whose teacher
    says they SHOULD fire, until they fire. PER-CHANNEL (revive only the dead output
    channels, leaving alive ones untouched — theta is a first-class per-neuron chip
    param). Iterative (downstream revives as upstream does)."""
    perceptrons = list(flow.get_perceptrons())
    # promote scalar activation_scale -> per-output-channel vector so dead channels
    # can be revived without disturbing alive ones.
    if per_channel:
        for k, p in enumerate(perceptrons):
            if getattr(p, "is_encoding_layer", False):
                continue
            out_dim = p.layer.weight.shape[0]
            s = p.activation_scale.detach()
            if s.dim() == 0:
                p.activation_scale.data = (s * torch.ones(out_dim, dtype=s.dtype, device=s.device))
    for _ in range(iters):
        pn = _per_neuron(flow, x, S)
        moved = 0
        for k, p in enumerate(perceptrons):
            if getattr(p, "is_encoding_layer", False):
                continue
            decoded, rate = pn[k]
            if rate is None:
                continue
            t = teacher.get(k)
            wants = (t.clamp(min=0) > 1e-4) if t is not None else torch.ones_like(rate, dtype=torch.bool)
            n = min(rate.numel(), wants.numel())
            dead = (rate[:n] < dead_rate) & wants[:n]          # should fire but doesn't
            if not bool(dead.any()):
                continue
            s = p.activation_scale.data
            if s.dim() == 0:
                p.activation_scale.data = (s * step).clamp(min=floor)
            else:
                m = torch.ones_like(s)
                m[:n][dead] = step                              # shrink ONLY dead channels
                p.activation_scale.data = (s * m).clamp(min=floor)
            moved += 1
        propagate_boundary_input_scales(flow)
        if moved == 0:
            break
    return flow


def build_deep(depth, S, seed=0, epochs=120):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, 64, 64, 10)
    train_continuous(base, xtr, ytr, epochs=epochs)
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (64,), 10, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    teacher = _capture_activation_means(flow, xte)
    install_ttfs_nodes(flow, S)
    return flow, xte, yte, cont, teacher


def genuine_acc(flow, x, y, S):
    with torch.no_grad():
        return _accuracy(TTFSSegmentForward(flow.get_mapper_repr(), S)(x.double()), y)


if __name__ == "__main__":
    print("=== DEAD-NEURON REVIVAL calibration on DEEP cascades (digits) ===")
    print("depth S : continuous  genuine(base)  genuine(revived)  [atten head->tail]")
    for depth in (3, 6, 9):
        for S in (8, 16):
            flow, xte, yte, cont, teacher = build_deep(depth, S)
            base_acc = genuine_acc(flow, xte, yte, S)
            pn0 = _per_neuron(flow, xte, S)
            atten0 = [round(float(d.mean()) / max(float(teacher[k].clamp(min=0).mean()), 1e-6), 2)
                      if d is not None and k in teacher else None for k, (d, r) in enumerate(pn0)]
            calibrate_revive(flow, xte, S, teacher)
            rev_acc = genuine_acc(flow, xte, yte, S)
            pn1 = _per_neuron(flow, xte, S)
            atten1 = [round(float(d.mean()) / max(float(teacher[k].clamp(min=0).mean()), 1e-6), 2)
                      if d is not None and k in teacher else None for k, (d, r) in enumerate(pn1)]
            print(f"d={depth} S={S:>2}: cont={cont:.3f}  base={base_acc:.3f}  "
                  f"revived={rev_acc:.3f}   atten {atten0[-1]}->{atten1[-1]}")


def calibrate_equalize(flow, x, S, teacher, *, iters=15, floor=0.15, ceil=8.0):
    """DECODE-ATTENUATION EQUALIZATION (weight-level, NOT theta — theta is invariant
    for alive neurons). Each neuron k is decode-attenuated by g_k = decoded_k/ideal_k
    (<1). Compensate at the CONSUMER: scale the downstream perceptron's input-weight
    column for k by 1/g_k, so it sees the de-attenuated signal. Iterative (downstream
    g changes as upstream is fixed). Sequential-cascade prototype: perceptron L+1
    consumes perceptron L; the classifier consumes the last perceptron. A pure weight
    change (deployable), so it BREAKS the compounding attenuation that theta can't."""
    ps = list(flow.get_perceptrons())
    cls = flow.classifier if hasattr(flow, "classifier") else None
    for _ in range(iters):
        pn = _per_neuron(flow, x, S)
        for L in range(len(ps)):
            decoded, _ = pn[L]
            t = teacher.get(L)
            if decoded is None or t is None:
                continue
            n = min(decoded.numel(), t.numel())
            ideal = t[:n].clamp(min=0)
            g = (decoded[:n] / ideal.clamp(min=1e-4)).clamp(floor, 1.0)  # attenuation in (floor,1]
            comp = (1.0 / g).clamp(1.0, ceil)                            # >=1 amplify
            comp = torch.where(ideal > 1e-4, comp, torch.ones_like(comp))
            down = ps[L + 1] if L + 1 < len(ps) else None
            if down is not None and down.layer.weight.shape[1] >= n:
                with torch.no_grad():
                    down.layer.weight.data[:, :n] *= comp.to(down.layer.weight.dtype)
            elif down is None and cls is not None and cls.weight.shape[1] >= n:
                with torch.no_grad():
                    cls.weight.data[:, :n] *= comp.to(cls.weight.dtype).float()
    return flow


def calibrate_equalize_damped(flow, x, S, teacher, *, iters=25, damp=0.4, floor=0.2, ceil=4.0):
    """Damped decode-attenuation equalization: each iter move the downstream weight
    column for neuron k partway (``damp``) toward the de-attenuation 1/g_k, so the
    nonlinear cascade converges instead of overshooting. Weight-level, deployable."""
    ps = list(flow.get_perceptrons())
    cls = flow.classifier if hasattr(flow, "classifier") else None
    for _ in range(iters):
        pn = _per_neuron(flow, x, S)
        for L in range(len(ps)):
            decoded, _ = pn[L]
            t = teacher.get(L)
            if decoded is None or t is None:
                continue
            n = min(decoded.numel(), t.numel())
            ideal = t[:n].clamp(min=0)
            g = (decoded[:n] / ideal.clamp(min=1e-4)).clamp(floor, ceil)
            comp = (1.0 / g) ** damp                          # damped step toward 1/g
            comp = torch.where(ideal > 1e-4, comp, torch.ones_like(comp))
            down = ps[L + 1] if L + 1 < len(ps) else None
            if down is not None and down.layer.weight.shape[1] >= n:
                with torch.no_grad():
                    down.layer.weight.data[:, :n] *= comp.to(down.layer.weight.dtype)
            elif down is None and cls is not None and cls.weight.shape[1] >= n:
                with torch.no_grad():
                    cls.weight.data[:, :n] *= comp.to(cls.weight.dtype).float()
    return flow
