"""THE TEST of the user's thesis: "calibrate correctly -> deep cascade deploys
losslessly, no tricks (residuals/shallow)".

mechanism.py proved a STATIC per-layer theta-trim lifts d=3 cascaded genuine from
0.074 (chance) to 0.909 == the staircase. So the collapse is a CORRECTABLE GAIN,
not lost information. The oracle brute-force does not scale past d~4, and prior
heuristic calibrations capped at ~0.41 -- the CEILING is high but the calibration
ALGORITHM did not reach it.

The genuine cascade (TTFSSegmentForward) is DIFFERENTIABLE. So calibrate per-neuron
theta by GRADIENT through the genuine cascade with the WEIGHTS FROZEN -- a
calibration-only tune (no weight learning). If theta-alone recovers deep cascades
to the staircase, the user is right and we have the deployable recipe.

Compare three budgets on the same deep flow:
  cold genuine | theta-only calib (weights frozen) | full FT (weights too)

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/theta_calibrate.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from ft_budget import DEV, build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402


def _promote_theta_per_channel(flow):
    """Make each perceptron's activation_scale a per-output-channel trainable vector
    (so individual dead neurons can be revived). Encoding layer kept fixed (its scale
    is pinned by the input-encoding contract)."""
    thetas = []
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        out_dim = p.layer.weight.shape[0]
        s = p.activation_scale.detach()
        vec = (s * torch.ones(out_dim, dtype=s.dtype, device=s.device)
               if s.dim() == 0 else s.clone())
        param = nn.Parameter(vec, requires_grad=True)
        p.set_activation_scale(param)
        thetas.append(param)
    return thetas


def _freeze_weights(flow):
    for p in flow.get_perceptrons():
        p.layer.weight.requires_grad_(False)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(False)


def calibrate_theta(flow, xtr, ytr, S, *, steps=600, lr=5e-2, bs=256, seed=0):
    """Gradient calibration of per-neuron theta through the genuine cascade,
    WEIGHTS FROZEN. theta must stay positive -> optimize log-theta."""
    _freeze_weights(flow)
    thetas = _promote_theta_per_channel(flow)
    # reinstall nodes so TTFSActivation picks up the new per-channel scale params
    from cascade_fixtures import install_ttfs_nodes
    install_ttfs_nodes(flow, S)
    # the install copies activation_scale by reference, so the live params are the
    # perceptron scales; recollect them as the optimisation targets.
    logs = []
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        s = p.activation_scale
        if not isinstance(s, nn.Parameter):
            s = nn.Parameter(s.detach().clone(), requires_grad=True)
            p.set_activation_scale(s)
        s.requires_grad_(True)
        logs.append(s)
    opt = torch.optim.Adam(logs, lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    for _ in range(steps):
        idx = torch.randint(0, xtr.shape[0], (bs,), generator=g).to(xtr.device)
        x, y = xtr[idx].double(), ytr[idx]
        logits = drv(x)
        opt.zero_grad(); lossf(logits, y).backward(); opt.step()
        with torch.no_grad():
            for s in logs:
                s.clamp_(min=1e-3)
    return flow


def full_ft(flow, xtr, ytr, S, *, steps=600, lr=2e-3, bs=256, seed=0):
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    for _ in range(steps):
        idx = torch.randint(0, xtr.shape[0], (bs,), generator=g).to(xtr.device)
        x, y = xtr[idx].double(), ytr[idx]
        logits = drv(x)
        opt.zero_grad(); lossf(logits, y).backward(); opt.step()
    return flow


if __name__ == "__main__":
    STEPS = 600
    print(f"=== theta-only calibration vs full FT (device={DEV}, steps={STEPS}) ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'stair':>6} {'cold':>6} "
          f"{'theta-calib':>11} {'full-FT':>8}")
    for depth in (3, 6, 9):
        for S in (8, 16):
            flow, xtr, ytr, xte, yte, cont, _t, _b = build(depth, S)
            stair = ttfs_staircase_acc(flow, xte, yte, S)
            cold = genuine_acc(flow, xte, yte, S)
            fth = calibrate_theta(copy.deepcopy(flow), xtr, ytr, S, steps=STEPS)
            tca = genuine_acc(fth, xte, yte, S)
            ff = full_ft(copy.deepcopy(flow), xtr, ytr, S, steps=STEPS)
            ffa = genuine_acc(ff, xte, yte, S)
            print(f"{depth:>2} {S:>3} | {cont:>6.3f} {stair:>6.3f} {cold:>6.3f} "
                  f"{tca:>11.3f} {ffa:>8.3f}")
