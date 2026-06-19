"""How much genuine fine-tuning does weight-equalization need to recover full
accuracy on a DEEP cascade? Accuracy-vs-FT-budget for {baseline init, equalized
init}. The equalized init fixes the MEAN attenuation; FT then learns the
per-sample-robust weights. Question: does equalized+FT reach full accuracy with
LESS FT (and HIGHER) than baseline+FT?

Uses a BatchNorm deep MLP so the continuous ANN actually trains at depth (BN folds
into the linear at conversion, so the deployed cascade is the plain deep cascade).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from cascade_lab import _accuracy, _calibrate_scales, _capture_activation_means, digits_task  # noqa: E402
from revive import calibrate_equalize_damped, genuine_acc  # noqa: E402
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402


class _DeepMLPBN(nn.Module):
    """depth Linear+BN+ReLU stages — BN lets a deep MLP train; folds at conversion."""

    def __init__(self, depth, width, in_dim, out_dim):
        super().__init__()
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU())
            for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        return x


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _calibrate_cuda(flow, x):
    """Device-aware activation-scale calibration (the shared helper writes CPU
    tensors, which breaks a cuda flow)."""
    caps: dict = {}
    handles = [
        p.register_forward_hook(
            lambda _m, _i, o, p=p: caps.__setitem__(
                p, max(caps.get(p, 0.0), float(o.detach().abs().max())))
        )
        for p in flow.get_perceptrons()
    ]
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    for p in flow.get_perceptrons():
        p.set_activation_scale(
            torch.tensor(max(caps[p], 1e-3), dtype=torch.float64, device=x.device))


def build(depth, S, seed=0, epochs=200, lr=2e-3):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = (t.to(DEV) for t in digits_task(seed=seed + 1))
    base = _DeepMLPBN(depth, 96, 64, 10).to(DEV)
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    base.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(base(xtr.float()), ytr).backward(); opt.step()
    base.eval()
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (64,), 10, device=str(DEV))
    _calibrate_cuda(flow, xtr[:256])
    teacher = _capture_activation_means(flow, xte)
    install_ttfs_nodes(flow, S)
    return flow, xtr, ytr, xte, yte, cont, teacher, base


def ft_genuine(flow, xtr, ytr, S, steps, lr=2e-3, bs=256, seed=0):
    """Fine-tune through the GENUINE single-spike cascade (differentiable subsume)."""
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
    import copy
    DEPTH, S = 6, 8
    flow, xtr, ytr, xte, yte, cont, teacher, _base = build(DEPTH, S)
    base0 = genuine_acc(flow, xte, yte, S)
    eqflow = copy.deepcopy(flow)
    calibrate_equalize_damped(eqflow, xte, S, teacher)
    eq0 = genuine_acc(eqflow, xte, yte, S)
    print(f"=== FT-budget: depth={DEPTH} S={S}, continuous={cont:.3f} ===")
    print(f"cold: baseline={base0:.3f}  equalized={eq0:.3f}")
    print(f"{'FT_steps':>9} {'baseline+FT':>12} {'equalized+FT':>13}")
    prev = 0
    bflow, eflow = copy.deepcopy(flow), copy.deepcopy(eqflow)
    for steps in (50, 100, 200, 400, 800, 1600):
        ft_genuine(bflow, xtr, ytr, S, steps - prev)
        ft_genuine(eflow, xtr, ytr, S, steps - prev)
        prev = steps
        print(f"{steps:>9} {genuine_acc(bflow, xte, yte, S):>12.3f} "
              f"{genuine_acc(eflow, xte, yte, S):>13.3f}   (cont {cont:.3f})")
