"""DIRECTION 3 diagnostic baseline: is a NON-NEGATIVE-weight net already cold-lossless?

If mixed-sign CANCELLATION is THE cause of the cold cascade collapse, then a net whose
INTERIOR weights are constrained >= 0 (no cancellation: every running partial sum is a
monotone non-decreasing lower bound of the complete sum) should have COLD genuine ~=
staircase, BY CONSTRUCTION, with no fine-tuning.

We train such a net FROM SCRATCH (so cont stays high, unlike abs()-ing a trained net):
each interior Linear has weights reparameterized as softplus(raw) >= 0; the classifier
(last layer) stays signed. Compare cold genuine vs staircase vs cont at d in {6,9}.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(__file__)
_ART = "/home/yigit/repos/research_stuff/mimarsinan/docs/research_artifacts_for_cascaded_ttfs_tuning/experiments"
for _p in (_HERE, _ART, os.path.join(_ART, ".."), "/home/yigit/repos/research_stuff/mimarsinan/tests"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_lab import _accuracy, digits_task, _capture_activation_means  # noqa: E402
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from recipe_harness import genuine_acc  # noqa: E402
from ft_budget import _calibrate_cuda, DEV  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


class _NonNegLinear(nn.Module):
    """Linear with weights constrained >= 0 via softplus reparameterization."""

    def __init__(self, a, b, nonneg=True):
        super().__init__()
        self.nonneg = nonneg
        self.raw = nn.Parameter(torch.randn(b, a) * 0.1)
        self.bias = nn.Parameter(torch.zeros(b))

    def forward(self, x):
        w = nn.functional.softplus(self.raw) if self.nonneg else self.raw
        return nn.functional.linear(x, w, self.bias)


class _NonNegMLP(nn.Module):
    """Interior layers non-negative (no cancellation); classifier signed."""

    def __init__(self, depth, width, in_dim, out_dim):
        super().__init__()
        dims = [in_dim] + [width] * (depth - 1) + [out_dim]
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            last = i == depth - 1
            layers.append(nn.Sequential(
                _NonNegLinear(a, b, nonneg=not last),
                nn.BatchNorm1d(b),
                nn.ReLU() if not last else nn.Identity(),
            ))
        self.stages = nn.ModuleList(layers)

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        return x


def _materialize_nonneg(base):
    """Replace each _NonNegLinear with a plain nn.Linear holding softplus(raw) so the
    converter sees standard Linear layers (the constraint is baked into the weights)."""
    for stage in base.stages:
        nnl = stage[0]
        lin = nn.Linear(nnl.raw.shape[1], nnl.raw.shape[0])
        w = nn.functional.softplus(nnl.raw) if nnl.nonneg else nnl.raw
        with torch.no_grad():
            lin.weight.copy_(w)
            lin.bias.copy_(nnl.bias)
        stage[0] = lin
    return base


def build_nonneg(depth, S, seed=0, epochs=250, lr=2e-3):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = (t.to(DEV) for t in digits_task(seed=seed + 1))
    base = _NonNegMLP(depth, 96, 64, 10).to(DEV)
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    base.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(base(xtr.float()), ytr).backward(); opt.step()
    base.eval()
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    _materialize_nonneg(base)
    flow = convert_torch_model(base, (64,), 10, device=str(DEV))
    _calibrate_cuda(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    return flow, xte, yte, cont


def run():
    print("=== DIAGNOSTIC: non-negative interior weights, trained from scratch ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'stair':>6} {'cold_gen':>9} | {'gap(cold→stair)':>15}")
    for depth in (6, 9):
        for S in (8,):
            flow, xte, yte, cont = build_nonneg(depth, S, seed=0)
            stair = ttfs_staircase_acc(flow, xte, yte, S)
            cold = genuine_acc(flow, xte, yte, S)
            print(f"{depth:>2} {S:>3} | {cont:>6.3f} {stair:>6.3f} {cold:>9.3f} | "
                  f"{stair - cold:>15.3f}", flush=True)


if __name__ == "__main__":
    run()
