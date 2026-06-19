"""Calibration-ONLY recovery of the cascaded cascade: greedy per-layer theta.

The oracle proved a STATIC per-layer theta-trim matches full FT (d=3: 0.944 ==
FT-alone) -- the collapse is a correctable firing-gain. revive.py fails to find it
(it over-shrinks, and crucially shrinks the LAST layer which the oracle keeps at
1.0 -> saturation -> collapse). Brute-force oracle is O(grid^depth); this is the
deployable O(depth*grid) coordinate descent:

  for a few passes, for each layer in order, pick the gamma (theta multiplier) that
  maximises accuracy on a CALIBRATION batch (held-out train), holding others fixed.

NO weight training. gamma calibrated on train, reported on test. If this reaches
the FT-alone numbers across depth, the user's thesis holds: calibrate correctly ->
deep cascade deploys without tricks (and without FT).

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/greedy_theta.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "tests"))
from ft_budget import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from theta_calibrate import full_ft  # noqa: E402
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402


def _acc(flow, x, y, S):
    with torch.no_grad():
        return _accuracy(TTFSSegmentForward(flow.get_mapper_repr(), S)(x.double()), y)


def greedy_theta(flow, xcal, ycal, S, *, grid=(1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1), passes=2):
    """Coordinate-descent per-layer theta multiplier on a calibration batch."""
    percs = flow.get_perceptrons()
    base = [p.activation_scale.clone() for p in percs]
    gammas = [1.0] * len(percs)

    def apply():
        for k, p in enumerate(percs):
            p.set_activation_scale(base[k] * gammas[k])
        install_ttfs_nodes(flow, S)

    apply()
    best_acc = _acc(flow, xcal, ycal, S)
    for _ in range(passes):
        for k in range(len(percs)):
            best_g, cur = gammas[k], best_acc
            for g in grid:
                gammas[k] = g
                apply()
                a = _acc(flow, xcal, ycal, S)
                if a > cur:
                    cur, best_g = a, g
            gammas[k] = best_g
            best_acc = cur
        apply()
    return gammas, best_acc


if __name__ == "__main__":
    print("=== calibration-only greedy per-layer theta (no FT) vs FT-alone ===")
    print(f"{'d':>2} {'S':>3} | {'stair':>6} {'cold':>6} {'greedy-θ':>9} "
          f"{'greedy+FT':>10} {'FT-alone':>9} | gamma")
    for depth in (3, 6, 9):
        for S in (8, 16):
            flow, xtr, ytr, xte, yte, cont, _t, _b = build(depth, S)
            stair = ttfs_staircase_acc(flow, xte, yte, S)
            cold = genuine_acc(flow, xte, yte, S)
            xcal, ycal = xtr[:512], ytr[:512]
            gflow = copy.deepcopy(flow)
            gammas, _ = greedy_theta(gflow, xcal, ycal, S)
            g_acc = genuine_acc(gflow, xte, yte, S)
            gft = full_ft(copy.deepcopy(gflow), xtr, ytr, S, steps=600)
            gft_acc = genuine_acc(gft, xte, yte, S)
            ff = full_ft(copy.deepcopy(flow), xtr, ytr, S, steps=600)
            ff_acc = genuine_acc(ff, xte, yte, S)
            gstr = "[" + ",".join(f"{g:.2f}" for g in gammas) + "]"
            print(f"{depth:>2} {S:>3} | {stair:>6.3f} {cold:>6.3f} {g_acc:>9.3f} "
                  f"{gft_acc:>10.3f} {ff_acc:>9.3f} | {gstr}")
