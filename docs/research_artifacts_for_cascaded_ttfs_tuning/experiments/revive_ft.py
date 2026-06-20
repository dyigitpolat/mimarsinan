"""THE WAY OUT: data-grounded REVIVE (no gradient) -> FT, vs FT-alone, at depth.

theta_calibrate.py showed gradient theta-calibration FAILS to revive (dead neurons
emit no spike -> no surrogate gradient -> stuck at chance for d>=6). But the oracle
proved a recovering theta EXISTS (d=3: 0.909 == staircase). The fix for the dead-
neuron-no-gradient problem is a DATA-GROUNDED revive: directly lower theta for
under-firing neurons the teacher says SHOULD fire (revive.calibrate_revive) -- no
gradient needed. Then a short FT polishes the revived (now-differentiable) cascade.

Compares, on the SAME GPU flow, across depth x S:
  cold | revive | revive+FT | FT-alone | (stair ceiling)

If revive+FT > FT-alone (esp. at d=9 where FT-alone is unstable), the revive-then-
FT curriculum is the deployable recipe and the user's thesis holds: calibrate the
firing regime first, then FT closes to the staircase.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/revive_ft.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "tests"))
sys.path.insert(0, os.path.join(_HERE, ".."))
from ft_budget import DEV, build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from theta_calibrate import full_ft  # noqa: E402
from revive import calibrate_revive  # noqa: E402


def _revive(flow, xte, S, teacher, **kw):
    f = copy.deepcopy(flow)
    calibrate_revive(f, xte, S, teacher, per_channel=False, **kw)
    return f


if __name__ == "__main__":
    STEPS = 600
    print(f"=== revive -> FT vs FT-alone (device={DEV}, FT steps={STEPS}) ===")
    print(f"{'d':>2} {'S':>3} | {'stair':>6} {'cold':>6} {'revive':>7} "
          f"{'revive+FT':>10} {'FT-alone':>9}")
    for depth in (3, 6, 9):
        for S in (8, 16):
            flow, xtr, ytr, xte, yte, cont, teacher, _b = build(depth, S)
            stair = ttfs_staircase_acc(flow, xte, yte, S)
            cold = genuine_acc(flow, xte, yte, S)
            rv = _revive(flow, xte, S, teacher)
            rv_acc = genuine_acc(rv, xte, yte, S)
            rvft = full_ft(copy.deepcopy(rv), xtr, ytr, S, steps=STEPS)
            rvft_acc = genuine_acc(rvft, xte, yte, S)
            ft = full_ft(copy.deepcopy(flow), xtr, ytr, S, steps=STEPS)
            ft_acc = genuine_acc(ft, xte, yte, S)
            print(f"{depth:>2} {S:>3} | {stair:>6.3f} {cold:>6.3f} {rv_acc:>7.3f} "
                  f"{rvft_acc:>10.3f} {ft_acc:>9.3f}")
