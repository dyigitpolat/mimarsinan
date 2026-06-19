"""DECISIVE monotonicity test (user's invariant): larger S CANNOT reduce accuracy;
if it does, a mechanism is mis-implemented. Same ANN weights across S (build with a
fixed seed -> S does not affect ANN training), only the cascade resolution + per-S
gain calibration vary. Evaluate the GENUINE cascade vs S.

  - staircase(S): the complete-sum forward, the known-monotonic CONTROL.
  - genuine cold(S): chance (death cascade) — uncalibrated.
  - genuine + gain(S): per-S gain-corrected (scalar theta revive), FIXED ANN weights.

If genuine+gain is NOT monotonically increasing toward cont as S grows, the genuine
forward/decode has an S-dependent bug (NOT an optimization wall).

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/monotonic.py
"""

from __future__ import annotations

import copy
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from mimarsinan.spiking.gain_correction import apply_cascaded_gain_correction  # noqa: E402


def run(depth, S_list=(4, 8, 16, 32, 64, 128)):
    print(f"\n=== d={depth}: genuine cascade vs S (FIXED ANN weights, per-S gain calib) ===")
    print(f"{'S':>4} | {'cont':>6} {'stair':>6} {'cold':>6} {'gain':>6} | {'gap(gain→cont)':>14}")
    for S in S_list:
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=0)
        stair = ttfs_staircase_acc(flow, xte, yte, S)
        cold = genuine_acc(flow, xte, yte, S)
        g = copy.deepcopy(flow)
        apply_cascaded_gain_correction(g, S, rule="relative")
        gc = genuine_acc(g, xte, yte, S)
        print(f"{S:>4} | {cont:>6.3f} {stair:>6.3f} {cold:>6.3f} {gc:>6.3f} | "
              f"{cont - gc:>14.3f}", flush=True)


if __name__ == "__main__":
    run(6)
    run(9)
