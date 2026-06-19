"""DECISIVE near-lossless test: can genuine cascaded FT reach the continuous ANN?

Frozen-weight calibration is S-independent (~3-4pp below staircase), but FT can
EXPLOIT timing budget (d=6: FT S=8 0.944 -> S=16 0.957). So sweep S and measure the
genuine-FT gap to the CONTINUOUS ANN (the true lossless target), at depth. If the
gap closes to <~1pp at practical S, cascaded near-lossless IS reachable by FT+S.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/decisive_nearloss.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from chase_2pp import _val_split  # noqa: E402
import recipe_kd_baseline as KD  # noqa: E402
import recipe_blend_curriculum as BL  # noqa: E402

STEPS = 1200


def run(depth, S, seed=0):
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
    stair = ttfs_staircase_acc(flow, xte, yte, S)
    xt, yt, xv, yv = _val_split(xtr, ytr, seed=seed)
    out = {"cont": cont, "stair": stair}
    for tag, mod in (("kd", KD), ("blend", BL)):
        f = mod.train(copy.deepcopy(flow), xt, yt, xv, yv, S, base, teacher,
                      steps=STEPS, seed=seed)
        out[tag] = genuine_acc(f, xte, yte, S)
    return out


if __name__ == "__main__":
    print(f"=== genuine cascaded FT vs S: gap to CONTINUOUS (steps={STEPS}) ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'stair':>6} {'kd-FT':>6} {'blend':>6} "
          f"| {'gap(kd→cont)':>12} {'gap(blend→cont)':>15}")
    for depth in (6, 9):
        for S in (16, 32, 64):
            r = run(depth, S)
            best = max(r["kd"], r["blend"])
            print(f"{depth:>2} {S:>3} | {r['cont']:>6.3f} {r['stair']:>6.3f} "
                  f"{r['kd']:>6.3f} {r['blend']:>6.3f} | "
                  f"{r['cont']-r['kd']:>12.3f} {r['cont']-r['blend']:>15.3f}")
