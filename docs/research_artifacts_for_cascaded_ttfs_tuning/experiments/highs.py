"""Can genuine cascaded reach NEAR the continuous ANN at HIGHER S, with proper
optimisation? The earlier S-sweep used a fixed budget, no grad-clip, and the weaker
baseline recipes — and genuine FT degraded at high S. This re-tests with the COMBO
recipe + gradient clipping + a generous budget, since higher S raises the staircase
ceiling toward the ANN (d=9 staircase: S16 0.952 -> S32 0.965 = cont). If combo
tracks that rising ceiling, genuine cascaded is near-lossless at high S.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/highs.py
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
import recipe_combo as C  # noqa: E402

STEPS = 2500
SEEDS = (0,)


def run(depth, S):
    accs, stairs = [], []
    cont = None
    for seed in SEEDS:
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
        stairs.append(ttfs_staircase_acc(flow, xte, yte, S))
        f = C.train(copy.deepcopy(flow), xtr, ytr, xtr, ytr, S, base, teacher,
                    steps=STEPS, seed=seed, grad_clip=1.0)
        accs.append(genuine_acc(f, xte, yte, S))
    t = torch.tensor(accs)
    return cont, sum(stairs) / len(stairs), float(t.mean()), float(t.std())


if __name__ == "__main__":
    print(f"=== combo + grad_clip, gap to CONTINUOUS vs S (steps={STEPS}, seeds={SEEDS}) ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'stair':>6} {'combo':>13} | {'gap→cont':>9} {'gap→stair':>10}")
    for depth in (6, 9):
        for S in (8, 16, 32, 48):
            cont, stair, m, sd = run(depth, S)
            print(f"{depth:>2} {S:>3} | {cont:>6.3f} {stair:>6.3f} {f'{m:.3f}±{sd:.3f}':>13} | "
                  f"{cont - m:>9.3f} {stair - m:>10.3f}", flush=True)
