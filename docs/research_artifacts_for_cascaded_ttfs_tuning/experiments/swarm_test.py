"""Can the S WARM-START break the high-S optimisation barrier and reach the ANN-
level ceiling? At high S the staircase ceiling = the ANN, but DIRECT combo degrades
(d=9 S=32: ceiling 0.965, combo 0.911). Warm-start trains combo at the easy S=16
basin, then continues at the high deploy-S. Compares: staircase ceiling | direct
combo | warm-start combo, at the high-S operating points.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/swarm_test.py
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
import recipe_combo_swarm as W  # noqa: E402

STEPS = 2500
SEEDS = (0, 1)


def _mean_std(vals):
    t = torch.tensor(vals)
    return float(t.mean()), float(t.std())


def run(depth, S):
    stair, direct, warm = [], [], []
    cont = None
    for seed in SEEDS:
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
        stair.append(ttfs_staircase_acc(flow, xte, yte, S))
        d = C.train(copy.deepcopy(flow), xtr, ytr, xtr, ytr, S, base, teacher,
                    steps=STEPS, seed=seed, grad_clip=1.0)
        direct.append(genuine_acc(d, xte, yte, S))
        w = W.train(copy.deepcopy(flow), xtr, ytr, xtr, ytr, S, base, teacher,
                    steps=STEPS, seed=seed, grad_clip=1.0, warm_S=16, warm_frac=0.5)
        warm.append(genuine_acc(w, xte, yte, S))
    return cont, _mean_std(stair), _mean_std(direct), _mean_std(warm)


if __name__ == "__main__":
    print(f"=== S warm-start vs direct combo at high S (steps={STEPS}, seeds={SEEDS}) ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'stair':>13} {'direct':>13} {'warm-start':>13} | {'warm gap→cont':>13}")
    for depth, S in ((6, 32), (9, 32), (9, 48)):
        cont, st, di, wa = run(depth, S)
        print(f"{depth:>2} {S:>3} | {cont:>6.3f} {f'{st[0]:.3f}±{st[1]:.3f}':>13} "
              f"{f'{di[0]:.3f}±{di[1]:.3f}':>13} {f'{wa[0]:.3f}±{wa[1]:.3f}':>13} | "
              f"{cont - wa[0]:>13.3f}", flush=True)
