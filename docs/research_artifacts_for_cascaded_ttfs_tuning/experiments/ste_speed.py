"""Speed + robustness of the staircase-backward STE (mix=0.5), the lossless lever.
How FEW steps reach the ceiling at d=9 S=32, and does it hold across seeds?
Targets: lossless (gap to ceiling < ~1pp) in under 2 min.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/ste_speed.py
"""

from __future__ import annotations

import copy
import os
import sys
import time

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
import recipe_staircase_ste as STE  # noqa: E402

SEEDS = (0, 1)
STEP_BUDGETS = (400, 800, 1200)


def run(depth, S, mix=0.5):
    cache = {s: build(depth, S, seed=s) for s in SEEDS}
    stair = sum(ttfs_staircase_acc(*cache[s][:1], cache[s][3], cache[s][4], S)
                for s in SEEDS) / len(SEEDS)
    cont = cache[SEEDS[0]][5]
    print(f"\n=== d={depth} S={S} mix={mix}: cont={cont:.3f} ceiling={stair:.3f} "
          f"(seeds={SEEDS}) ===")
    print(f"{'steps':>6} {'genuine(mean±std)':>18} {'gap→ceil':>9} {'time_s(mean)':>12}")
    for steps in STEP_BUDGETS:
        accs, times = [], []
        for s in SEEDS:
            flow, xtr, ytr, xte, yte, _c, teacher, base = cache[s]
            f = copy.deepcopy(flow)
            t0 = time.time()
            STE.train(f, xtr, ytr, xtr, ytr, S, base, teacher,
                      steps=steps, seed=s, mix=mix)
            times.append(time.time() - t0)
            accs.append(genuine_acc(f, xte, yte, S))
        t = torch.tensor(accs)
        m, sd = float(t.mean()), float(t.std())
        tm = sum(times) / len(times)
        print(f"{steps:>6} {f'{m:.3f}±{sd:.3f}':>18} {stair - m:>9.3f} {tm:>12.1f}",
              flush=True)


if __name__ == "__main__":
    run(9, 32, mix=0.5)
    run(6, 32, mix=0.5)   # universality: does mix=0.5 hold at moderate depth?
