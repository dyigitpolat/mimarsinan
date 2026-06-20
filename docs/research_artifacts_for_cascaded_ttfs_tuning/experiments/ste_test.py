"""Test the staircase-backward STE at the d=9 S=32 plateau. Sweep the backward mix
(0 = pure genuine surrogate == combo; 1 = pure staircase complete-sum gradient).
Does injecting the clean staircase gradient onto the genuine basin cross the ~0.95
plateau toward the 0.965 ceiling?

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/ste_test.py
"""

from __future__ import annotations

import copy
import os
import sys
import time

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
import recipe_staircase_ste as STE  # noqa: E402

STEPS = 1500


def run(depth, S, mixes=(0.0, 0.5, 0.75, 1.0), seed=0):
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
    stair = ttfs_staircase_acc(flow, xte, yte, S)
    print(f"\n=== d={depth} S={S}: cont={cont:.3f} staircase(ceiling)={stair:.3f} "
          f"(steps={STEPS}) ===")
    print(f"{'mix':>5} {'genuine':>8} {'gap→ceil':>9} {'time_s':>7}")
    for mix in mixes:
        f = copy.deepcopy(flow)
        t0 = time.time()
        STE.train(f, xtr, ytr, xtr, ytr, S, base, teacher,
                  steps=STEPS, seed=seed, mix=mix)
        dt = time.time() - t0
        a = genuine_acc(f, xte, yte, S)
        print(f"{mix:>5.2f} {a:>8.3f} {stair - a:>9.3f} {dt:>7.1f}", flush=True)


if __name__ == "__main__":
    run(9, 32)
    run(6, 32, mixes=(0.0, 0.75, 1.0))
