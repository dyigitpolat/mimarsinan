"""Near-lossless head-to-head at the GOOD operating point (S=16, where genuine FT
is most trainable). All recipes, all-data (no val-split — it costs ~1.5pp on this
tiny task), longer budget, 2 seeds. Primary metric: gap to the CONTINUOUS ANN.

The S-sweep proved the cap is OPTIMISATION (higher S = longer cascade = harder),
not expressiveness (staircase=0.98 same arch). So the question is which training
recipe drives the genuine cascade closest to its own lossless staircase.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/focused_nearloss.py [recipe ...]
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
from bench_recipes import discover  # noqa: E402

CONFIGS = [(6, 16), (9, 16)]
SEEDS = (0, 1)
STEPS = 1500


def main():
    names = sys.argv[1:] or None
    recipes = discover(names)
    print(f"=== near-lossless @ S=16 (all-data, steps={STEPS}, seeds={SEEDS}) ===")
    print(f"recipes: {list(recipes)}")
    cache = {}
    for depth, S in CONFIGS:
        for seed in SEEDS:
            cache[(depth, S, seed)] = build(depth, S, seed=seed)
        cont = cache[(depth, S, SEEDS[0])][5]
        stair = sum(ttfs_staircase_acc(*cache[(depth, S, s)][:1], cache[(depth, S, s)][3],
                                       cache[(depth, S, s)][4], S) for s in SEEDS) / len(SEEDS)
        print(f"\n--- d={depth} S={S}  cont={cont:.3f}  staircase={stair:.3f} ---")
        print(f"{'recipe':>20} | {'genuine(mean±std)':>18} | {'gap→cont':>9}")
        for name, train in recipes.items():
            accs = []
            for seed in SEEDS:
                flow, xtr, ytr, xte, yte, _cont, teacher, base = cache[(depth, S, seed)]
                f = copy.deepcopy(flow)
                try:
                    f = train(f, xtr, ytr, xtr, ytr, S, base, teacher,
                              steps=STEPS, seed=seed) or f
                    accs.append(genuine_acc(f, xte, yte, S))
                except Exception as e:
                    print(f"    [{name} seed={seed}] ERROR {type(e).__name__}: {e}")
                    accs.append(0.0)
            t = torch.tensor(accs)
            m, sd = float(t.mean()), float(t.std())
            print(f"{name:>20} | {f'{m:.3f}±{sd:.3f}':>18} | {cont - m:>9.3f}")


if __name__ == "__main__":
    main()
