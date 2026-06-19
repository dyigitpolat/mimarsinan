"""Benchmark all `recipe_*.py` for FAST + STABLE cold-ANN -> cascaded-TTFS FT.

Builds each (depth,S,seed) ANN ONCE (cached) and runs every recipe on a deepcopy,
at a SHORT budget (speed) and a LONGER budget (ceiling), over multiple seeds
(stability). Reports per recipe: mean±std genuine accuracy at each budget, and the
mean gap to the analytical staircase ceiling.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/bench_recipes.py
    # restrict:  ... bench_recipes.py kd_baseline blend_curriculum
"""

from __future__ import annotations

import copy
import glob
import importlib
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
from chase_2pp import _val_split  # noqa: E402

CONFIGS = [(6, 16), (9, 16)]
SEEDS = (0, 1, 2)
BUDGETS = (200, 800)


def discover(names=None):
    recipes = {}
    for path in sorted(glob.glob(os.path.join(_HERE, "recipe_*.py"))):
        mod = importlib.import_module(os.path.splitext(os.path.basename(path))[0])
        if hasattr(mod, "NAME") and hasattr(mod, "train"):
            if names is None or mod.NAME in names:
                recipes[mod.NAME] = mod.train
    return recipes


_BUILD_CACHE = {}


def cached_build(depth, S, seed):
    key = (depth, S, seed)
    if key not in _BUILD_CACHE:
        _BUILD_CACHE[key] = build(depth, S, seed=seed)
    return _BUILD_CACHE[key]


def _stats(vals):
    t = torch.tensor(vals)
    return float(t.mean()), float(t.std())


def main():
    names = sys.argv[1:] or None
    recipes = discover(names)
    print(f"recipes: {list(recipes)}")
    for depth, S in CONFIGS:
        stair_vals = []
        for seed in SEEDS:
            flow, xtr, ytr, xte, yte, cont, teacher, base = cached_build(depth, S, seed)
            stair_vals.append(ttfs_staircase_acc(flow, xte, yte, S))
        stair_m, _ = _stats(stair_vals)
        print(f"\n=== d={depth} S={S}  staircase(ceiling)={stair_m:.3f} "
              f"cont={cont:.3f} ===")
        header = f"{'recipe':>20} | " + " | ".join(
            f"acc@{b}(mean±std)".rjust(18) for b in BUDGETS)
        print(header)
        for name, train in recipes.items():
            cells = []
            for budget in BUDGETS:
                accs = []
                for seed in SEEDS:
                    flow, xtr, ytr, xte, yte, cont, teacher, base = cached_build(depth, S, seed)
                    xt, yt, xv, yv = _val_split(xtr, ytr, seed=seed)
                    f = copy.deepcopy(flow)
                    try:
                        f = train(f, xt, yt, xv, yv, S, base, teacher,
                                  steps=budget, seed=seed) or f
                        accs.append(genuine_acc(f, xte, yte, S))
                    except Exception as e:  # a broken recipe must not kill the sweep
                        print(f"    [{name} seed={seed} b={budget}] ERROR {type(e).__name__}: {e}")
                        accs.append(0.0)
                m, sd = _stats(accs)
                cells.append(f"{m:.3f}±{sd:.3f}".rjust(18))
            print(f"{name:>20} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
