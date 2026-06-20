"""Forward monotonicity with an ALIVE model: FT once at a base S, then evaluate the
GENUINE cascade (fixed weights) across eval-S. User's invariant: an S=16-trained
model evaluated at S=32 must be >= its S=16 accuracy (finer resolution cannot hurt).

  - If genuine_acc(fixed model) is monotonic-up in eval-S -> forward is correct; the
    high-S degradation in highs.py was a TRAINING-from-scratch failure (the S=32
    optimum exists -- deploy the S=16 weights at S=32 -- but FT didn't find it).
  - If it DROPS at higher eval-S -> the genuine forward has an S-scale bug (the same
    weights mis-decode at finer resolution).

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/monotonic2.py
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
import recipe_combo as C  # noqa: E402

EVAL_S = (8, 16, 32, 64)


def run(depth, base_S, steps=1500):
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, base_S, seed=0)
    f = C.train(copy.deepcopy(flow), xtr, ytr, xtr, ytr, base_S, base, teacher,
                steps=steps, seed=0, grad_clip=1.0)
    print(f"\n=== d={depth}, model FT'd at S={base_S} (cont={cont:.3f}); "
          f"eval genuine at each S (FIXED weights) ===")
    print(f"{'eval_S':>6} {'genuine':>8} {'staircase':>10}")
    for S in EVAL_S:
        g = genuine_acc(f, xte, yte, S)
        s = ttfs_staircase_acc(f, xte, yte, S)
        print(f"{S:>6} {g:>8.3f} {s:>10.3f}", flush=True)


if __name__ == "__main__":
    run(9, 16)
    run(6, 16)
