"""A1 -- THE LOSS LADDER (cascade-conversion loss budget).

Decompose the ANN -> deployed cascaded single-spike TTFS accuracy loss into NAMED,
CUMULATIVE sources, so each ladder rung's delta is exactly that transformation's cost:

  rung 0  cont        : the continuous ANN accuracy            (the LOSSLESS target)
  rung 1  staircase   : + activation quantization to S levels  (ttfs_staircase_acc,
                        cycle_accurate OFF -- the optimal linear single-spike timing
                        decode at T+1 levels). The QUANTIZATION loss = cont - staircase.
  rung 2  genuine-STE : + genuine single-spike CASCADE firing, BEST-trained via
                        recipe_staircase_ste (forward genuine fire-once ramp, backward
                        mix*staircase + (1-mix)*genuine, mix=0.5, ~800 steps). The
                        CASCADE-CODE loss = staircase - genuine.

Every loss is reported as a DELTA from cont (pp). Sweep S in {8,16,32,64} at d=9 and
d=6. The per-S table localizes WHICH source dominates and how each shrinks with S.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/loss_ladder.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from recipe_harness import build, genuine_acc  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc  # noqa: E402
import recipe_staircase_ste as ste  # noqa: E402


def ladder_at(depth, S, *, steps, seed, mix=0.5):
    """Return one ladder row dict for (depth, S). The STE-trained genuine cascade
    is evaluated on the PURE genuine deploy path (no gradient hedging)."""
    torch.manual_seed(seed)
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)

    stair = ttfs_staircase_acc(flow, xte, yte, S)
    cold = genuine_acc(flow, xte, yte, S)

    t0 = time.time()
    ste.train(flow, xtr, ytr, xte, yte, S, base, teacher,
              steps=steps, seed=seed, mix=mix)
    gen = genuine_acc(flow, xte, yte, S)
    train_s = time.time() - t0

    return {
        "depth": depth, "S": S,
        "cont": cont, "staircase": stair, "cold_gen": cold, "genuine": gen,
        "quant_loss": cont - stair,        # rung 0->1 : activation quantization
        "cascade_loss": stair - gen,       # rung 1->2 : genuine single-spike cascade code
        "total_loss": cont - gen,          # ANN -> deployed
        "train_s": train_s,
    }


def _fmt_row(r):
    pp = lambda v: f"{100.0 * v:+6.2f}"  # noqa: E731
    return (f"{r['depth']:>2} {r['S']:>4} | "
            f"{r['cont']:>6.4f} {r['staircase']:>9.4f} {r['genuine']:>9.4f} | "
            f"{pp(r['quant_loss'])} {pp(r['cascade_loss'])} {pp(r['total_loss'])} | "
            f"{r['cold_gen']:>6.4f} {r['train_s']:>7.1f}s")


def _dominant(r):
    q, c = r["quant_loss"], r["cascade_loss"]
    if abs(q - c) < 0.005:
        return "TIE"
    return "QUANT" if q > c else "CASCADE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[9, 6])
    ap.add_argument("--svals", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mix", type=float, default=0.5)
    args = ap.parse_args()

    print(f"=== A1 LOSS LADDER  (STE mix={args.mix}, {args.steps} steps, "
          f"seed {args.seed}) ===")
    print("rung0 cont (ANN, lossless target) | rung1 +S-level activation quant "
          "(staircase) | rung2 +genuine single-spike cascade (STE-trained)\n")
    header = (f"{'d':>2} {'S':>4} | {'cont':>6} {'stair':>9} {'genuine':>9} | "
              f"{'quantLp':>6} {'cascLp':>6} {'totLp':>6} | "
              f"{'coldgen':>6} {'train':>8}  dominant")
    print(header)
    print("-" * len(header))

    rows = []
    t_all = time.time()
    for depth in args.depths:
        for S in args.svals:
            r = ladder_at(depth, S, steps=args.steps, seed=args.seed, mix=args.mix)
            rows.append(r)
            print(_fmt_row(r) + f"  {_dominant(r)}", flush=True)
        print()

    print(f"(all losses in pp vs cont; total wall {time.time() - t_all:.0f}s)\n")

    print("=== DOMINANCE + SHRINK-WITH-S (per depth) ===")
    for depth in args.depths:
        drows = sorted((r for r in rows if r["depth"] == depth), key=lambda r: r["S"])
        print(f"d={depth}:")
        for r in drows:
            print(f"   S={r['S']:>3}: quant {100*r['quant_loss']:+5.2f}pp  "
                  f"cascade {100*r['cascade_loss']:+5.2f}pp  -> dominant {_dominant(r)}")
        if len(drows) >= 2:
            lo, hi = drows[0], drows[-1]
            print(f"   shrink S={lo['S']}->{hi['S']}: "
                  f"quant {100*lo['quant_loss']:+5.2f}->{100*hi['quant_loss']:+5.2f}pp  "
                  f"cascade {100*lo['cascade_loss']:+5.2f}->{100*hi['cascade_loss']:+5.2f}pp")
    return rows


if __name__ == "__main__":
    main()
