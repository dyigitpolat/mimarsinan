"""Incremental baseline: deep-residual genuine-spiking collapse, prints PER MODE.

Reuses the production deploy path from probe_residual_genuine_spiking_sweep.deploy
so the substrate is IDENTICAL. Smaller eval set + single seed + flush-per-mode so
the collapse numbers land fast and incrementally (the 3-mode sweep only prints
after a whole depth finishes).
"""
from __future__ import annotations

import copy
import sys

import torch

import probe_residual_genuine_spiking_sweep as P

P.N_EVAL = 200  # smaller eval for speed; deterministic slice of the same task


def run(depth, seed, T):
    P.T = T
    Xtr, ytr, Xe, ye = P.make_task(seed)
    cal_x = Xtr[: P.N_CAL]
    torch.manual_seed(seed)
    base = P.train(P.ResidualStack(depth), Xtr, ytr, seed=seed)
    print(f"depth={depth} seed={seed} T={T} W={P.W} chance={1.0/P.NC:.3f}", flush=True)
    print(f"{'mode':<16}{'ANN':>7}{'NF':>8}{'deployed':>10}{'ret%':>8}{'nseg':>6}", flush=True)
    for name, mode, sched in P.MODES:
        m = copy.deepcopy(base)
        flow, hcm, teacher, ns = P.deploy(
            m, cal_x, mode, ttfs_cycle_schedule=(sched or "cascaded"))
        ann, nf, hc = P.measure(flow, hcm, teacher, Xe, ye, mode)
        ret = hc / ann if ann > 0 else 0.0
        print(f"{name:<16}{ann:>7.3f}{nf:>8.3f}{hc:>10.3f}{100*ret:>7.1f}%{ns:>6}", flush=True)


if __name__ == "__main__":
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    run(depth, seed, T)
