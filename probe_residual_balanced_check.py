"""Adversarial confound check: re-measure the fix-study deploy with a
balanced (per-class) metric to test whether retention is real discrimination
or collapse-to-majority-class on the imbalanced synthetic eval set.

Reuses the IDENTICAL production deploy path (deploy_with_fix from the study)
and the SAME trained base + eval slice. For each (fix, mode) reports:
  - top1 retention (the study's metric)            = dep_top1 / ann_top1
  - balanced(macro-recall) retention               = dep_bal  / ann_bal
  - n_classes_predicted by the deployment           (collapse detector)
  - majority-baseline top1 on this eval slice       (degeneracy floor)

Usage: PYTHONPATH=src:spikingjelly env/bin/python probe_residual_balanced_check.py <fix> <depth> <seed> <T> [modes]
"""
from __future__ import annotations

import sys
import torch

import probe_residual_genuine_spiking_sweep as P
import probe_residual_fix_study as S

P.N_EVAL = 200
S.P.N_EVAL = 200


def balanced_acc(pred, ye, nc):
    recalls = []
    for c in range(nc):
        m = ye == c
        if m.sum() == 0:
            continue
        recalls.append((pred[m] == c).float().mean().item())
    return sum(recalls) / len(recalls)


def deployed_preds(hcm, Xe, T):
    with torch.no_grad():
        dt = next(hcm.parameters()).dtype if any(True for _ in hcm.parameters()) else Xe.dtype
        out = hcm(Xe.to(dt)).float() / float(T)
    return out.argmax(1)


def run(fix, depth, seed, T, modes):
    Xtr, ytr, Xe, ye = P.make_task(seed)
    Xe, ye = Xe[:P.N_EVAL], ye[:P.N_EVAL]
    cal_x = Xtr[:P.N_CAL]
    torch.manual_seed(seed)
    base = P.train(P.ResidualStack(depth), Xtr, ytr, seed=seed)
    nc = P.NC
    bc = torch.bincount(ye, minlength=nc)
    maj = bc.max().item() / ye.shape[0]

    teacher_ref = None
    print(f"FIX={fix} depth={depth} seed={seed} T={T} eval_n={ye.shape[0]} "
          f"class_counts={bc.tolist()} majority_top1={maj:.3f}")
    print(f"{'mode':<15}{'ann_t1':>7}{'dep_t1':>7}{'ret_t1':>8}"
          f"{'ann_bal':>8}{'dep_bal':>8}{'ret_bal':>8}{'dep_ncls':>9}")
    sel = [m for m in P.MODES if m[0] in modes]
    for name, mode, sched in sel:
        flow, hcm, teacher, ns = S.deploy_with_fix(fix, base, cal_x, Xtr, ytr, mode, sched, T)
        with torch.no_grad():
            tpred = teacher(Xe.to(next(teacher.parameters()).dtype)).argmax(1)
        dpred = deployed_preds(hcm, Xe, T)
        ann_t1 = (tpred == ye).float().mean().item()
        dep_t1 = (dpred == ye).float().mean().item()
        ann_bal = balanced_acc(tpred, ye, nc)
        dep_bal = balanced_acc(dpred, ye, nc)
        ret_t1 = dep_t1 / ann_t1 if ann_t1 > 0 else 0.0
        ret_bal = dep_bal / ann_bal if ann_bal > 0 else 0.0
        ncls = len(torch.unique(dpred))
        print(f"{name:<15}{ann_t1:>7.3f}{dep_t1:>7.3f}{100*ret_t1:>7.1f}%"
              f"{ann_bal:>8.3f}{dep_bal:>8.3f}{100*ret_bal:>7.1f}%{ncls:>9}")


if __name__ == "__main__":
    fix = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    modes = sys.argv[5].split(",") if len(sys.argv) > 5 else ["lif", "ttfs_cascaded", "ttfs_sync"]
    run(fix, depth, seed, T, modes)
