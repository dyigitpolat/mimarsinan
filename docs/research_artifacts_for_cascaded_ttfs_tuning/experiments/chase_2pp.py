"""Chase the last ~2pp on a deep cascade: regularized genuine fine-tuning recipes.

The plain genuine FT plateaus at ~0.959 (cont 0.981) and OVERFITS beyond ~800 steps
(0.844 @1600) — the cascade's harder per-sample decision surface overfits. Levers:
KD (soft teacher targets), cosine LR, SWA (flat minima), val-based early-stop.
Question: how close to the continuous 0.981 can a well-regularized genuine FT get?
"""

from __future__ import annotations

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
from ft_budget import build, genuine_acc  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402


def _val_split(xtr, ytr, frac=0.15, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(xtr.shape[0], generator=g).to(xtr.device)
    n = int(frac * xtr.shape[0])
    return xtr[perm[n:]], ytr[perm[n:]], xtr[perm[:n]], ytr[perm[:n]]


def ft(flow, xtr, ytr, xva, yva, S, base, *, steps=1500, lr=2e-3, bs=256, seed=0,
       kd=False, kd_T=3.0, kd_alpha=0.3, cosine=False, swa=False, swa_start=0.6,
       label_smooth=0.0, eval_every=75):
    """Regularized genuine-cascade FT. Returns (best_val_state, swa_state, log)."""
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
             if cosine else None)
    ce = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    g = torch.Generator().manual_seed(seed)
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    if kd:
        base.eval()
        with torch.no_grad():
            pass
    best_val, best_state = -1.0, None
    swa_acc, swa_n = None, 0
    swa_from = int(swa_start * steps)
    log = []
    for t in range(steps):
        idx = torch.randint(0, xtr.shape[0], (bs,), generator=g).to(xtr.device)
        x, y = xtr[idx].double(), ytr[idx]
        logits = drv(x)
        loss = ce(logits, y)
        if kd:
            with torch.no_grad():
                tl = base(x.float()) / kd_T
            loss = kd_alpha * loss + (1 - kd_alpha) * (
                F.kl_div(F.log_softmax(logits / kd_T, -1), F.softmax(tl, -1),
                         reduction="batchmean") * kd_T * kd_T)
        opt.zero_grad(); loss.backward(); opt.step()
        if sched:
            sched.step()
        if swa and t >= swa_from:
            sd = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            if swa_acc is None:
                swa_acc = sd; swa_n = 1
            else:
                for k in swa_acc:
                    if swa_acc[k].is_floating_point():
                        swa_acc[k].mul_(swa_n / (swa_n + 1)).add_(sd[k] / (swa_n + 1))
                swa_n += 1
        if (t + 1) % eval_every == 0:
            with torch.no_grad():
                va = _accuracy(drv(xva.double()), yva)
            if va > best_val:
                best_val = va
                best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            log.append((t + 1, va))
    return best_state, swa_acc, log


def run(flow, xtr, ytr, xva, yva, xte, yte, S, base, **kw):
    f = copy.deepcopy(flow)
    best, swa, log = ft(f, xtr, ytr, xva, yva, S, base, **kw)
    out = {}
    if best is not None:
        f.load_state_dict(best); out["earlystop"] = genuine_acc(f, xte, yte, S)
    if swa is not None:
        f.load_state_dict(swa); out["swa"] = genuine_acc(f, xte, yte, S)
    return out


if __name__ == "__main__":
    DEPTH, S = 6, 8
    flow, xtr_full, ytr_full, xte, yte, cont, _, base = build(DEPTH, S)
    xtr, ytr, xva, yva = _val_split(xtr_full, ytr_full)
    print(f"=== chase the last 2pp: depth={DEPTH} S={S}, continuous={cont:.3f} ===")
    recipes = {
        "plain (const lr, earlystop)": dict(),
        "cosine LR": dict(cosine=True),
        "KD": dict(kd=True),
        "label_smooth=0.1": dict(label_smooth=0.1),
        "KD+cosine+SWA": dict(kd=True, cosine=True, swa=True),
        "KD+cosine+SWA+ls0.05": dict(kd=True, cosine=True, swa=True, label_smooth=0.05),
    }
    for name, kw in recipes.items():
        out = run(flow, xtr, ytr, xva, yva, xte, yte, S, base, steps=1500, **kw)
        s = "  ".join(f"{k}={v:.4f}" for k, v in out.items())
        print(f"  {name:>26}: {s}   (cont {cont:.3f}, gap {cont - max(out.values()):.4f})")
