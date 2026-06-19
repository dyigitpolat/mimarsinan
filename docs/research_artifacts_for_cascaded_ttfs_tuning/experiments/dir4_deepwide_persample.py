"""DIRECTION 4: deep+wide scaling of the hedged staircase-backward STE + per-sample
residual diagnosis of the genuine cascade vs the staircase ceiling.

Two questions:
  (a) Does recipe_staircase_ste (forward=genuine fire-once cascade, backward=
      mix*staircase + (1-mix)*genuine surrogate, mix=0.5) hold LOSSLESS as we go
      DEEP (d in 6,9,12,15) and WIDE (width in 96,192,384)? Find where it breaks.
  (b) After STE training, does the genuine cascade match the staircase PER-SAMPLE
      (fraction of test samples with identical argmax / logit correlation), or only
      in AGGREGATE accuracy (a hidden residual cascade-code loss)?

No src change. Parametrizes ft_budget.build over width (it hardcodes 96). Eval is
PURE genuine (the deployed metric). The staircase is the lossless ceiling.

Run:  source env/bin/activate
      python docs/.../experiments/dir4_deepwide_persample.py quick   # d6/9 w96/192 S8 1 seed
      python docs/.../experiments/dir4_deepwide_persample.py         # full grid, 3 seeds
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from cascade_lab import _accuracy, _capture_activation_means, digits_task  # noqa: E402
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from ft_budget import DEV, _DeepMLPBN, _calibrate_cuda  # noqa: E402
from recipe_staircase_ste import train as ste_train  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


def build(depth, S, width, seed=0, epochs=200, lr=2e-3):
    """ft_budget.build, parametrized over the hidden WIDTH (it hardcodes 96)."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = (t.to(DEV) for t in digits_task(seed=seed + 1))
    base = _DeepMLPBN(depth, width, 64, 10).to(DEV)
    opt = torch.optim.Adam(base.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    base.train()
    for _ in range(epochs):
        opt.zero_grad(); lossf(base(xtr.float()), ytr).backward(); opt.step()
    base.eval()
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (64,), 10, device=str(DEV))
    _calibrate_cuda(flow, xtr[:256])
    teacher = _capture_activation_means(flow, xte)
    install_ttfs_nodes(flow, S)
    return flow, xtr, ytr, xte, yte, cont, teacher, base


def genuine_logits_eval(flow, x, S):
    with torch.no_grad():
        return TTFSSegmentForward(flow.get_mapper_repr(), S)(x.double())


def staircase_logits_eval(flow, x):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    flow.double().eval()
    with torch.no_grad():
        return flow(x.double())


def per_sample_stats(gen, stair, y):
    """Per-sample agreement between the genuine cascade and the staircase ceiling.

    Returns: genuine_acc, staircase_acc, argmax_agreement (frac test samples with
    identical argmax), logit_corr (mean per-sample Pearson r over the 10 logits),
    and the conditional agreements (where staircase is right / where it is wrong)."""
    gp = gen.argmax(-1)
    sp = stair.argmax(-1)
    agree = (gp == sp).double().mean().item()
    g = gen - gen.mean(-1, keepdim=True)
    s = stair - stair.mean(-1, keepdim=True)
    num = (g * s).sum(-1)
    den = (g.norm(dim=-1) * s.norm(dim=-1)).clamp(min=1e-12)
    corr = (num / den).mean().item()
    gen_acc = (gp == y).double().mean().item()
    stair_acc = (sp == y).double().mean().item()
    s_correct = sp == y
    agree_on_correct = (gp[s_correct] == sp[s_correct]).double().mean().item() \
        if s_correct.any() else float("nan")
    agree_on_wrong = (gp[~s_correct] == sp[~s_correct]).double().mean().item() \
        if (~s_correct).any() else float("nan")
    return dict(gen_acc=gen_acc, stair_acc=stair_acc, argmax_agree=agree,
                logit_corr=corr, agree_on_stair_correct=agree_on_correct,
                agree_on_stair_wrong=agree_on_wrong)


def _ann_logits(base, x):
    base.eval()
    with torch.no_grad():
        return base(x.float()).double()


def run_one(depth, width, S, seed, steps, mix=0.5):
    """Train the STE; diagnose the genuine cascade vs FROZEN lossless targets.

    The right per-sample reference is the FROZEN cold staircase / continuous ANN —
    NOT the trained flow's own staircase. The STE backward constrains only the
    genuine forward, so the trained flow's staircase forward drifts off the genuine
    one (it leaves the regime where staircase==genuine); comparing against that
    moving/broken reference understates the true match. We report BOTH so the
    aggregate-vs-per-sample residual is unambiguous."""
    t0 = time.time()
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, width, seed)
    cold_stair = staircase_logits_eval(flow, xte).clone()   # FROZEN lossless ceiling
    ann = _ann_logits(base, xte)                            # FROZEN continuous ANN target
    cold_gen = genuine_logits_eval(flow, xte, S)
    cold = per_sample_stats(cold_gen, cold_stair, yte)

    install_ttfs_nodes(flow, S)
    ste_train(flow, xtr, ytr, xte, yte, S, base, teacher,
              steps=steps, seed=seed, mix=mix)

    ste_gen = genuine_logits_eval(flow, xte, S)
    # Per-sample agreement against the FROZEN cold staircase (the lossless ceiling).
    ste = per_sample_stats(ste_gen, cold_stair, yte)
    # Aux: agreement against the FROZEN continuous ANN, and the trained-flow staircase.
    trained_stair = staircase_logits_eval(flow, xte)
    gp = ste_gen.argmax(-1)
    ste["agree_vs_ann"] = (gp == ann.argmax(-1)).double().mean().item()
    ste["agree_vs_trained_stair"] = (gp == trained_stair.argmax(-1)).double().mean().item()
    ste["cold_stair_acc"] = (cold_stair.argmax(-1) == yte).double().mean().item()
    return dict(depth=depth, width=width, S=S, seed=seed, cont=cont,
                cold=cold, ste=ste, secs=time.time() - t0)


def report(depths, widths, Ss, seeds, steps):
    import numpy as np
    print("=" * 108)
    print("DIR4: deep+wide hedged staircase-backward STE (mix=0.5) + per-sample residual")
    print(f"  steps={steps}  seeds={list(seeds)}  task=digits(in=64,10cls)")
    print("=" * 108)
    rows = []
    for S in Ss:
        for depth in depths:
            for width in widths:
                seed_rows = [run_one(depth, width, S, s, steps) for s in seeds]
                rows.extend(seed_rows)

                def agg(side, key):
                    return float(np.mean([r[side][key] for r in seed_rows]))
                cont = float(np.mean([r["cont"] for r in seed_rows]))
                secs = float(np.mean([r["secs"] for r in seed_rows]))
                stair = agg("ste", "cold_stair_acc")    # FROZEN lossless ceiling
                cold_gen = agg("cold", "gen_acc")
                ste_gen = agg("ste", "gen_acc")
                gap = stair - ste_gen
                print(f"\n--- d={depth:>2} w={width:>3} S={S} | cont={cont:.3f} "
                      f"staircase(ceiling)={stair:.3f} | {secs:.0f}s/seed ---", flush=True)
                print(f"    COLD genuine={cold_gen:.3f}  "
                      f"argmax_agree(cold,vs frozen stair)={agg('cold','argmax_agree'):.3f}", flush=True)
                print(f"    STE  genuine={ste_gen:.3f}  "
                      f"(gap to frozen staircase {gap:+.3f})", flush=True)
                print(f"    STE  per-sample vs FROZEN staircase: argmax_agree="
                      f"{agg('ste','argmax_agree'):.3f} logit_corr={agg('ste','logit_corr'):.3f}",
                      flush=True)
                print(f"    STE  per-sample vs FROZEN ANN: argmax_agree={agg('ste','agree_vs_ann'):.3f}"
                      f"  | vs TRAINED(moving) staircase: {agg('ste','agree_vs_trained_stair'):.3f}",
                      flush=True)
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "full"
    if which == "quick":
        report(depths=(6, 9), widths=(96, 192), Ss=(16,), seeds=(0,), steps=600)
    elif which == "mid":
        report(depths=(6, 9, 12), widths=(96, 192, 384), Ss=(16,), seeds=(0,), steps=600)
    elif which == "sres":   # resolution sweep at the hard depth
        report(depths=(9,), widths=(96, 384), Ss=(8, 16, 32), seeds=(0,), steps=800)
    else:
        report(depths=(6, 9, 12, 15), widths=(96, 192, 384), Ss=(16,),
               seeds=(0, 1, 2), steps=800)
    print("\nDONE")


if __name__ == "__main__":
    main()
