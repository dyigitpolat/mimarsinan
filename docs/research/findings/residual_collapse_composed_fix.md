# Deep-residual genuine-spiking collapse: composed-fix study

Composes the highest-leverage fixes from the single-fix sweep
(`residual_collapse_fix_study.md`) into one solution and measures composed
retention vs baseline + best-single on the byte-identical shared probe.

## Shared probe (identical across every fix)

`ResidualStack` depth=20, IN=16 W=24 NC=4, T=16, N_EVAL=200, seeds {0,1,2}.
Retention = deployed_HCM_top1 / ann_top1, through the production deploy SSOT
(`probe_residual_genuine_spiking_sweep.deploy`). Driver: `probe_residual_fix_study.py`.

```
PYTHONPATH=src:spikingjelly env/bin/python probe_residual_fix_study.py <fix> 20 <seed> <T> [modes]
```

## The composition

QAT (train the residual weights through the genuine differentiable spike
forward) is the single load-bearing pillar. The single-fix sweep left a residual
1.5-5.5pp gap on the moderate seed-0 (largest on cascaded) and, separately, plain
QAT *destabilized* cascaded on seed-1 (72.4%). The composition keeps QAT as the
pillar and selects a **per-mode genuine-refine recipe by physics** — the only
lever that composed positively on top of QAT:

| mode      | recipe                              | rationale |
|-----------|-------------------------------------|-----------|
| lif       | standard QAT (200 steps, lr 1.5e-3, ramp 0.5) | rate code already near-lossless under QAT |
| cascaded  | aggressive QAT (600 steps, lr 3e-3, ramp 0.25) | the death-cascade firing-gain deficit needs more genuine budget + a faster teacher->genuine handoff |
| sync      | longer-refine QAT (400 steps, lr 1.5e-3, ramp 0.3) | the pure-genuine tail closes the per-merge 1/T re-quant compounded across blocks |

Driver fix name: `qat_best_per_mode`.

## Composed retention (3-seed average)

| fix             | lif    | ttfs_cascaded | ttfs_sync |
|-----------------|--------|---------------|-----------|
| baseline        | 89.3 % | 49.8 %        | 41.9 %    |
| best-single QAT | 94.6 % | 85.6 %        | 95.1 %    |
| **COMPOSED**    | 94.6 % | **93.0 %**    | **96.4 %**|

Per seed (lif / cascaded / sync):

```
                       seed0 (ANN .905)   seed1 (ANN .960)   seed2 (ANN .995)
baseline:            83.4/50.3/49.7      84.4/84.4/76.0     100./14.6/ 0.0
best-single QAT:     89.0/84.5/87.3      94.8/72.4/97.9     100./100./100.
COMPOSED best/mode:  89.0/86.7/92.3      94.8/92.2/96.9     100./100./100.
```

## Verdict: IMPROVED (cascaded crosses 0.9 on average; seed-0 still short)

- **Cascaded** is the headline win: best-single 85.6% -> composed **93.0%** avg
  (+7.4pp), now over 0.9. The largest per-seed rescue is seed-1 cascaded
  72.4% -> 92.2% (+19.8pp): plain QAT's gentle ramp destabilized the cascaded
  death-cascade that seed; the aggressive recipe fixes it. Deterministic
  (re-runs bit-identical).
- **Sync** 95.1% -> 96.4% avg, clears 0.9 on every seed (seed-0 87.3% -> 92.3%).
- **lif** 94.6% avg, unchanged by the composition (QAT alone is sufficient).
- **Catastrophic seed-2** stays 100% on all three modes — the composition does
  not break the severe-collapse rescue.

## Residual gap (honest)

The remaining gap to lossless lives entirely on **seed-0**: lif 89.0%,
cascaded 86.7% (~3.3pp short), sync 92.3%. On seed-0 the deployed top1 tracks the
genuine NF forward tightly (cascaded dep 0.785 vs the real HCM is the ceiling),
so the cap is the **genuine-cascade forward accuracy itself**, not a
deploy-time mismatch — i.e. the death-cascade is not fully trained out at this
width (W=24) within the QAT budget. The cascaded mode is the deepest residual:
it is the one mode where the single-spike timing cascade compounds the
firing-gain deficit block-over-block.

## PTC-side levers that did NOT compose (measured, negative/neutral on top of QAT)

| composed lever                         | cascaded | note |
|----------------------------------------|----------|------|
| qat_gain (per-depth gain corr)         | 83.4 %   | == QAT; gain is near-noise once weights are in the basin (readout absorbs static gains) |
| qat_highT (deploy T=32)                | 82.9 %   | slightly DOWN; sync also hurt |
| qat_more_highT (deploy T=32)           | 85.1 %   | T-raise helps cascaded a hair but HURTS sync (85.6%) — T is not a free composed knob here |

The decoupled higher-S deploy and the per-depth gain correction — both predicted
useful by the cascaded-TTFS research — are confirmed **non-additive on top of
QAT for deep residual**: once QAT moves the weights into the spiking basin, the
PTC-side calibration fixes have nothing left to correct, and the T-raise trades
cascaded vs sync.

## Next composition to try

To close the seed-0 cascaded ~3.3pp residual without trading sync:
1. **Width**: the seed-0 cap looks capacity-bound (W=24); widen to W=48 and
   re-measure whether cascaded QAT then reaches lossless — isolates capacity
   from optimization.
2. **Staircase-backward STE inside the cascaded QAT loop** (fix 6) as the
   *gradient*, not a standalone PTC fix: replace the genuine cascade backward in
   `_qat_adapt_flow` with `mix*staircase + (1-mix)*genuine` so the deep
   high-block cascaded gradient is well-conditioned — the one untested
   QAT-gradient composition (the refuted-standalone STE re-purposed as the QAT
   backward).
3. **Per-block KD** (fix 3): add a block-wise spike-train distillation term to
   the cascaded QAT loss so each residual block's genuine output is supervised,
   not just the final logits — targets the compounding-across-blocks deficit
   directly.

## Reproduce

```
PYTHONPATH=src:spikingjelly env/bin/python probe_residual_fix_study.py qat_best_per_mode 20 0 16
PYTHONPATH=src:spikingjelly env/bin/python probe_residual_fix_study.py qat_best_per_mode 20 1 16
PYTHONPATH=src:spikingjelly env/bin/python probe_residual_fix_study.py qat_best_per_mode 20 2 16
```

Branch `residual-collapse-compose` (off the single-fix study commit). Isolated
prototype scripts only; NOT merged to main.
