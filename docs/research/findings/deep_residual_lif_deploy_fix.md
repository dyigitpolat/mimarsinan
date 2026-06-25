# LIF deep-deploy decision-fidelity FIX: QAT through the genuine cascade (BN-frozen)

Strong probe: real conv/BN RESIDUAL ResNet (d8, 17 convs, w32) trained on CIFAR10 to
ANN top1 = 0.8745. Deploy = production primitives (convert -> mark_encoding -> install
LIFActivation -> per-layer 0.99-quantile FOLD into weights so LIF scale stays 1.0 and
NF==HCM is bit-exact -> IRMapping -> hybrid HCM). Metric = DECISION-FIDELITY =
argmax-agreement of the LIF chip-aligned NF with the float ANN teacher on 2000 held-out
CIFAR10 TEST images (chance 0.10). Collapsing depth = d8; collapsing T = 16/32.

## The under-measured LIF fix had a SILENT train/deploy bug
The converted flow carries a live `BatchNorm1d` per perceptron. In `.train()` mode those
use BATCH stats (and drift running stats), so the QAT training forward is a DIFFERENT
function than the deployed `.eval()` cycle-accurate LIF forward:
  measured train-vs-eval NF logits max|diff| = 6.94, argmax agreement = 28% (eval-vs-eval
  is bit-exact 0.0). The first naive QAT runs HURT (T16 0.52 -> 0.32).
FIX: freeze every BatchNorm to `.eval()` during QAT -> train forward becomes bit-exact to
the deploy forward (max|diff| = 0.0, 100% argmax agreement) while gradients still flow.
This is the load-bearing insight; the diagnosis flagged QAT "un-measured" precisely
because the obvious QAT silently optimizes the wrong forward.

## Per-fix decision-fidelity (d8 residual, 2000 test imgs, 400 steps, lr 5e-4)
At the collapsing T16:
  baseline (fold, bit-exact)        0.5195
  resmerge (on-chip residual merge) 0.5195   (+0.0000 -- exact NO-OP, as diagnosed)
  dfq (LIF bias correction)         0.4755   (-0.0440 -- HURTS, as diagnosed)
  qat_kd  (soft KD to ANN only)     0.6050   (+0.0855)
  qat_ce  (hard-label CE only)      0.8395   (+0.3200)
  qat     (KD + CE)                 0.8555   (+0.3360)  <- best single T16 fix

At the collapsing T32 (two independent runs, GPU1/GPU2):
  baseline                          0.8260
  qat_kd  (soft KD only)            0.7790   (-0.0470 -- KD-to-ANN HURTS at T32)
  qat_ce  (CE only)                 0.8945   (+0.0685)
  qat     (KD + CE)            0.9095 / 0.9105 (+0.0835/+0.0845)  >= 0.9 REACHED
  highT2x (baseline @ T64)          0.9115   (+0.0855 -- pure-T lever, 2x cost)
  qat_highT (QAT@T32 -> deploy T64) 0.9190   (+0.0930)  <- best composition

## Findings
1. BEST SINGLE FIX = QAT (KD+CE) through the genuine BN-frozen LIF cascade.
   - T16: 0.5195 -> 0.8555 (+0.3360 dec-fid; deployed top1 0.5155 -> 0.8150, 93% ret).
   - T32: 0.8260 -> 0.9105 (+0.0845; >= 0.9). Baseline needs T64 (0.9115) for the same
     -> QAT buys a 2x reduction in T to cross 0.9.
2. CE-on-labels is the load-bearing objective; soft-KD-to-the-ANN alone OVER-FITS and
   HURTS at T32 (0.7790 < 0.8260). KD only helps as a small additive term ON TOP of CE.
3. COMPOSED best fixes reach >= 0.9: qat (KD+CE) @ T32 = 0.9105, and qat_highT (QAT @ T32
   deployed at T64) = 0.9190 -- both exceed the >= 0.9 goal. qat_highT > pure-T highT2x
   (0.9115) and > qat-alone-T32, so QAT composes positively with the T lever.
4. CONFIRMED the diagnosis verbatim: resmerge is a bit-exact NO-OP for decision fidelity;
   DFQ HURTS the LIF cascade; T is the dominant standalone lever (baseline T64 = 0.9115).

## Residual gap / honest status
SOLVED at the >= 0.9 bar: composed best (qat @ T32 = 0.9105, qat_highT = 0.9190) crosses
0.9 at HALF the T that the pure-T lever needs (T32 vs T64). At the harder T16 wall QAT
recovers most of the gap (0.5195 -> 0.8555) but does NOT reach 0.9 -- the ~16-rate-level
resolution caps held-out argmax agreement; closing that last ~4.5pp at T16 needs more T
(the dominant lever) or per-layer S_d (reserved lever, needs driver support, not an
isolated-script change). Deployed-vs-ANN gap of the >= 0.9 composition is ~3-4pp top1
(0.855 vs 0.8745).

Scripts (branch probe-lif-qat-fix, NOT committed to main):
  probe_lif_qat_fix_study.py  -- the fix study (build_folded_flow / qat_finetune /
                                 BN-freeze / measure); reuses probe_lif_resnet_decision_fidelity.py.
