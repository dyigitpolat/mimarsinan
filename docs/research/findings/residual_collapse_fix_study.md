# Deep-residual genuine-spiking collapse: SOLUTION study

Branch: `residual-collapse-fix-study` (NOT merged to main). Prototype:
`probe_residual_fix_study.py` (reuses the production SSOT in
`probe_residual_genuine_spiking_sweep.py`: `deploy / measure / make_task /
ResidualStack / train`).

## Shared probe (locked, byte-identical across all fixes)

`ResidualStack` depth **20**, IN=16, W=24, NC=4, T=16, N_EVAL=200, Adam lr=3e-3
150 epochs. Two collapse seeds are reported:
- **seed 0** — all three genuine modes collapse moderately (the iteration probe).
- **seed 2** — TTFS collapses catastrophically (sync **0.0%**), the rescue stress test.

Retention = `deployed_HCM_top1 / ann_top1`. ANN(seed0)=0.905, ANN(seed2)=0.995.

Command form:
`PYTHONPATH=src:spikingjelly env/bin/python probe_residual_fix_study.py <fix> 20 <seed> <T> <modes>`

## Retention table (baseline -> with-fix), deployed/ann on the shared probe

### seed 0 (ANN 0.905), T=16

| fix | lif | ttfs_cascaded | ttfs_sync | command |
|---|---|---|---|---|
| baseline | 83.4% | 50.3% | 49.7% | `... baseline 20 0 16` |
| (2) QAT spiking-aware train | **89.0%** | **84.5%** | **87.3%** | `... qat 20 0 16` |
| (8) revive-then-refine (300 steps) | 87.3% | 84.5% | 72.9% | `... revive_refine 20 0 16` |
| (4) per-depth gain correction | 83.4% | 47.5% | 59.7% | `... gain 20 0 16` |
| (5) scale-aware + STRONG DFQ | (std) | 28.7% | 61.9% | `... scale_dfq_strong 20 0 16` |
| (7) higher S, T=32 | 83.4% | 66.3% | 40.3% | `... highT 20 0 32` |
| (7) higher S, T=64 | 83.4% | 6.1% | 23.2% | `... highT 20 0 64` |
| (7) higher S, T=128 | 83.4% | 33.1% | 60.8% | `... highT 20 0 128` |

### seed 2 (ANN 0.995), T=16 — catastrophic-collapse rescue test

| fix | ttfs_cascaded | ttfs_sync | command |
|---|---|---|---|
| baseline | 14.6% | **0.0%** | `... baseline 20 2 16` |
| (2) QAT spiking-aware train | **100.0%** | **100.0%** | `... qat 20 2 16` |

(lif at seed 2 is already 100.0% baseline — that seed's collapse is TTFS-specific.)

### fix (1) on-chip residual merge — dedicated probe `probe_residual_sync_ttfs.py 20` (3-seed avg)

| arch | sync HCM retention |
|---|---|
| plain 20-layer (no residual) | 31.5% |
| residual_T0 (host rate-add, the baseline) | **75.0%** |
| residual_T1 (on-chip merge) | 45.5% |

## Verdict per candidate

- **(2) QAT / spiking-aware training = THE FIX.** Train the weights through the
  genuine differentiable spike forward (`chip_aligned_segment_forward` for LIF,
  `TTFSSegmentForward` for TTFS) with a frozen-teacher -> genuine blend ramp
  (200 steps, lr 1.5e-3, ramp 0.5). Lifts every mode: seed-0 +5.6 / +34.2 /
  +37.6 pp; and on the catastrophic seed-2 it takes a **0.0% / 14.6% dead
  deployment to 100.0% retention**. This is the load-bearing fix F4 predicted on
  ResNet-50; here it is MEASURED on a deep residual backbone. Best single fix.
- **(8) revive-then-refine** ties QAT on cascaded but the longer/faster ramp
  overshoots the sync basin (72.9% < QAT 87.3%). QAT's shorter ramp is better.
- **(1) on-chip residual merge: REFUTED as a fix.** It makes sync retention
  WORSE (75.0% host-add -> 45.5% on-chip), the D2 1/T-per-merge re-quant
  compounding across 20 blocks. The Tier-0 host rate-add (already the baseline)
  is the better residual handling; the residual skip itself HELPS (plain-20
  31.5% < residual-T0 75.0%).
- **(7) higher S: not a reliable fix.** Erratic single-seed (T32 66% -> T64 6%
  -> T128 33% for cascaded) and LIF is completely T-invariant (lif flat 83.4%
  across all T — the lif collapse is not an S problem). Raising deploy-S alone
  does not solve the deep-residual collapse.
- **(4) gain correction: near-noise / mildly negative** (cascaded 50.3->47.5),
  matching the prior cascaded verdict; cascaded-only by physics.
- **(5) scale-aware + strong DFQ: mixed** — helps sync (+12pp) but the
  aggressive first-moment matching destabilizes cascaded (50.3 -> 28.7). Not a
  clean standalone fix; the standard q=0.99 DFQ in the baseline is already near
  its useful limit.

## Honest status

A single fix DID reach >=0.9 retention: **QAT reaches 1.00 retention on the
catastrophic seed-2 (both TTFS modes), rescuing a fully-dead (0.0%)
deployment.** On the moderate seed-0 QAT reaches 0.89 (lif) / 0.845 (cascaded) /
0.873 (sync) — close to but not >=0.9, residual gap ~1.5-5.5 pp, plausibly
closable with more genuine-refine steps / multi-seed-tuned ramp. QAT is the
clear, robust, highest-leverage solution; every PTC-only calibration fix
(gain / higher-S / residual-merge / strong-DFQ) is at best partial and several
are negative.
