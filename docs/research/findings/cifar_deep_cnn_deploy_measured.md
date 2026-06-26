# Finding: deep_cnn deployed on CIFAR — the first measured 32×32×3 numbers (the ffcv gap closed)

**Status:** MEASURED (single seed s0, production `run.py` pipeline, parity-locked).
**Closes:** the "ZERO cifar in the ledger" gap — the prior 60/60 CIFAR crashes were a missing
`ffcv` dependency, NOT a deployment collapse. Unblocked with `MIMARSINAN_DISABLE_FFCV=1`
(`data_provider.py` kill-switch); the dry-run capacity gate ([[capacity_dryrun_gate]]) correctly
excludes the infeasible d8 cells.

## The measured grid (deep_cnn w16, ttfs_cycle_based, GPU-leased run.py, s0)

| cell | ANN | deployed-SNN | retention | production gate | verdict |
|---|---|---|---|---|---|
| CIFAR-10 d4 synchronized | 0.796 | **0.700** | −9.5pp | **PASS (rc=0)** | ✅ deploys, parity-locked |
| CIFAR-10 d6 synchronized | 0.835 | **0.749** | −8.6pp | **PASS (rc=0)** | ✅ deploys, parity-locked |
| CIFAR-10 d4 cascaded | 0.797 | 0.564 | −23.3pp | FAIL (rc=1) | ❌ death-cascade |
| CIFAR-10 d6 cascaded | 0.833 | 0.603 | −23.0pp | FAIL (rc=1) | ❌ death-cascade |
| CIFAR-100 d4 synchronized | 0.478 | 0.331 | −14.7pp | FAIL (rc=1) | ❌ below retention floor |
| CIFAR-100 d6 synchronized | 0.519 | 0.381 | −13.8pp | FAIL (rc=1) | ❌ below retention floor |
| CIFAR-100 d4 cascaded | 0.481 | 0.176 | −30.5pp | FAIL (rc=1) | ❌ catastrophic |
| CIFAR-100 d6 cascaded | 0.527 | 0.233 | −29.5pp | FAIL (rc=1) | ❌ catastrophic |

`rc=1` = the pipeline's **TTFS Cycle Fine-Tuning retention assertion** fired (deployed < 0.85×ANN);
the deployed value is the genuine spiking-sim accuracy at that step (the assertion message quotes the
exact number), NOT stale — but the run is gate-rejected and never reaches a clean Hard-Core deployment.
CIFAR-100 deployed is real signal (0.33 ≈ 33× chance), just below the production tolerance.

## What it answers

- **SNN deployment WORKS on CIFAR-10 synchronized** — the first measured CIFAR deployed numbers
  (0.70 / 0.75), and the on-chip path is **faithful**: `NF↔SCM per-neuron parity 0.0000% mismatch`,
  `torch↔deployed-sim parity 1.0000`. The deployment is bit-exact; the accuracy cost is real.
- **The wall is the dataset, and it is now LOCATED**: retention degrades monotonically with dataset
  difficulty — MNIST ~lossless → CIFAR-10 −9pp (passes) → **CIFAR-100 −14pp (fails the gate)**. The
  CIFAR-10→CIFAR-100 jump (10→100 classes) is where synchronized deployment crosses the production
  retention floor.
- **Cascaded fails on natural images everywhere** (−23 to −30pp) — the firing-gain death cascade,
  now measured on CIFAR (consistent with the SVHN/FashionMNIST pattern).
- **The loss is in the SPIKING FINE-TUNING step, not the chip**: for the passing cells the trajectory is
  ANN 0.796 → spiking-FT 0.704 → mapped 0.700 (mapping/quant ~lossless). So the accuracy lever is the
  spiking-aware training, not the hardware mapping.

## Caveats (honesty)

- Single seed (s0) — directional, not mean±std. deep_cnn w16 is a SMALL vehicle (CIFAR-10 ANN only
  0.80/0.83, CIFAR-100 0.48/0.52); this measures the deployment GAP, not a SOTA CIFAR number.
- The rc=1 numbers are spiking-FT-step accuracies (gate-rejected), not clean Hard-Core deployments.
- Not yet harvested into `runs/campaign/ledger.jsonl` (ran as a direct GPU-leased sweep, not the
  campaign harvest path).

## Established next lever

The bottleneck (spiking fine-tuning, not mapping) is exactly what the deep-residual QAT recipe addressed
([[research_must_study_solutions]], `deep_residual_lif_deploy_fix.md`): QAT (KD+CE) through the genuine
LIF/TTFS cascade lifted the d8 residual CIFAR-10 ResNet from chance-ish to ≥0.9 decision-fidelity. Applying
that to these deep_cnn cells is the grounded path to push CIFAR-10 synchronized past −9pp and pull
CIFAR-100 synchronized back above the retention floor.
