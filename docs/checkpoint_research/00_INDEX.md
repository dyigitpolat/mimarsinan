# Research Checkpoint — Engineering Handoff

**Date:** 2026-06-27 · **Baseline:** `9383d1bc` (tuner unification phase 3) · **HEAD:** `05a375af`  
**Purpose:** durable engineering documentation for the ~296-commit delta since phase 3. Every number below is **MEASURED** (artifact in `data/` or a named finding doc) or explicitly flagged open.

**Companion docs (do not duplicate here):**

- [`docs/research/PROGRAM_CHECKPOINT_v2.md`](../research/PROGRAM_CHECKPOINT_v2.md) — program-wide science state (MNIST-era claims, validity tiers, death-cascade law)
- [`docs/research/ROADMAP.md`](../research/ROADMAP.md) — dependency-ordered forward plan
- [`docs/research/HYPERVOLUME.md`](../research/HYPERVOLUME.md) — deployment hypervolume axis SSOT

---

## TL;DR

Since phase 3, the codebase gained a **full tuning refactor** (axes, orchestration, genuine-cascade paths), **campaign automation** (scheduler/director/GPU queue + capacity gates), **mapping capabilities** (Tier-1 residual, D5 attention, scheduling-aware capacity), and **ImageNet reachability** (71.97% ANN, deploy VALID).

The headline **research result** from the latest fidelity push:

> For `deep_cnn` on CIFAR, production deployed-SNN accuracy **plateaus at ~0.71–0.73** (−9 to −13pp below ANN). This gap is **robust to every config lever tested** (T, CE/KD α, calibration q, budget/epochs). Chip lowering remains **bit-exact** (NF↔SCM 0.0%). The loss is entirely in the **spiking conversion fine-tune**, which uses a value-domain proxy ramp that **force-disables** the genuine-cascade QAT toolkit for synchronized schedules.

The **proven near-lossless recipe** (QAT KD+CE through BN-frozen genuine cascade, 400+ steps) exists in `probe_lif_qat_fix_study.py` but is **off-pipeline**. Wiring it into production is the primary engineering frontier.

**Landed this session (on main):**

1. **`72b091a0`** — dry-run capacity gate (~10% GPU waste eliminated)
2. **`91b9d20e`** — ffcv unblock → first measured CIFAR deployed grid
3. **`05a375af`** — configurable `kd_ce_alpha` / `kd_temperature` (default-preserving)

---

## Claim-scope matrix

Reconciles [`PROGRAM_CHECKPOINT_v2.md`](../research/PROGRAM_CHECKPOINT_v2.md) with CIFAR findings. v2 claims hold within their stated scope; CIFAR extends the frontier.

| Claim | Scope | Status | Evidence |
|---|---|---|---|
| Sync TTFS ≤3pp lossless | Valid vehicles, digit datasets (MNIST/FMNIST/KMNIST) | **Holds** | v2 §2.1, WS3 |
| Cascaded death-cascade dual-axis law | Valid convnets, digit datasets | **Holds** | [WS3_depth_firing_gain.md](../research/findings/WS3_depth_firing_gain.md) |
| Chip lowering bit-exact | All valid vehicles incl. CIFAR | **Holds** | CIFAR grid NF↔SCM 0.0000% |
| No firing-gain rescue on valid convnet | MNIST deep_cnn d6+ | **Holds** | v2 §2.3 |
| Production sync conversion retains ANN accuracy | CIFAR natural images, plain `deep_cnn` | **Fails** (−9..−14pp ceiling) | [data/00_cifar_baseline_grid.jsonl](data/00_cifar_baseline_grid.jsonl) |
| Config levers (T, α, q, budget) close CIFAR gap | CIFAR-10 d4 synchronized | **Refuted** | [data/01_*.jsonl](data/) … [data/03_*.jsonl](data/) |
| QAT through genuine cascade ≥0.9 dec-fid | Residual ResNet d8, off-pipeline | **Proven** | [deep_residual_lif_deploy_fix.md](../research/findings/deep_residual_lif_deploy_fix.md) |
| ImageNet adapted deployed-SNN acc | ResNet-50 | **Unmeasured** | [F4_imagenet_resnet50.md](../research/findings/F4_imagenet_resnet50.md) |

**Intellectual spine:** mapping is solved; **production conversion method** is the bottleneck on hard datasets.

---

## Where accuracy is lost (CIFAR d4 synchronized)

Per-step metric trajectory (production `run.py`):

```
ANN (Pretraining)              0.7956
TTFS Cycle Fine-Tuning         0.7044   ← entire loss here
Weight Quantization            0.7019
Normalization Fusion           0.7005
Soft Core Mapping (spiking)    0.7004   ← NF↔SCM 0.0%, torch↔sim 1.0
Hard Core Mapping              (rc=0)
```

---

## Document map

| File | Contents |
|---|---|
| [01_DELTA_SINCE_9383d1bc.md](01_DELTA_SINCE_9383d1bc.md) | Commit-grouped changelog (296 commits → 10 themes) |
| [02_TUNING_ENGINEERING.md](02_TUNING_ENGINEERING.md) | Proxy vs genuine, adaptation trifecta, config SSOT, test contracts |
| [03_CAMPAIGN_OPS.md](03_CAMPAIGN_OPS.md) | Reproduction contract, campaign topology, GPU gotchas |
| [04_MAPPING_AND_VALIDITY.md](04_MAPPING_AND_VALIDITY.md) | Capacity gates, Tier-1 residual, D5 attention, validity tiers |
| [05_MEASURED_RESULTS.md](05_MEASURED_RESULTS.md) | Consolidated numbers with trace IDs |
| [06_NEXT_WORK.md](06_NEXT_WORK.md) | Prioritized engineering tickets + acceptance criteria |
| [data/](data/) | Raw JSONL experiment records |
| [data/SCHEMA.md](data/SCHEMA.md) | JSONL field definitions |
| [repro/](repro/) | Persisted sweep drivers |

---

## Quick reproduction

From project root with `env` activated:

```bash
export MIMARSINAN_DISABLE_FFCV=1
./docs/checkpoint_research/repro/run_single_smoke.sh
```

Full sweeps: [repro/README.md](repro/README.md).

---

## Related finding docs

- [capacity_dryrun_gate.md](../research/findings/capacity_dryrun_gate.md)
- [cifar_deep_cnn_deploy_measured.md](../research/findings/cifar_deep_cnn_deploy_measured.md)
- [deep_residual_lif_deploy_fix.md](../research/findings/deep_residual_lif_deploy_fix.md)
- [F4_imagenet_resnet50.md](../research/findings/F4_imagenet_resnet50.md)
- [WS_permode_landscape.md](../research/findings/WS_permode_landscape.md)
