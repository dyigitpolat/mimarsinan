# Measured Results — Traceable Tables

Every number cites a source file. Do not cite numbers from this doc without checking the source.

**Digit-dataset science (MNIST/FMNIST/KMNIST):** see [`PROGRAM_CHECKPOINT_v2.md`](../research/PROGRAM_CHECKPOINT_v2.md) §2 and [`WS3_depth_firing_gain.md`](../research/findings/WS3_depth_firing_gain.md) — not duplicated here.

---

## §A — CIFAR baseline grid

**Source:** [`data/00_cifar_baseline_grid.jsonl`](data/00_cifar_baseline_grid.jsonl)  
**Vehicle:** `deep_cnn` w16, `ttfs_cycle_based`, production `run.py`, seed s0, T=4  
**Finding:** [cifar_deep_cnn_deploy_measured.md](../research/findings/cifar_deep_cnn_deploy_measured.md)

| cell | ANN | deployed | retention | gate |
|---|---|---|---|---|
| CIFAR-10 d4 synchronized | 0.796 | 0.700 | −9.5pp | PASS (rc=0) |
| CIFAR-10 d6 synchronized | 0.835 | 0.749 | −8.6pp | PASS (rc=0) |
| CIFAR-10 d4 cascaded | 0.797 | 0.564 | −23.3pp | FAIL (rc=1) |
| CIFAR-10 d6 cascaded | 0.833 | 0.603 | −23.0pp | FAIL (rc=1) |
| CIFAR-100 d4 synchronized | 0.478 | 0.331 | −14.7pp | FAIL (rc=1) |
| CIFAR-100 d6 synchronized | 0.519 | 0.381 | −13.8pp | FAIL (rc=1) |
| CIFAR-100 d4 cascaded | 0.481 | 0.176 | −30.5pp | FAIL (rc=1) |
| CIFAR-100 d6 cascaded | 0.527 | 0.233 | −29.5pp | FAIL (rc=1) |

**Per-step loss localization (CIFAR-10 d4 sync):**

| Step | Metric |
|---|---|
| ANN (Activation Analysis) | 0.7956 |
| TTFS Cycle Fine-Tuning | 0.7044 |
| Weight Quantization | 0.7019 |
| Normalization Fusion | 0.7005 |
| Soft Core Mapping | 0.7004 |

Chip path: NF↔SCM 0.0000% mismatch, torch↔sim 1.0.

---

## §B — Refuted config levers (CIFAR-10 d4 synchronized)

Base: same vehicle as §A unless noted.

### B.1 Temporal resolution T

**Source:** [`data/01_ttfs_T_sweep.jsonl`](data/01_ttfs_T_sweep.jsonl) (+ T=4 from §A)

| T | retention |
|---|---|
| 4 (baseline) | −9.5pp |
| 8 | −8.7pp |
| 16 | −9.4pp |
| 32 | −10.1pp |
| 64 | −9.5pp |

**Verdict:** REFUTED — flat across T=4..64.

### B.2 CE/KD objective α and calibration q

**Source:** [`data/02_ttfs_alpha_q_sweep.jsonl`](data/02_ttfs_alpha_q_sweep.jsonl)

| variant | retention |
|---|---|
| α=0.3 (base) | −8.8pp |
| α=0.6 | −9.6pp |
| α=1.0 | −9.8pp |
| q=1.0 | −8.9pp |
| α=1.0 + q=1.0 | −10.6pp |

**Verdict:** REFUTED — flat-to-worse.

### B.3 Fine-tune training budget

**Source:** [`data/03_budget_sweep.jsonl`](data/03_budget_sweep.jsonl)

| variant | ANN | deployed | retention | gradual_s |
|---|---|---|---|---|
| ttfs budget=4 ep=20 | 0.819 | 0.719 | −10.0pp | 5.5s |
| ttfs budget=16 ep=40 | 0.833 | 0.728 | −10.5pp | 7.4s |
| ttfs budget=40 ep=60 | 0.838 | 0.711 | −12.7pp | 7.4s (rc=1) |
| lif budget=16 ep=40 | 0.829 | 0.694 | −13.5pp | 10.3s (rc=1) |

**Verdict:** REFUTED — deployed pinned ~0.71–0.73 while ANN rose; retention worsened. Budget knob barely lengthened ramp (5.5→7.4s).

### B.4 LIF at default budget

**Source:** [`data/02_ttfs_alpha_q_sweep.jsonl`](data/02_ttfs_alpha_q_sweep.jsonl) (`lif_a0.3_base`)

| mode | ANN | deployed | retention |
|---|---|---|---|
| LIF default | 0.797 | 0.141 | −65.6pp (near chance) |

LIF at budget=16 ep=40 recovers to 0.694 (−13.5pp) but still far from near-lossless.

---

## §C — Deep-residual QAT (off-pipeline)

**Source:** [deep_residual_lif_deploy_fix.md](../research/findings/deep_residual_lif_deploy_fix.md)  
**Vehicle:** conv/BN residual ResNet d8 w32, CIFAR-10, ANN 0.8745  
**Metric:** decision-fidelity (argmax agreement with float ANN on 2000 test images)

### T16 (collapsing regime)

| fix | dec-fid | Δ vs baseline |
|---|---|---|
| baseline (fold) | 0.5195 | — |
| resmerge (Tier-1) | 0.5195 | +0.0000 (NO-OP) |
| qat_kd | 0.6050 | +0.0855 |
| qat_ce | 0.8395 | +0.3200 |
| **qat (KD+CE)** | **0.8555** | **+0.3360** |

### T32 (≥0.9 target)

| fix | dec-fid |
|---|---|
| baseline | 0.8260 |
| qat (KD+CE) | 0.9105 |
| qat_highT (QAT@T32 deploy@T64) | 0.9190 |

**Load-bearing insight:** BN must be frozen to `.eval()` during QAT (train/eval max|diff| 6.94 → 0.0). CE-on-labels is load-bearing; KD alone hurts at T32.

**Scripts:** `probe_lif_qat_fix_study.py`, `probe_lif_resnet_decision_fidelity.py`

---

## §D — ImageNet F4

**Source:** [F4_imagenet_resnet50.md](../research/findings/F4_imagenet_resnet50.md)

| Claim | Value | Status |
|---|---|---|
| ANN top-1 (official val) | 71.97% in 61.3 min (2× RTX PRO 6000) | MEASURED |
| Deploy validity | VALID (~66.6% on-chip) | MEASURED |
| Scheduled reachability | ~16 reprogram + ~142 reuse phases | MEASURED |
| Naive/PTC LIF deploy | Chance (0–7.8% at T≤64) | MEASURED |
| **Adapted deployed-SNN acc** | — | **NOT MEASURED** |

---

## §E — Digit-dataset science (pointer)

Summarized in v2; key headlines only:

| Claim | Worst case | Source |
|---|---|---|
| Sync TTFS lossless on valid vehicles | ≤3.06pp (FMNIST d10) | v2 §2.1 |
| Cascaded death-cascade dual-axis | d10 FMNIST casc→ANN 20.97pp | WS3 |
| No firing-gain rescue on valid convnet | θ-cotrain broken on Conv2D | v2 §2.3 |
| Five-mode landscape (mmixcore MNIST) | Cascaded −0.84pp vs sync sibling | v2 §2.4 |
| Honest deep_cnn coverage | 6/8 cells (SVHN untested) | v2 §2.6 |

Full tables: [`WS3_depth_firing_gain.md`](../research/findings/WS3_depth_firing_gain.md) (2277 lines — link, do not copy).

---

## §F — Capacity dry-run gate

**Source:** [capacity_dryrun_gate.md](../research/findings/capacity_dryrun_gate.md)

| Metric | Value |
|---|---|
| Campaign runs crashed at mapping (pre-fix) | 126/1267 (~10%, ~40 GPU-h) |
| Crash configs rejected by dry-run | 60/60 |
| Done configs admitted | 80/80 |
| False rejections | 0 |

---

## §G — Synthesis

| Layer | CIFAR status |
|---|---|
| Chip lowering (NF↔SCM, torch↔sim) | Bit-exact |
| Mapping / quant / norm-fusion | ~lossless (~0.3pp) |
| Spiking conversion fine-tune | **−9..−14pp ceiling** (method, not config) |
| Off-pipeline QAT recipe | **≥0.9 dec-fid proven** on residual ResNet |

The conversion gap is a **method ceiling** on the production synchronized proxy-ramp path, not a config bug.
