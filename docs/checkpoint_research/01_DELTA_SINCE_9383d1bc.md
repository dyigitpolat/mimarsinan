# Delta since `9383d1bc` (tuner unification phase 3)

**Baseline commit:** `9383d1bc` — "tuner unification phase 3"  
**Current HEAD:** `05a375af` — configurable conversion-loss weighting + research checkpoint  
**Span:** 296 commits · ~794 files · +106k / −3k lines

This document groups the delta by engineering theme. Status tags:

- **LANDED** — merged to main, default-off byte-identical where applicable, tests present
- **MEASURED** — science result with artifact (finding doc, JSONL, probe)
- **ISOLATED** — probe/study code or branch evidence, not production-wired
- **MEASURED-DEAD** — lever tested and refuted on CIFAR (config cannot close the gap)

---

## 1. Tuning subsystem refactor (LANDED)

The monolithic `SmartSmoothAdaptation` loop became a layered architecture.

| Area | Key paths | Commits (representative) |
|---|---|---|
| Adaptation axes | `src/mimarsinan/tuning/axes/` | `dc06fcc5`…`8707b160` |
| Orchestration services | `tuning/orchestration/{adaptation_driver,rate_scheduler,recovery_engine,acceptance_sensor,checkpoint_guard}.py` | `83beaf0b`…`7116d8de` |
| Uniform rate-tuner seam (E1) | `orchestration/rate_tuner_seam.py` | `d5a07ff4`, `2eed6445` |
| Optimization driver axis (E2) | `orchestration/optimization_driver.py` | `2e6ad754`, `bbe6b673` |
| Calibration pipeline (E3) | `orchestration/calibration_pipeline.py` | `7c02e2d1`, `cf398b07` |
| ConversionPolicy keystone (E4) | `orchestration/conversion_policy.py` | `c4dfaf22`, `c0ad0d49` |
| RampStrategy polymorphism (P1) | `orchestration/ramp_strategy.py` | `1ef5a1d7` |
| TTFS adaptation plan (V7) | `orchestration/ttfs_adaptation_plan.py` | `c21a909b`, `28361249` |
| SpikingModePolicy (V2) | `chip_simulation/spiking_mode_policy.py` | `cd3a70b5`, `5b5db104` |
| DeploymentPlan SSOT (V1) | `pipelining/core/deployment_plan.py` | `5b5db104` |
| Fast ladder (E2 path) | `orchestration/fast_ladder.py` | `66e82f88` |
| Genuine TTFS ramps | `tuners/ttfs_cycle_adaptation_tuner.py` | `8f8094d9`, `88e931ef` |
| Configurable KD loss | `kd_ce_alpha`, `kd_temperature` in `config_schema/defaults.py` | `05a375af` |

**Removed:** `smart_smooth_adaptation.py`, `basic_interpolation.py`, `per_layer_schedule.py`.

**Engineering doc:** [02_TUNING_ENGINEERING.md](02_TUNING_ENGINEERING.md)

---

## 2. Campaign & GPU infrastructure (LANDED)

Entirely new since baseline (~5k LOC under `scripts/campaign/` + `scripts/gpu/`).

| Component | Path | Commit |
|---|---|---|
| Research loop CLI | `scripts/campaign/research_loop.py` | `17ba3954` |
| Scheduler (queue fill + gates) | `scripts/campaign/scheduler.py` | `18eee5dd` |
| Director (backlog growth) | `scripts/campaign/director.py` | `7056f5a9` |
| F-harness matrix | `scripts/campaign/experiment_matrix.py` | `af07db41` |
| Coverage report (E1) | `scripts/campaign/coverage_report.py` | `405f7737` |
| Self-defense guards (A4) | `scripts/campaign/guards.py` | `5606e62b` |
| GPU queue + runner | `scripts/gpu/gpu_queue.py`, `campaign_runner.py` | `5886a60f` |
| GPU lease | `scripts/gpu/gpu_lease.py`, `with_gpu.py` | `e7a192d2` |
| Safe merge | `scripts/campaign/safe_merge.sh` | `336b49eb` |
| Dry-run capacity gate | `mapping/verification/capacity/dryrun.py` + scheduler wiring | `72b091a0` |

**Engineering doc:** [03_CAMPAIGN_OPS.md](03_CAMPAIGN_OPS.md)

---

## 3. Mapping & validity (LANDED + MEASURED)

| Item | Status | Path / finding |
|---|---|---|
| On-chip majority gate (50%) | LANDED | `mapping/verification/onchip_majority.py` — `f71fe92a` |
| Gate-v2 tiered validity (20%/50%, params+MACs) | LANDED | `65f6b3ad` — doc: `docs/research/VALIDITY_AUDIT.md` |
| E2 static validity pre-check at enqueue | LANDED | `6be1897e` |
| E4 scheduling-aware capacity estimate | LANDED | `90dd0162`, `6c8d99a2` |
| Dry-run real packer at enqueue | LANDED | `72b091a0` — [capacity_dryrun_gate.md](../research/findings/capacity_dryrun_gate.md) |
| Residual Tier-0 (host-side add) | LANDED | `206dc2a8` |
| Residual Tier-1 (on-chip merge) | LANDED default-off | `a393eaa8` — [D2_tier1_deployable.md](../research/findings/D2_tier1_deployable.md) |
| D5 on-chip attention/LN characterization | LANDED | `86adc596` — [D5_onchip_attention.md](../research/findings/D5_onchip_attention.md) |
| D3 scheduled-build probe | LANDED | `314a8aea` — [D3_scheduled_build_probe.md](../research/findings/D3_scheduled_build_probe.md) |
| D4 structured pruning knob | LANDED default-off | `0e268cc3`, `f9129558` |
| GAP-1 attribution fix (C3) | LANDED | `1112068b` |
| Cost emit + P2 band | LANDED | `80c0ee8b`, `7dab5338` |

**Engineering doc:** [04_MAPPING_AND_VALIDITY.md](04_MAPPING_AND_VALIDITY.md)

---

## 4. Model vehicles & data (LANDED)

| Vehicle | Path | Commit |
|---|---|---|
| `deep_mlp` (depth probe) | `models/` | `4b2b2da3` |
| `deep_cnn` (configurable depth) | `models/` | `0e9066ef` |
| `lenet5` | `models/` | `4f77f6da` |
| SqueezeNet (B4) | `models/` | `8c584a87` |
| Pretrained bridge (B3/D6) | `models/pretrained_bridge.py` | `6f63f9a8`, `42721b5e` |
| Cheap datasets (FMNIST/KMNIST/SVHN) | `data_handling/` | `39f2689c` |
| θ-cotrain cascaded gate-fix | `mapping/` | `fa4eeaa9` |
| Per-channel theta conv broadcast | `mapping/` | `51faea68` |

---

## 5. Program waves (science + docs)

| Wave | Headline | Key commits |
|---|---|---|
| Wave 1 | A4 guards, B4 SqueezeNet, C2 cost, B1 cross-sim screen | `24622419` |
| Wave 2 | C3 GAP-1, A2/A3 axis collapse, coverage 0.23%→3.75% | `d7a9515e` |
| Wave 3 | E5 Pareto, D3 scheduled-build, F5 draft | `f17f0ed2` |
| Wave 4 | Cost emit, residual-T1 intrinsic 1/T limit | `33afcbeb` |
| Wave 5 | B3 pretrained bridge, D7 percentile-norm baseline | `f9012f7f` |
| Wave 6 | Pretrained validity sweep, ResNet-50 region | `9a5a8936` |
| Wave 7 | F-harness, D5 attention, ImageNet recipe, D6 deploy bridge | `a1790fe2` |
| Wave 8 | F2/F3 wiring, CIFAR breadth generator, ImageNet deploy harness | `423b7eaf` |
| Wave 9 | A2 semantic screen, D4 pruning cost, D2 Tier-1 deployable | `76ec62d0` |
| Wave 10 | U1 NF deploy path, memory-bounded measurement | (F4 docs) |
| Wave 11 | R7 keep-best, target floor, ImageNet val leak fix, FFCV kill-switch | `0b622c82`…`05a375af` |

**Program state docs:** [`PROGRAM_CHECKPOINT_v2.md`](../research/PROGRAM_CHECKPOINT_v2.md), [`ROADMAP.md`](../research/ROADMAP.md)

---

## 6. ImageNet / F4 (MEASURED partial)

| Result | Status | Evidence |
|---|---|---|
| ResNet-50 ANN 71.97% (61 min, 2× GPU) | MEASURED | [F4_imagenet_resnet50.md](../research/findings/F4_imagenet_resnet50.md) |
| Deploy VALID + scheduled-feasible (~16/142 phases) | MEASURED | same |
| Naive/PTC LIF deploy = chance | MEASURED | same §3 |
| Adapted deployed-SNN accuracy | **Unmeasured** | Wave-11 bounded run pending |

**Code:** `scripts/gpu/train_imagenet_fast.py`, `templates/imgnet_resnet50_pretrained.json`

---

## 7. CIFAR conversion fidelity (MEASURED + MEASURED-DEAD)

| Result | Status | Evidence |
|---|---|---|
| First measured CIFAR deployed grid (ffcv unblock) | MEASURED | `91b9d20e`, [cifar_deep_cnn_deploy_measured.md](../research/findings/cifar_deep_cnn_deploy_measured.md) |
| Sync CIFAR-10 deploys (−9pp) but passes gate | MEASURED | `data/00_cifar_baseline_grid.jsonl` |
| Sync CIFAR-100 fails retention gate (−14pp) | MEASURED | same |
| Cascaded CIFAR death-cascade (−23..−30pp) | MEASURED | same |
| Lever T (4→64) | MEASURED-DEAD | `data/01_ttfs_T_sweep.jsonl` |
| Lever α (CE/KD weight) | MEASURED-DEAD | `data/02_ttfs_alpha_q_sweep.jsonl` |
| Lever q (activation clip) | MEASURED-DEAD | same |
| Lever budget/epochs | MEASURED-DEAD | `data/03_budget_sweep.jsonl` |
| LIF default budget = near-chance | MEASURED | `data/02` (`lif_a0.3_base`) |

**Conclusion:** production synchronized proxy-ramp conversion has a **method ceiling** on natural images; config knobs do not close it.

**Engineering doc:** [05_MEASURED_RESULTS.md](05_MEASURED_RESULTS.md) §A–B

---

## 8. Deep-residual QAT fix (ISOLATED + MEASURED)

| Result | Status | Evidence |
|---|---|---|
| QAT (KD+CE) through BN-frozen genuine LIF cascade ≥0.9 dec-fid | MEASURED | [deep_residual_lif_deploy_fix.md](../research/findings/deep_residual_lif_deploy_fix.md) |
| Tier-1 on-chip merge = NO-OP for dec-fid | MEASURED-DEAD | [residual_collapse_fix_study.md](../research/findings/residual_collapse_fix_study.md) |
| Production pipeline does not run this recipe | ISOLATED | `probe_lif_qat_fix_study.py` (off-pipeline) |

**Next work:** [06_NEXT_WORK.md](06_NEXT_WORK.md) P0 tickets

---

## 9. Certification & finish lane (LANDED)

| Item | Commit |
|---|---|
| Phase 2 certification harness + floor book | `fee647a2` |
| Fix B certified fast recipe on matrix templates | `3ab8fb6d` |
| Phase 3 scorecard (6/9 MET) | `30767959` |
| A5 per-FT-pass wall timing | `a8b690bd` |
| A6 ConversionPolicy keystone activation | `c0ad0d49` |

Docs: `docs/certification/`, `docs/CERTIFICATION_PROTOCOL.md`

---

## 10. What did NOT land (honest gaps)

- Genuine-cascade QAT wired into production synchronized fine-tune
- CIFAR sweep rows in `runs/campaign/ledger.jsonl` (ran as direct GPU lease, not harvest path)
- ImageNet adapted LIF deployed accuracy
- `tuning_budget_scale` meaningfully lengthening gradual ramp (bug/limitation documented in [02_TUNING_ENGINEERING.md](02_TUNING_ENGINEERING.md))
- Untracked probes at repo root: `probe_sweep_fold.py`, `probe_sweep_decision_fidelity.py`, `probe_train_resnet.py`

---

## Navigation

| Doc | Purpose |
|---|---|
| [00_INDEX.md](00_INDEX.md) | Entry point, claim-scope matrix |
| [02_TUNING_ENGINEERING.md](02_TUNING_ENGINEERING.md) | Conversion engine for engineers |
| [03_CAMPAIGN_OPS.md](03_CAMPAIGN_OPS.md) | Run reproduction, GPU ops |
| [04_MAPPING_AND_VALIDITY.md](04_MAPPING_AND_VALIDITY.md) | Mapping gates & capabilities |
| [05_MEASURED_RESULTS.md](05_MEASURED_RESULTS.md) | All numbers with trace IDs |
| [06_NEXT_WORK.md](06_NEXT_WORK.md) | Prioritized engineering tickets |
| [repro/](repro/) | Persisted sweep drivers |
