# mimarsinan Research Program Checkpoint

*Author's checkpoint — ANN→SNN deployment fidelity under genuine single-spike TTFS.*
*Scope is honest: every number below is from the science-valid ledger; confounds are
named inline, not buried.*

---

## 1. Vision & Mission

The publication mandate is **automatic genericity at breadth**: a single deployment
pipeline that takes an arbitrary trained torch model and lowers it onto a spiking
neuromorphic chip *without per-model hand-tuning*, and does so faithfully across a
matrix of **architectures × datasets × difficulty regimes**. The chip lowering must be
provably lossless (bit-exact mapping), so that whatever accuracy is lost is attributable
to one understood place — the trainable ANN→single-spike fold — and nowhere else.

The intended publication shape is two-regime:

- **Pretrained-near-SOTA headline.** Bridge a near-SOTA pretrained backbone
  (timm/torchvision ResNet-50 / ViT-B) onto the chip and report the deployed accuracy
  gap to the floating-point model. This is the "does it survive on a real model" claim.
- **From-scratch profiling.** Train models from scratch across the matrix to *profile*
  where and why fidelity breaks (depth, dataset margin, architecture), with the firing
  mechanics fully explained rather than reported as a black-box number.

Both regimes are to be reported with **published-baseline head-to-heads (RMP / QCFS /
percentile-norm), confidence intervals, ablations, and an energy/latency Pareto** — the
rigor a venue expects, not just a single accuracy table.

This checkpoint records what is **validated** today (the cheap from-scratch diagnostic
tier, on a strict validity gate) and what remains **unbuilt** (the rigor foundation and
the expensive pretrained-SOTA frontier).

---

## 2. Where We Stand — Honest One-Paragraph Verdict

The **current-vehicle program is complete on the cheap-diagnostic tier**: on a strict
on-chip-majority validity gate, we have a clean, reproducible characterization of genuine
single-spike TTFS deployment fidelity across depth, dataset, architecture, and the five
spiking modes — synchronized TTFS is the lossless deep default, cascaded TTFS carries a
real dual-axis (depth × dataset) death-cascade deficit, and no firing-gain rescue lever
survives on a valid convnet. The **publication frontier is unbuilt**: there are no
published-baseline head-to-heads, no residual/transformer backbones for deep-trainable
points, no CIs/ablations bundle, no energy/latency Pareto allocator, and no pretrained
near-SOTA bridge (ResNet-50 / ViT-B, CIFAR→ImageNet). The science is honest and bounded;
the venue bar is not yet met.

---

## 3. Validated Findings

All numbers below are from the **86 science-valid ledger rows** (158 total; 72 excluded —
70 `INVALID_host_majority` deep_mlp w64 rows + 2 `cluster=VALIDITY` bookkeeping rows).
Valid vehicles: **deep_cnn** (98.5–99.5% on-chip), **lenet5** (99.1%), **mlp_mixer_core**
(90.1%). The chip lowering is bit-exact everywhere (NF↔SCM per-neuron 0.0000%, torch↔sim
parity 1.0): **all loss lives in the trainable ANN→single-spike TTFS fold.**

### 3.1 Synchronized TTFS is the lossless deployment default (the positive headline)

Synchronized `ttfs_cycle_based` holds the ANN ceiling at every measured depth and dataset
on valid vehicles. It is the **unconditional deep-model default.**

| Vehicle / dataset | sync → ANN gap | Verdict |
|---|---|---|
| deep_cnn MNIST, every depth d4–d12 | ≤ 0.18pp (d12 sync 0.9917 vs ANN 0.9887 = **+0.30pp**, sd 0.07pp) | Lossless |
| deep_cnn FMNIST | d4 1.86 → d6 2.78 → d8 2.98 → d10 3.06pp | MET throughout |
| deep_cnn KMNIST | d4 1.99 → d6 0.80 → d8 0.55 → d10 0.40pp | MET |
| lenet5 MNIST (full-test SCM) | sync 0.9891 vs ANN 0.9913 = **+0.22pp** | Lossless |

Every synchronized cell on a valid vehicle is within **≤ 3.06pp** of its ANN. (Reference
only, INVALID deep_mlp: synchronized is still near-lossless on MNIST — d4 +1.01pp, the
only MET deep_mlp cell — but grows with depth/difficulty to d8 MNIST +1.67 / FMNIST +2.60
/ KMNIST +6.72pp. That off-MNIST growth is a **training/conversion** problem in the TTFS-FT
step, not a mapping problem: map-drop is ~0 or slightly positive, the chip mapping is
bit-exact.)

### 3.2 Cascaded single-spike death-cascade — a dual-axis law (depth × dataset)

Cascaded `ttfs_cycle` (genuine single-spike, latency-gated, ramp-reconstructed) carries a
real firing-gain deficit that compounds with **both** depth and dataset margin.
Synchronized stays pinned at the ANN ceiling on every cell, so **casc→sync gap = pure
firing-gain signal** (synchronized is the confound-free reference).

**(a) Depth axis — deep_cnn MNIST (width 16, S=4), clean rc=0.** A *sharp depth-threshold*,
not a smooth widening:

| Depth | cascaded | casc→sync |
|---|---|---|
| d4 | 0.9883 | −0.15pp (lossless/tied) |
| d5 | 0.9917 | +0.07pp (lossless/tied) |
| d6 | 0.9383 | **+5.2pp (sharp onset)** |
| d8 | 0.9517 | +4.16pp (plateau) |
| d10 | 0.9517 | −4.00pp (bounded plateau) |
| d12 | (n=1, inconclusive) | — |

Verdict **CONFIRMED-WITH-CONFOUND**: lossless ≤ d5, then a sustained **~4–5pp plateau**
≥ d6. The literal "monotonically widening with depth" framing is **refuted on MNIST** (the
gap is 5.2 / 4.16 / 4.00pp at d6 / d8 / d10 — it shrinks, driven by cascaded seed variance).
The earlier rc=1-confounded read inflated d10 to 13.86pp; the **clean rc=0 read is ~4pp.**

**(b) Dataset axis at fixed depth — the MNIST no-collapse corner does NOT generalize.**
Cascade re-opens the moment dataset margin tightens:

| deep_cnn | MNIST | FMNIST | KMNIST |
|---|---|---|---|
| d4 casc→sync | −0.15 (lossless) | +3.90 | +6.19 |
| d5 casc→sync | +0.07 (lossless) | +6.03 | +4.62 |

**(c) Depth × dataset compound — the headline ladder.** deep_cnn **FashionMNIST**
casc→sync is **monotone-widening**, all clean rc=0, continuous d4–d10:

| deep_cnn FMNIST | d4 | d6 | d8 | d10 |
|---|---|---|---|---|
| casc→sync | +3.90 | +6.11 | +11.34 | **+17.91** |
| casc→ANN | 5.76 | 9.11 | 14.28 | **20.97** |

deep_cnn **KMNIST** casc→sync widens overall: d4 +6.19 → d6 +5.85 (n=1 prov) → d8 +7.19 →
d10 +15.98pp (the d6→d8 dip is within 200-sample noise).

**Worst case = deep × hard:** deep_cnn **d10 FMNIST** casc 0.7250 vs ANN 0.9347 =
**20.97pp** deployed→ANN; **d10 KMNIST** casc 0.8025 vs ANN 0.9663 = **16.38pp** — the
largest cascaded deficits in the whole table. **Depth and task-hardness multiply.** Best =
shallow × easy (d4 MNIST −0.15pp).

**(d) Architecture-dependent onset — CNN delays but does NOT abolish the cascade.**

- **lenet5** (shallow CNN, IR max-lat ~3, 2 neural segments): mild and dataset-ordered,
  low variance. Cascaded n=1000 deployed→ANN: **MNIST 0.39pp** (near-lossless; the n=50
  flat [.98,.98,.98] was subsample rounding noise) < **KMNIST 3.06pp** (mild; replicate
  3.54, 6-seed 3.30) < **FMNIST 7.86pp** (lossy, a real residual on a valid CNN, barely
  moved from n=50). Paired full-test SCM lenet5/MNIST: cascaded 0.9835 vs sync 0.9891 =
  0.56pp (no death-cascade; casc SCM==HCM 0.9846 → mapping lossless, the sub-pp loss is
  mode-intrinsic). lenet5 **SVHN** cascaded ≈ **19pp** but **non-finalized** (crashed at
  SoftCoreMapping rc=1; pre-crash 0.674 vs sync 0.8605) — lower-confidence.
- **deep_cnn** (deep CNN, width 16): onset at **d6** (vs deep_mlp's d4). The conv-shared /
  pooled structure delays the cascade; the deficit tracks the **length of the greedy
  single-spike partial-sum chain**, so a CNN needs more layers to build a chain as long as
  the plain-MLP stack.
- (Reference, INVALID) **deep_mlp**: earliest/worst onset — d4 casc→sync +4.32pp,
  d8 +9.27pp MNIST / +15.71pp FMNIST (~8× larger than CNN at fixed MNIST). Dataset is the
  *dominant* axis (LeNet5 swept 0.91pp MNIST → 5.52 KMNIST → 5.99 FMNIST → 19.07 SVHN).
  Non-monotonic at d6, so the literal monotonic phrasing is bounded there too.

### 3.3 Rescue is negative on the valid convnet

The first valid deep_cnn firing-gain-deficit cell (d6 MNIST onset, `dcnn_d6_onset_gatefix_rescue`,
on-chip 99.41%, 3 seeds, S=4) was gridded over two orthogonal rescue knobs:

- **conversion_policy** (controller revive→refine routing): **DOES NOT RESCUE.**
  cpFalse 0.9500 vs cpTrue 0.8983 → cp lift **−5.17pp mean / −1.50pp median**. High
  variance; cpTrue s2 = 0.77 is a *genuine rc=0 finalized collapse* (on-chip 99.41%,
  NF↔SCM 1.0000, torch↔sim 1.0000 — a real bad basin, not a crash). The +2pp s0 lift is a
  single-seed artifact.
- **ttfs_theta_cotrain** (per-channel θ gain-trim): **BROKEN — rc=1 on Conv2D.** All 6
  cotTrue runs finalize rc=1 with `RuntimeError "[ModelRepresentation] forward failed at
  node Conv2DPerceptronMapper(name='features_3')"` (torch shape mismatch 28 vs 16 at dim 3).
  The 0.99+ `__target_metric.json` floats are **stale pre-deployment ANN-stage artifacts**,
  not valid metrics.

**Conclusion:** on the valid convnet there is **no working firing-gain rescue knob**; the
~5pp d6 plateau has no fix today. Synchronized stays the unconditional deep_cnn default
(sync ceiling at d6 = 0.9904 full 10k, ANN 0.992). The much-cited **+7pp controller
"rescue" (0.8754 → 0.9452) was only ever on the INVALID deep_mlp d8**; an ablation
(`ttfs_blend_fast:false`) isolated it as the **controller driver** (gradual 8-rung ramp +
adaptive post-finalize recovery, finalize-cliff ~0.45 → ~0.94), **not** the keystone /
policy / escalation (cpFalse 0.9396 ≈ cpTrue 0.9417 with the fast ladder off). On valid
lenet5/FMNIST the rescue is only **partial**: conversion_policy closes ~17% (+1.17pp,
0.846 → 0.8577), then a hard ~5.6–5.9pp AC2 floor remains; theta_cotrain=TRUE was never
successfully run there. conversion_policy is a **deficit-proportional** lever (no-op on
near-lossless lenet5/MNIST −0.07pp; +1.17pp on mild FMNIST; +6/+10pp on severe INVALID
deep_mlp d8), not a blanket boost — and it moves no valid cell to AC2-MET. The
ESCALATE-vs-MATCH branch was **never separable** in a real run (`propose_recipe` always
proposes driver=controller; the escalation path is exercised only in unit tests with an
artificially-dead cascade).

### 3.4 Five-mode landscape ranking

Fixed vehicle: **mlp_mixer_core / MNIST / S=4 / n=3** (15/15 rc=0). Deployed 3-seed mean vs
ANN, best → worst:

| Rank | Mode | Deployed | Gap | Verdict |
|---|---|---|---|---|
| 1 | ttfs (analytical) | 0.9807 | +0.15pp | Lossless (real-valued V, not binary; least-lossy instrument) |
| 2 | ttfs_quantized | 0.9773 | +0.55pp | ~Lossless (analytical V + act-quant + offload; fastest, ~500s) |
| 3 | ttfs_cycle **synchronized** | 0.9607 | +2.19pp | Not lossless but tightest (sd 0.25pp) |
| 4 | lif | 0.9600 | +2.29pp | Not lossless — **budget** statement only (LIF reaches ≥ANN at stabilize 400→6000; this S=4 config does not) |
| 5 | ttfs_cycle **cascaded** | 0.9523 | +3.11pp | **Lossy outlier**, last by every measure (sd ±0.91pp) |

Cascaded is **−0.84pp below its synchronized sibling** (clean confound-free delta — only
knob is `ttfs_cycle_schedule`; per-seed sync−casc all positive +1.62/+0.18/+0.72) and
−2.5..−2.9pp vs the analytical modes. It is the only genuine single-spike
ramp-reconstructed binary schedule, and its highest variance is the cold-cascade
death-cascade fragility — a shallow-depth corroboration of the §3.2 depth law.

---

## 4. System Architecture

### 4.1 The ANN→SNN deployment pipeline

`run.py` → `src/main.py:run_pipeline()` builds a `DeploymentPipeline` from a JSON config.
The engine (`pipelining/pipeline.py:Pipeline`) runs an ordered list of `PipelineStep`s,
each with a **class-level data contract** (`REQUIRES/PROMISES/UPDATES/CLEARS`) over a
namespaced `PipelineCache` (keys `"{step}.{key}"`). The engine verifies the requires/
promises DAG at assembly time and enforces a per-step accuracy tolerance via
`step.pipeline_metric()` **on the full test set**. Step order is single-source-of-truth:
`get_pipeline_step_specs(config)` resolves a `DeploymentPlan` (`pipelining/core/deployment_plan.py`
— the **one** place deployment flags are read, grep-guarded), then `StepPlan.resolve(plan)`
filters an ordered registry by each step's `applies_to(plan)`.

**Always-on sequence:** model build → (pretrain | `WeightPreloadingStep`) → optional
`TorchMappingStep` → optional `PruningAdaptationStep` → `ActivationAnalysisStep` →
activation-family adaptation → `NormalizationFusionStep` → `SoftCoreMappingStep` →
`HardCoreMappingStep` → backend sim steps. The activation family branches on
`plan.is_lif_style`: LIF → `LIFAdaptationStep`; analytical/TTFS →
`ClampAdaptationStep` (+`ActivationShiftStep` + act-quant chain when
`activation_quantization`), with `TTFSCycleAdaptationStep` for ttfs_cycle_based.
`weight_quantization` adds weight-quant steps + `CoreQuantizationVerificationStep`.

**Pretrain / convert.** Native torch models (`models/deep_cnn.py:DeepCNN`,
`models/lenet5.py`, `models/perceptron_mixer/`, `models/deep_mlp.py`) train at full speed;
`torch_mapping/converter.py:convert_torch_model` (FX trace → `MapperGraphConverter`) wraps
MM⁺→BN?→ACT segments into `Perceptron`s on a Mapper DAG. `ActivationAnalysisStep` records
per-perceptron stats/scales used by the tuners.

**Tuning (smooth adaptation).** `tuning/` drives transformations 0→1 while holding
accuracy: `AdaptationManager` (decorator rates), `SmoothAdaptationTuner` (calibrate →
`RateScheduler` rate-search → stabilize at rate=1.0), `TuningBudget` (all thresholds from
`accuracy_se = 0.5/√N`), `LRRangeFinder`. The optimization-driver axis (`controller`
default | `fast` ladder) is resolved in `DeploymentPlan` and consumed uniformly via
`plan.optimization_driver_for_family(...)`.

**Normalization fusion.** `transformations/normalization_fusion.py` folds BN/norm affine
into the preceding Linear. Captured trap: layer-replacing steps must call
`refresh_perceptron_bias_references` or TTFSActivation bias refs go stale (the cascaded
NF↔SCM incident).

**Mapping (torch → IR → cores).** `mapping/ir/` = `IRGraph`/`NeuralCore`/`ComputeOp`.
`SoftCoreMappingStep` materializes weights into the IR, runs IR pruning, then the
on-chip-majority gate (§4.4), the SCM rung-2 metric, and the NF↔SCM per-neuron parity gate.
`HardCoreMappingStep` packs into physical `HardCore`s via the single greedy
`placement_engine.py:run_placement` (a `Materializer` strategy: layout shape-only vs
runtime weight-bearing), governed by a `platform.MappingStrategy`
(`allow_coalescing/neuron_splitting/scheduling`). Output is a `HybridHardCoreMapping`
(neural segments + host ComputeOp stages); identity 1:1 mappings feed the rung-2 SCM gate.

**Spiking simulation.** Primary deployable simulator = `SpikingHybridCoreFlow` (float64
membrane math, per-core latencies, segment boundaries; forward routes on
`SpikingModePolicy.decode_mode()`: timing→`_forward_ttfs`, count→`_forward_rate`). Three
external backends, capability-validated up-front by `BACKEND_REGISTRY.selected_step_specs(plan)`
against `_BACKEND_CAPS`: **nevresim** (C++ codegen→compile→execute), **Lava/Loihi**
(LIF only, TTFS rejected), **SANA-FE** (energy/latency + HCM parity). All share
`run_hybrid_stages` and the inter-stage contract `hybrid_semantics.py`
(LIF/rate→count/T; TTFS→[0,1]).

### 4.2 Spiking modes (firing × sync)

Per `chip_simulation/spiking_mode_policy.py:policy_for_spiking_mode`:

- **lif** — signed integrate-and-fire rate code (`LIFActivation`, subtractive/Novena reset);
  the lossless-capable mode.
- **ttfs** — pointwise-analytical NF (continuous).
- **ttfs_quantized** — analytical: propagates real-valued V across cores (floor-staircase +
  half-step bias), **not genuine spiking**; excluded from the NF↔SCM per-neuron gate by
  design.
- **ttfs_cycle_based** — the genuine binary single-spike mode, two schedules:
  - **CASCADED** — latency-gated, fires once, ramp reconstruction at the consumer
    `membrane(t)=Σ wⱼ·(t−tⱼ)`; greedy/lossy → **the death-cascade vehicle**. Decision-level
    NF↔SCM argmax agreement ~1.0.
  - **SYNCHRONIZED** — latency groups run sequentially, full S-step window, grid-quantized
    stage input (`ttfs_input_grid_quantize`). NF == deployment kernel, per-neuron bit-exact
    (gate budget 0.02).

**Encoding-layer placement** (`encoding_layer_placement`): `"subsume"` (default) marks
segment-start encoder perceptrons as **host** ComputeOps (off-chip spike-train generators);
`"offload"` clears the flag so they map as **on-chip** NeuralCores and the flow encodes raw
segment input directly — a larger hardware-accelerated surface, functionally identical
under signed-IF (offload HCM == subsume to 1e-6). *(This flag is the lever in §6(a).)*

### 4.3 The autonomous campaign loop

A **4-layer never-idle GPU research loop** (`scripts/campaign/` + `scripts/gpu/`),
filesystem-coordinated through a crash-safe queue; the design intent (from the headers) is
that refill must **not** depend on a human — GPUs stay busy as long as the backlog has
enabled work. The shared substrate is `gpu_queue.py:GpuQueue`, a persistent filesystem
queue under `$MIM_CAMPAIGN_DIR` (default `runs/campaign/q/`: `pending/ running/ done/
failed/`, one JSON file per job, atomic `os.rename` **is** the claim — no double-run —
ordered by `priority` then enqueue time), with `STOP`/`PAUSE` sentinels. The campaign's
results land as ledger rows under `runs/campaign/ledger.jsonl`, which the validity gate and
this checkpoint tally.

**Certification + acceptance.** `certification.py`: a `CertificationCell` = `(firing × sync
× backend)` key `mode[/schedule]@backend[#variant]`; a `RegressionFloor` = frozen
{deployed_accuracy, wall_clock} + eps/budget; `certify()` is the gate (PASS iff acc ≥
floor−eps AND wall ≤ budget; MISSING_FLOOR never silently passes). The absolute-AC overlay
(F1) adds `ac1_target` (deployed goal), `ac2_reference` (ANN lossless ref), `ac5_budget_s`
(per-fine-tuning-PASS wall), reported as an `AbsoluteVerdict`. Standing per-run gates live
in `deployment_faithfulness.py` (`scm_torch_sim_parity_check`, `nf_scm_parity_samples`,
SANA-FE version-pin drift, `DEPLOYED_METRIC_PROTOCOL`). Cost/Pareto infra
(`cost_extraction.py`): `CostRecord`/`CostScatter`, `energy_proxy_neuron_steps =
Σ_d neurons_d·S_d`, `max_ft_pass_wall_s` for AC5.

### 4.4 The on-chip-majority gate (the decision rule)

`mapping/verification/onchip_majority.py:assert_onchip_majority_or_raise`, wired into
`SoftCoreMappingStep` after IR pruning, **default-on** (`onchip_majority_gate`, floor 0.5).
**on-chip = total_params − unique host-side ComputeOp params** (offloaded encoder
Linear/Conv, classifier readout, attention — deduped by module identity). It raises
`OnchipMajorityError` if **< 50%** of params are physically on chip. This is the
program's **decision rule for what must be on-chip**: a deployment counts as a real chip
result only when the chip is doing the majority of the computation, not when a fat host
encoder/classifier carries the model and the "chip" is a rounding error.

---

## 5. Validity Methodology

**The ≥50%-on-chip rule.** A ledger row is science-valid only if `on-chip params / total
≥ 0.5` (`deployment_validity != INVALID_host_majority`). This is the validity keystone:
the firing-fidelity claims are about computation that actually happens on the chip.

**deep_mlp retirement.** deep_mlp w64 is INVALID at **every depth** — its 784→64 *host*
encoder dominates the parameter count, leaving only **19.7% on-chip at d4 / 36.4% at d8**.
All deep_mlp deployment results are therefore **retired from the science** (70 ledger rows
excluded). Their phenomena are *real* — the d4 cascade onset, the +7pp keystone "rescue",
the depth/dataset gap ladders — but they are **not valid on-chip deployments**, so they
serve only as reference/context, never as headline evidence.

**Per-family on-chip fractions (valid vehicles):**

| Vehicle | on-chip | Status |
|---|---|---|
| lenet5 | 99.1% | VALID |
| deep_cnn | 98.5–99.5% | VALID |
| mlp_mixer_core | 90.1% | VALID |
| deep_mlp w64 | 19.7% (d4) / 36.4% (d8) | **INVALID — retired** |

**Ledger accounting.** 158 rows → 86 science-valid (WS3=100→…, by cluster WS3/WS6/WS7/
WS-mode/VALIDITY) after removing 72 (70 INVALID_host_majority deep_mlp + 2 VALIDITY
bookkeeping: `onchip_majority_audit`, `quarantine_coverage`).

**Confounds carried with the science (named, not hidden):**

1. **n=200 subsample mismatch.** All cascaded deep_cnn runs use
   `max_simulation_samples=200` (0.005 grid, ~1.5–3.5pp/seed binomial noise; deployed =
   exact 1/200 multiples) while **synchronized reports full 10k**. Read the **casc→sync
   gaps** (the d6–d10 5–18pp gaps are 2–5× the noise band), **not** 3rd decimals. lenet5
   n=1000 is ~±1pp; permode raw floats are instrument-mixed (cascaded nevresim vs full-test
   SCM sync — sign solid, magnitude mixed).
2. **rc=1 HCM packing crashes (now superseded).** The deep deep_cnn rungs
   (`dcnn_/pdcnndeep_/pdcnnladder_` d6–d12) finalized rc=1 at HardCoreMappingStep
   ("No more hard cores available", a greedy-pack capacity/packing infra failure) **after**
   SoftCoreMapping wrote `__target_metric.json` and **after** parity gates passed (NF↔SCM
   1.0, torch↔sim 0.9961–1.0). Those reads are genuine pre-crash SCM accuracies
   (CONFIRMED-WITH-CONFOUND), not clean deployments; the rc=1 d10 read was crash-inflated
   (13.86pp). The **enlarged bigcores config** (cores.count=480, 4× cores) re-ran these to
   clean FINALIZED_rc0 — the d4/d6/d8/d10 dataset cube + MNIST ladder are now VALID rc=0,
   so the cited numbers are the **bigcores rc=0** ones; the rc=1 reads are superseded.
3. **Thin seed counts at the deepest cells.** Several deepest cascaded cells are n=2 or n=1
   (d10 FMNIST cascaded s1 rc=−9 OOM; d10 KMNIST cascaded s0 rc=1; d6 KMNIST n=1 prov;
   d8 KMNIST n=2; d12 MNIST n=1 = survivor artifact, s0/s2 rc=−9 timed out at 3600s).
   Cascaded seed sd is wide (1.0–3.25pp); synchronized is tight (0.01–0.53pp).
4. **Resolution hardens the law.** Genuine n=1000 deep_cnn reads (8.51/11.14pp at d8/d10)
   are **larger** than the clean n=200 reads — the gap is *not* a grid artifact (d10 s0
   log shows a genuine mid-pipeline SCM collapse 0.9939→0.1873→0.7375 = death-cascade
   fragility).
5. **No at-chance confound on any valid cell** (all ANN ≫ chance 0.10–0.1135 — genuine
   firing-gain, not an untrained floor). The INVALID deep_mlp d≥12 (incl. d16/d24 @ w128)
   *are* ANN-training-floor confounds (plain Linear+ReLU never trains past d~8; pretrain
   peak 0.156, ends at chance for both modes) — the deep firing-gain test there is gated on
   a residual/norm backbone (WS2).

---

## 6. Proposed Changes & Improvements

These are the **corrections** to carry forward as first-class items — several un-block or
fix the gaps above with little or no code.

**(a) deep_mlp is RE-VALIDATABLE via `encoding_layer_placement="offload"` — one config flip,
no code change.** deep_mlp was retired only because its 784→64 *host* encoder dominated
params (19.7–36.4% on-chip). Flipping the encoder to `offload` maps it **on-chip** as a
NeuralCore (functionally identical under signed-IF; offload HCM == subsume to 1e-6), which
pushes the on-chip fraction over the 0.5 gate. **If a pure-MLP depth-law point is wanted
for the paper, un-retire deep_mlp this way** — it would recover the cleanest single
depth-law vehicle without the conv-shared-chain confound. (Note WS2: the deepest deep_mlp
needs a residual/norm backbone to train at all; `offload` only fixes the *validity* axis,
not the training-floor axis.)

**(b) GELU / non-ReLU needs NO new op.** The activation-adaptation step and tuners already
**transform-or-replace** activations (the ClampAdaptation / activation-quant chain operates
on whatever activation the converter records). Supporting GELU and other non-ReLU
activations for the transformer/pretrained backbones is **engineering polish on the
existing adaptation path**, not a new spiking primitive.

**(c) Gate-driven on-chip mapping for LayerNorm & attention.** Keep LayerNorm and attention
**host-side by default** and build an on-chip mapping for them **only if a specific
deployment fails the ≥50% gate.** Their parameter counts are small relative to the conv/
linear backbone, so most deployments will pass the gate with them on the host — do **not**
pre-build the tricky on-chip mappings speculatively. Let the validity gate be the trigger
that decides when the engineering cost is actually warranted.

**(d) Fix the `Conv2DPerceptronMapper` theta_cotrain forward bug.** The per-channel θ
gain-trim rescue lever is **broken on convnets**: all 6 cotTrue d6 runs crash rc=1 with the
torch shape mismatch (28 vs 16 at dim 3) in `converted_model_flow.forward` at
`features_3`. Fixing this Conv2D forward is the **only thing that makes the one principled
firing-gain rescue lever testable on the valid convnet** (§3.3) — without it, the ~5pp d6
plateau is untreatable by anything but synchronized.

**(e) Other system polish.** (i) Per-channel scales for the TTFS θ-trim path (the
per-source-scales fix) so the gain-trim is per-channel where the death-cascade is
per-channel. (ii) Sim-resolution defaults: raise the cascaded deep_cnn default off
`max_simulation_samples=200` toward the synchronized full-10k convention so casc/sync are
read on the *same* instrument (the n=1000 reads already show the law hardens, not
softens). (iii) Certification per-model references: the absolute-AC overlay should carry a
per-(model,dataset) `ac2_reference` so the lossless reference is the right ANN for each
cell, not a global one.

---

## 7. Remaining Future Work

### Wave-B — rigor / foundation (cheap, parallelizable)

These bring the existing valid-vehicle science to a venue bar; they are inexpensive and
mostly independent of each other.

- **Published-baseline head-to-heads (RMP / QCFS / percentile-norm).** Run the standard
  ANN→SNN conversion baselines on the *same* valid vehicles and report deployed accuracy
  side-by-side. *Unblocks:* the "vs prior art" claim a venue requires; situates
  synchronized-TTFS-lossless against the field. *Cost:* low — reuses the existing pipeline
  and vehicles; new adaptation configs, no new infra.
- **Residual / norm backbone for deep-trainable + transformers.** Add a residual backbone
  so plain-MLP/CNN trains past d~8 (removes the §5 confound-6 training-floor) and so
  transformer blocks become trainable vehicles. *Unblocks:* the deep depth-law beyond d8
  without an ANN-floor confound; the transformer arm of the architecture matrix; the
  deep_mlp depth-law point (with §6(a)). *Cost:* low–medium — model code + retrain.
- **CIs / ablations bundle.** Lift seed counts on the thin deepest cells (§5 confound-3),
  attach confidence intervals to every headline gap, and formalize the rescue-lever
  ablations (controller-vs-policy, fast-ladder on/off). *Unblocks:* statistical claims
  instead of point estimates; honest error bars on the dual-axis law. *Cost:* low (GPU time
  only, no new code).
- **Energy/latency Pareto allocator.** Use the existing `cost_extraction.py`
  (`energy_proxy_neuron_steps`, `max_ft_pass_wall_s`) to build a per-mode accuracy-vs-cost
  Pareto and an S-allocation allocator. *Unblocks:* the neuromorphic "is it worth it"
  argument (cascaded's traffic savings vs synchronized's fidelity); a Pareto figure.
  *Cost:* low–medium — the cost records already exist; the allocator and plots do not.

### Wave-C — publication headline (expensive, GPU-weeks)

- **Pretrained near-SOTA bridge (timm/torchvision ResNet-50 / ViT-B).** Lower a near-SOTA
  pretrained backbone onto the chip and report the deployed→FP gap. *Unblocks:* the
  pretrained-near-SOTA headline (§1) — the claim that the pipeline survives on real models,
  not just from-scratch toys. *Cost:* high — needs §6(b) GELU polish, §6(c) gate-driven
  attention/LN mapping, and substantial GPU time.
- **CIFAR → ImageNet via ffcv.** Scale the dataset axis from the MNIST-family diagnostics
  to CIFAR and then ImageNet (fast ffcv loaders). *Unblocks:* the dataset-margin axis at
  publication scale (the §3.2 dataset law currently tops out at KMNIST/SVHN). *Cost:* high
  — GPU-weeks for ImageNet training + deployment.
- **Dual-regime certification.** Certify both regimes (pretrained-near-SOTA and
  from-scratch-profiling) under the same AC overlay with per-model references (§6e-iii).
  *Unblocks:* a single coherent results table across both publication arms. *Cost:* medium
  on top of the two above.
- **The full architecture × dataset × regime matrix.** Fill the complete grid (CNN /
  residual / transformer) × (MNIST-family / CIFAR / ImageNet) × (pretrained / from-scratch)
  with synchronized-vs-cascaded fidelity + Pareto. *Unblocks:* the comprehensive
  breadth claim that is the program's reason for existing. *Cost:* highest — the union of
  all C items; this is the GPU-weeks publication bar.

---

## 8. Sequencing & Recommendation

**Do Wave-B baselines and Wave-B residual-backbone first.** They are independent of each
other and parallelizable: the baseline head-to-heads need only adaptation configs on
existing vehicles, and the residual backbone is self-contained model code. Together they
remove the largest remaining confound (the training-floor) and supply the prior-art
comparison — both are cheap and both are prerequisites for taking the program seriously at
a venue. The CIs/ablations and the Pareto allocator can run alongside on the GPU backlog
since the cost records and ledger already exist.

**Wave-C is the GPU-weeks publication bar and is gated on B.** The pretrained near-SOTA
bridge depends on the GELU polish (§6b) and gate-driven attention/LN mapping (§6c); the
ImageNet scaling depends on the residual backbone; dual-regime certification depends on
per-model references (§6e-iii). None of C should start before B's foundation is in place —
otherwise the expensive runs inherit the very confounds B exists to remove.

**Net recommendation:** the from-scratch diagnostic tier is *done and honest* — synchronized
is the lossless deep default, cascaded carries a confirmed dual-axis death-cascade, and no
rescue lever survives on a valid convnet. The next dollar goes to **B (baselines +
residual backbone)** to make that science publishable, then to **C** for the
pretrained-near-SOTA headline. Before spending C-scale GPU time, land the two cheap code
fixes that unblock it: the `Conv2DPerceptronMapper` theta_cotrain forward fix (§6d) so the
rescue question is settled on convnets, and the `encoding_layer_placement=offload`
re-validation (§6a) if a clean pure-MLP depth-law point is wanted.
