# Why `mlp_mixer_core` fails deployment accuracy in all five modes — and the function-preserving transformation that fixes the fixable part

**Verdict.** The mode-universal binder is structural: `TorchMLPMixerCore` is the
only tier-0 vehicle that is a **9-hop, normalization-free, residual-free ReLU
chain** (`src/mimarsinan/models/torch_mlp_mixer_core.py:24,43` — `self.ln =
nn.Identity()` in both mixer cores; no skip connections anywhere;
`patch_bn` at the stem is the only norm). That one structural fact produces
**two coupled failures** that every mode inherits:

1. **Float-envelope deficit (mode-universal).** All 15 mixer cells pretrain at
   0.9540–0.9703 — every one below the ~0.973 band the other vehicles clear.
   No function-preserving transformation can touch this component.
2. **Per-hop scalar-θ grid starvation (mode-graded, ∝ 1/S).** Per-channel
   activation ranges inside each hop spread over 2–3 decades (dead-to-hot
   1,870× measured), while calibration commits ONE scalar θ per perceptron
   (`transformations/activation_scale_policy.py:20-26`,
   `pipelining/pipeline_steps/adaptation/activation_analysis_step.py:163-181`).
   At S=4 the pooled θ leaves the MEDIAN channel only 2.1–3.6 of 4 usable
   levels, the cold tail zero, and 51–80 % of positive activation mass below
   one grid step — which the deployed floor/ceil kernels zero out —
   compounding across 9 hops: the AQ/LIF/cascade craters. This part IS
   fixable by an exact cross-layer scale migration, derived and prototyped
   below.

Everything below is evidence-first: campaign logs under
`/home/yigit/.slurmech/workspaces/mimarsinan-xlog1/runs/<run>/artifacts/stdout.log`;
prototype scripts and raw outputs archived at
`docs/research/findings/mixer_scale_pathology_artifacts/` (`proto.py`,
`qat_probe.py`, `clipped_proj.py`, `results.json`, `qat_results.json`,
`clipped_results.json`; GPU run 2026-07-10, RTX PRO 6000, repo env).

---

## 1. Decomposition: where each mode loses the points

Per-stage `[PROFILE]` trajectories (post-stage-recovery reads) + in-stage
`[MBH-ENDPOINT]` entries (the raw crater before recovery). Exemplar cells:

| cell | mode | S | pretrain | stage deltas (pp) | deepest in-stage crater | final | net conv. loss |
|---|---|---|---|---|---|---|---|
| t0_01 (420c022b) | lif | 4 | 0.9680 | shift −2.27, AQ +1.16, LIF **−6.91**, WQ +2.02 | LIF endpoint entry 0.8336 | 0.9078 | −6.02 |
| t01_01 (aa044231) | lif | 8 | 0.9637 | shift −0.66, AQ +0.70, LIF −1.91, WQ +1.06 | LIF entry 0.9269 | 0.9552 | −0.85 |
| t01_02 (193e76bd) | lif | 16 | 0.9630 | LIF −0.68, WQ +0.20 | (gauge PASS) | 0.9609 | −0.21 |
| t01_05 (a05afad2) | sync | 4 | 0.9627 | prune −0.78, AQ **−3.55**, WQ −1.00 (floor diverged) | AQ endpoint entries **0.632/0.709** | 0.9104 | −5.23 |
| t01_24 (42de6997) | sync | 8 | 0.9660 | prune −1.11, AQ −1.60, WQ +2.41 (16k floor QAT) | AQ entries ~0.63–0.94 | 0.9671 | +0.11 |
| t01_06 (b6c42524) | ttfsq | 8 | 0.9540 | clamp +0.83, AQ −1.71, WQ +0.97 | WQ entry 0.9342 | 0.9536 | −0.04 |
| t0_11 (e3a3782e) | ttfsq | 16 | 0.9567 | AQ −0.80, WQ +0.71 | WQ entry 0.9502 | 0.9637 | +0.70 |
| t01_07 (ba662d73) | ttfs | 8 | 0.9703 | clamp −0.02, WQ −0.31 (no AQ step at all) | — | 0.9670 | −0.33 |
| t01_23 (18cd5b1f) | ttfs | 8 | 0.9580 | clamp −0.37, WQ +1.52 (16k floor) | — | 0.9694 | +1.14 |
| t01_12 (ea6a5382) | casc | 8 | 0.9656 | FT hop-wall: k=1→2 rejected ×4, forced full-cascade read 0.4732, retention trip | full-transform 0.4739 | FAILED (0.5083) | collapse |

Reading, per mode:

- **ttfs (analytical)** deploys value-continuous — no `Activation Quantization`
  step appears in its trajectory. Its loss is envelope + WQ only (−0.33 pp on
  t01_07); the 16k-floor cells train THROUGH the pretrain read and saturate at
  the ~0.97 fbu asymptote (`tuning/orchestration/tuning_policy.py:64-70`).
  ttfs fails only because the envelope sits at 0.958–0.970.
- **ttfsq** installs the nearest-rounding grid — the most starvation-tolerant
  kernel (`tuning/orchestration/install_resolution/gauges.py:16-20,40-46`).
  Its AQ loses 0.8–1.7 pp and the endpoint claws most back; finals land ON the
  envelope (t01_06 final 0.9536 ≈ pretrain 0.9540). Envelope-bound.
- **lif** deploys the θ/T rate grid (`gauges.py:180-181`): crater −0.68 pp at
  S=16, −1.91 at S=8, −6.91 at S=4 — clean 1/S scaling of grid starvation,
  then recovery is budget-cut mid-climb (endpoints cut at +2.7–13.6 pp/1k
  slope; tier-0 caps `wq_endpoint_recovery_steps=2000`,
  `test_configs/generate.py:508-521`).
- **sync** (ceil kernel = floor in value space,
  `models/spiking/wire_semantics.py:26-37`) craters −1.6 to −1.9 pp at S=8 and
  −3.55 pp at S=4 with AQ endpoint entries at 0.63–0.71 — the "S=4 level
  starvation" already named in the group diagnosis. Fully-repaired floor cells
  (t01_24) still cap at 0.9671: envelope again.
- **casc** collapses at the k=1→2 hop of the prefix ladder: L=9 exceeds
  `PROVEN_RECOVERY_DEPTH = 6` (`gauges.py:31-34` — "t01_12 L=9 read 0.88 with
  a clean value gauge"); the compounding single-spike kernel over 9
  starved-grid hops is unrecoverable by rung training.

**The shared pp:** every mode carries the 0.3–1.9 pp envelope deficit
(pretrain 0.9540–0.9703 vs the 0.973 band); the quantized modes add a
value-grid crater that scales as 1/S and with kernel harshness
(floor/ceil ≫ nearest ≫ none). WQ as such is nowhere the binder
(raw projection −0.5/−0.6 pp, recovered wherever any budget exists).

Vehicle contrast at identical budgets (final `[PROFILE]` reads across the
50-cell campaign): pretrain envelopes — deep_cnn 0.9828–0.9949, lenet5
0.9893–0.9925, simple_mlp 0.9724–0.9807, deep_mlp 0.9547–0.9766, **mmixcore
0.9540–0.9703**. lenet5 sync S=4 (757b17e3): pretrain 0.9893, AQ −0.20 pp,
WQ −0.05 pp, deployed 0.9879 — despite its own gauge reading 3/4 hops starved.
Same kernel, same S, same starved gauge: 4 hops + 2 pp headroom survive it;
9 hops + zero headroom do not. (deep_mlp — the other deep unnormalized FC
chain — is the second-lowest family and also fails several quantized cells:
the pathology is the chain shape, not "the mixer" per se.)

## 2. The statistics: what the scalar θ cannot cover

Campaign gauges ([MBH-A6], warn-only, `activation_analysis_step.py:210-216` →
`gauges.py:185-207`):

- t01_05 (sync s4, q=1.0): **9/9 hops starved**; θ spans 1.81→15.21 across
  depth; `median_levels` 2.07–3.57 at the 4-step grid; `starved_mass`
  0.51–0.80 (fraction of positive mass below ONE grid step; log lines 222-232).
- t0_01 (lif s4, q=0.99): 9/9 starved, θ 1.58→9.55, starved_mass 0.53–0.74
  (lines 145-155).
- t0_21 (sync s8, q=1.0): 6/9 starved even at S=8 (lines 226-233).
- S=16 mixer cells: 0/9 starved (e3a3782e line 137) — matching the mild
  S=16 craters.
- lenet5 sync s4 (757b17e3 lines 141-143): 3/4 starved with similar
  median_levels (2.0–3.55) — **starvation alone does not decide the outcome;
  depth × headroom does.**

Prototype channel-resolved statistics (tier-0 recipe faithfully mirrored:
Adam 3e-3, wd 5e-5, 5 warmup + 2 cosine epochs, CE ls=0.1, MNIST 95/5;
envelopes REPRODUCE the campaign: mixer 0.9548, lenet5 0.9908, simple_mlp
0.9781). Per-hop live-channel q99 spread (`results.json`):

| hop | θ (q99 pooled) | ch q99 max/med/min | live ch | medlvl@S4 | starved mass@S4 |
|---|---|---|---|---|---|
| patch_embed_full | 3.20 | 4.17 / 3.06 / 1.11 | 32/32 | 3.82 | 0.68 |
| blocks_0_fc1 (tok) | 1.44 | 2.27 / 1.12 / 0.67 | 64/64 | 3.11 | 0.54 |
| blocks_0_fc2 (tok→**patch axis**) | 3.44 | 4.95 / 2.11 / 0.92 | **13/16** | 2.45 | 0.71 |
| blocks_1_fc1 (ch) | 2.13 | 3.40 / 1.61 / **0.05** | 62/64 | 3.03 | 0.70 |
| blocks_1_fc2 (ch) | 3.38 | 4.90 / 2.78 / **0.03** | 27/32 | 3.29 | 0.59 |
| blocks_2_fc1 (tok) | 2.87 | 4.22 / 2.16 / 0.14 | 53/64 | 3.02 | 0.61 |
| blocks_2_fc2 (tok→patch axis) | **13.95** | 15.72 / 12.03 / 1.03 | **11/16** | 3.45 | 0.58 |
| blocks_3_fc1 (ch) | 9.82 | 20.71 / 8.70 / 0.21 | 64/64 | 3.55 | 0.70 |
| blocks_3_fc2 (ch) | 13.05 | 15.55 / 12.51 / **0.008** | 23/32 | 3.84 | 0.61 |

vs lenet5 (worst live-channel spread 8.47/2.20 ≈ 3.8×, all channels within one
decade, θ 1.56–6.33) and simple_mlp (worst 2.76/0.27 ≈ 10×, θ 1.80–2.26).
The mixer's signature: (i) **live-channel dynamic range up to 1,870×**
(blocks_3_fc2: 15.55 vs 0.008) — cold channels get ZERO usable levels under
the pooled θ; (ii) **θ drift ×10 along depth** (1.44→13.95) with nothing to
re-center it (no norm, no skip); (iii) **the token-mixer fc2 hops' channels
ARE patch positions** (`torch_mlp_mixer_core.py:26,33` — fc2 maps
hidden→num_patches on the permuted tensor, so the perceptron's output-channel
axis is the patch index; MNIST digit mass is center-concentrated, hence 3–5 of
16 patch channels dead and the rest spread ~5×).

Raw one-shot grid projection (no recovery training; test acc; float refs
mixer 0.9548 / lenet5 0.9908 / simple_mlp 0.9781):

| kernel, S | mixer | lenet5 | simple_mlp |
|---|---|---|---|
| floor/ceil S=4 | **0.0892** (chance) | 0.365 | 0.892 |
| floor/ceil S=8 | 0.271 | 0.987 | 0.970 |
| floor/ceil S=16 | 0.942 | 0.991 | 0.976 |
| nearest S=4 | 0.868 | 0.988 | 0.974 |
| nearest S=8 | 0.941 | 0.990 | 0.977 |
| WQ 5-bit | 0.9513 | 0.9917 | 0.9755 |

Three campaign facts reproduce quantitatively: the 1/S crater scaling, the
nearest-kernel (ttfsq) tolerance, and the depth amplification (same starved
gauge, 4-hop lenet5 loses 0.4 pp at floor-S8 where the 9-hop mixer loses 68 pp
one-shot — the pipeline's ramped QAT + endpoint recovery then buys the mixer
back to its 0.92–0.95 campaign reads, at 600–16,000 recovery steps per stage).

## 3. The transformation: exact cross-layer channel-scale migration

### 3.1 Derivation

Consider adjacent hops A → σ → B on one path of the perceptron DAG: A affine
with weights `W_A ∈ R^{m×n}`, bias `b_A ∈ R^m`; σ the elementwise activation;
B consuming A's m output channels as its input features (`W_B ∈ R^{p×m}`).
Pick any s ∈ R^m with s_c > 0 and let S = diag(s). Transform

```
W_A ← S⁻¹ W_A      b_A ← S⁻¹ b_A      W_B ← W_B S      (b_B unchanged)
```

Exactness: σ = ReLU is positively homogeneous per channel,
σ(λx) = λσ(x) ∀λ>0, so

```
W_B S · σ(S⁻¹(W_A x + b_A)) + b_B = W_B S · S⁻¹ σ(W_A x + b_A) + b_B
                                  = W_B σ(W_A x + b_A) + b_B     ∀x. ∎
```

The float function of the whole network is unchanged — not approximately:
the only deviation is fp32 rounding (measured below at 2×10⁻⁶ on logits).
Per-channel activations, however, rescale as a_c → a_c/s_c, so the choice

```
s_c = q99_c / G   (live channels; s_c = 1 for dead ones),  G = geomean(live q99)
```

makes every live channel's q99 equal to G. The pooled θ then sits at ≈G and
`median_effective_levels = S·median_c(q99_c)/θ → S` — the gauge's maximum;
every channel uses the full grid.

### 3.2 Constraint set for exactness (all load-bearing)

1. **s_c > 0 strictly** (diag(s) invertible; homogeneity needs λ>0).
2. **Positive homogeneity of σ.** Holds for ReLU (the deployed
   `LeakyGradReLU` forward IS max(0,·) — only its GRADIENT leaks;
   `models/nn/activations/autograd.py:16-38`, re-exported via
   `models/nn/layers.py:10`) and
   LeakyReLU; **fails for GELU**. Gate on
   `perceptron.base_activation_name ∈ {ReLU, LeakyReLU}` — all tier-0 mixer
   configs use ReLU (`test_configs/tier0/t0_01_lif_mmixcore_wq_s4.json:39`).
3. **Producer bias scales with the row** (b_A ← S⁻¹b_A); consumer bias is
   untouched.
4. **Feature-adjacency through interposed ops.** Every op g between σ(A) and B
   on the DAG must be channelwise positively homogeneous:
   g(S⁻¹y) = S⁻¹g(y). Holds for reshape/flatten (`Ensure2DMapper`),
   permute/rearrange (`EinopsRearrangeMapper`), Max/AvgPool and mean over
   NON-channel axes, Identity. Fails for anything adding a bias or mixing
   channels.
5. **Fan-out closure.** If A's channel axis feeds several consumers, ALL must
   be column-scaled; a residual join requires the same s on both branches; any
   non-homogeneous consumer forces s_c = 1 on the shared channels.
6. **BN-attached perceptrons** (pre-NF): realize S⁻¹ on the norm's affine
   (γ←γ/s, β←β/s) or through
   `PerceptronTransformer.apply_effective_parameter_transform` — exact in both
   train and eval BN modes since batch statistics of the pre-norm activation
   are untouched.
7. **WQ-grid coupling — the measured hard constraint.** NAPQ quantizes each
   perceptron's (w,b) jointly on ONE symmetric grid scaled by
   max(|w|,|b|) (`transformations/normalization_aware_perceptron_quantization.py:41-46`).
   Unbounded equalization amplifies near-dead channels' rows/columns by
   1/s_c — measured ×794 on blocks_3_fc2 (live q99 0.008 vs G 6.59) — and one
   such outlier starves the entire shared 5-bit grid: **unclipped α=1
   equalization drove WQ-5bit from 0.9513 to 0.0892 (chance)** in the
   prototype. Therefore s must be clipped, s_c ∈ [1/r, r]; r=4 held WQ at
   0.9493 (−0.2 pp) while preserving the full gauge repair on migrated hops.
   (This is the same failure family the codebase already guards for biases via
   `clip_off_saturated_effective_bias`,
   `normalization_aware_perceptron_quantization.py:31-34`.) An α-tempered
   variant s_c = (q99_c/G)^α, α∈(0,1], is the SmoothQuant-style continuous
   version of the same trade.
8. **Ordering.** Must run BEFORE Activation Analysis: after θ install, the
   clamp/staircase decorators reference the scalar
   `perceptron.activation_scale` (`tuning/orchestration/adaptation_manager.py:227,254-256,271-274`)
   and rescaling weights under a stale θ is no longer function-preserving
   through the clamp. Must run AFTER pruning commits its masks (row/column
   scaling preserves zeros but changes magnitude ranking).

### 3.3 What is exactly migratable in this mixer

Producer channel axis = consumer feature axis holds for 5 of the 9 hops'
outputs: `tok.fc1→tok.fc2` and `ch.fc1→ch.fc2` inside each of the 4 mixer
cores, plus `blocks_3_fc2 → mean(dim=1) → classifier` (mean over patches is
channelwise homogeneous). **Not migratable exactly:** the token-mixer fc2
outputs (channel axis = patch positions, consumed by the channel mixer as
batch-like rows under weight sharing — scaling patch p's slice cannot be
absorbed into weights shared across patches once biases exist), and
`patch_embed → token-fc1` (same axis flip). This is an honest structural
limit of weight-space migration on weight-shared axes; see §6 for the two
exact escapes (per-channel θ, biasless mixers).

## 4. Prototype results

Setup: `proto.py` / `qat_probe.py` / `clipped_proj.py` (archived in
`mixer_scale_pathology_artifacts/`);
tier-0 mixer (patch 4×4, c32, fc 64/64, 2 blocks, ReLU), pipeline-faithful
θ/q99/gauge math (code cited in headers). Equalization applied to the 5 exact
pairs, s from measured per-channel q99, then θ RE-measured (as running
Activation Analysis after the pass).

**Exactness.** max|Δlogit| = 2.1×10⁻⁶ over 2048 test images, argmax agreement
1.000000 (α=1.0, α=0.5, and clipped variants alike).

**Gauge repair.** Every migrated hop reads `medlvl@S4 ≈ 4.0` (the grid
maximum; 4.00/4.00/4.02/4.00/4.00)
post-equalization; e.g. blocks_3_fc2's θ falls 13.05→6.59 and its channel
q99 collapses to a point mass at G. Non-migratable hops (blocks_0_fc2,
blocks_2_fc2, blocks_1_fc2) retain their spread — the residual predicted by
§3.3.

**WQ interaction.** Unclipped α=1.0: WQ-5bit 0.9513→0.0892 (constraint 7).
Clip r=4: 0.9493. α=0.5 unclipped: 0.9482.

**Raw projection (one-shot install, no training), clipped r=4
(`clipped_results.json`):**

| kernel, S | pre-eq | post-eq (clip 4) | Δ |
|---|---|---|---|
| floor/ceil S=4 | 0.0892 | 0.0892 | 0 (dead either way one-shot) |
| floor/ceil S=8 | 0.2710 | **0.4295** | **+15.9 pp** |
| floor/ceil S=16 | 0.9416 | 0.9449 | +0.3 pp |
| nearest S=4 | 0.8679 | 0.8943 | +2.6 pp |
| nearest S=8 | 0.9407 | 0.9430 | +0.2 pp |
| nearest S=16 | 0.9502 | 0.9487 | −0.2 pp (noise) |

**QAT-exit probe (the pipeline-faithful question: does recovery training from
an equalized model land higher?).** One-shot install + 600 steps STE QAT at
the deployed ceil kernel, Adam 2e-3 cosine (mirrors `endpoint_floor_lr`,
`tuning_policy.py:64`), keep-best, 3 seeds (`qat_results.json`):

| S | arm | entry | exits (3 seeds) | mean exit |
|---|---|---|---|---|
| 4 | pre | 0.0892 | 0.1135 / 0.1135 / 0.1135 | 0.1135 (constant-class collapse) |
| 4 | post | 0.0892 | 0.1135 / 0.1135 / 0.1135 | 0.1135 (ditto) |
| 8 | pre | 0.2710 | 0.9302 / 0.9345 / 0.9331 | 0.9326 |
| 8 | post | **0.4293** | 0.9380 / 0.9347 / 0.9376 | **0.9368** (+0.4 pp) |

Ramped install (pipeline-style rate ramp over 600 steps, random-mask blend as
`RandomMaskAdjustmentStrategy`+`MixAdjustmentStrategy`) + 600 further steps at
full rate, 1200 total, 3 seeds (`clipped_results.json`):

| S | arm | exits (3 seeds) | mean ± sd |
|---|---|---|---|
| 4 | pre | 0.4139 / 0.3730 / 0.4922 | 0.4264 ± 0.049 |
| 4 | post | 0.4217 / 0.5332 / 0.6005 | **0.5185 ± 0.074 (+9.2 pp)** |
| 8 | pre | 0.8333 / 0.9166 / 0.8864 | 0.8788 ± 0.034 |
| 8 | post | 0.9112 / 0.9330 / 0.9257 | **0.9233 ± 0.009 (+4.5 pp)** |

**Interpretation.** With only 5/9 hops migratable and the float function
bit-preserved, equalization: (i) lifts the raw conversion entry by +15.9 pp at
the S=8 ceil/floor kernel — exactly the kernel×S regime where the campaign's
sync/lif/casc cells crater; (ii) raises every paired QAT exit (12/12 pairs
≥, most strictly >): +0.4 pp when the install is one-shot, +4.5 pp (S=8) and
+9.2 pp (S=4) under the pipeline-style ramped install; (iii) cuts seed
variance ~4× at ramped S=8 (0.034→0.009) — the same divergence axis as the
campaign's 3.5 pp identical-config floor coin-flip (t0_21 0.9316 vs t01_24
0.9671). At S=4 a one-shot install is information-dead in both arms
(everything below θ/4 floors to zero through 9 hops; both arms collapse to
the majority class) — S=4 is reachable only through the ramped/laddered
install the pipeline already uses, where the equalized arm gains most. The
probe budgets (600–1200 steps) are deliberate miniatures of the pipeline's
staged ladders; the paired deltas, not the absolute levels, are the
measurement.

## 5. Integration point

- **Seam:** a new function-preserving step between `Pruning Adaptation` and
  `Activation Analysis` —
  `src/mimarsinan/pipelining/core/pipelines/deployment_specs.py:47-48`.
- **Mechanism module:** sibling of the existing training-time fold,
  `src/mimarsinan/transformations/normalization_fusion.py` → e.g.
  `src/mimarsinan/transformations/channel_scale_equalization.py`.
- **Adjacency discovery:** walk the mapper DAG (`Mapper.source_mapper` chains,
  `src/mimarsinan/mapping/mappers/base.py:35-44`;
  `PerceptronMapper`, `mapping/mappers/perceptron_mapper.py:17-54`) from each
  consumer perceptron through homogeneous structural mappers to its producer;
  channel axes are owner-declared
  (`models/perceptron_mixer/perceptron.py:99-103`,
  `activation_channel_axis` at `perceptron.py:21-45` fails loud).
- **Calibration stats:** reuse `collect_channel_stats` /
  `ChannelStatsAccumulator.per_channel_q99`
  (`tuning/orchestration/install_resolution/capture.py:16-58,112-137`) — the
  A6 machinery already rides the Activation-Analysis forward, so the pass can
  share one capture with the gauge.
- **Raw writes:** `PerceptronTransformer` effective-parameter transforms for
  BN-attached perceptrons
  (`transformations/perceptron/perceptron_transformer.py`), plain
  `layer.weight/bias` scaling otherwise; refresh TTFS bias references as
  `fuse_into_perceptron` does (`transformations/normalization_fusion.py:47`).
- **Config:** one recipe knob (off / α / clip r), registered in
  `config_schema/registry/entries_conversion.py` beside
  `activation_scale_quantile` (`entries_conversion.py:161`); wired per
  conversion family in `tuning/orchestration/conversion_policy.py` (the AQ
  recipe-knob tables at `conversion_policy.py:41-111`).
- **Verification hook:** the step's postcondition is the §3.1 identity — a
  unit test asserts logit equality to fp tolerance on a fixed batch, and the
  A6 gauge re-read shows `medlvl == levels` on every migrated hop.

## 6. Genericity and honest limits

**Generic by construction.** The pass has no workload or architecture
constants: s is measured from calibration data on whatever perceptron DAG
exists; on vehicles with homogeneous channels (lenet5, deep_cnn: BN'd convs,
spread within one decade) s_c ≈ 1 and the pass is a no-op up to fp rounding.
It helps ANY vehicle with heterogeneous channel scales feeding a scalar-θ
grid — deep_mlp (the second-worst family, same chain shape) is the first
beneficiary after the mixer. It composes with, and does not replace, the
lever-plan items (C1–C4 recovery-geometry fixes, P3 pretrain keep-best) in
`docs/research/findings/tier0_passall_lever_plan.md`.

**What it cannot fix, by construction:**

1. **The float envelope.** Function preservation means pretrain 0.954–0.970
   stays. The envelope deficit needs training-side levers (P3; more epochs;
   or giving the vehicle back its norms/skips at build time — a model change,
   not a transformation).
2. **Weight-shared axes.** Token-fc2's per-patch spread survives (§3.3). Two
   exact escapes exist: (a) **per-channel θ** — the scale-propagation layer
   already carries per-channel thetas verbatim
   (`mapping/mappers/scale_propagation.py:63-72`), and chip cores hold
   per-NEURON thresholds, so extending `activation_scale` to a vector on the
   starved hops is representable end-to-end; (b) **biasless mixer variants**
   (the `nobias` configs) make the channel-mixer patch-homogeneous, turning
   the cross-pair migration exact.
3. **Intra-channel heavy tails.** starved_mass stays ~0.5–0.6 on equalized
   hops: ReLU outputs are heavy-tailed WITHIN a channel, and a linear
   per-channel scale cannot reshape that distribution. This is precisely the
   regime the nearest-rounding kernel tolerates (ttfsq thresholds,
   `gauges.py:16-20`) and the floor/ceil kernels do not.
4. **The cascade depth wall.** casc's L=9 > `PROVEN_RECOVERY_DEPTH=6` collapse
   is a temporal-kernel compounding limit diagnosed separately; equalization
   reduces per-hop θ (hence first-fire delay θ/⟨drive⟩,
   `gauges.py:127-132`) but cannot shorten the chain.
