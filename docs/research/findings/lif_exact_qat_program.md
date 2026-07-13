# LIF exact-QAT: training the deployed integer-staircase composition — diagnosis, theory, prototype, integration

**Date:** 2026-07-13. **Question.** The lossless mandate rejects the "capacity
cap" framing for the LIF mixer cells, and the campaign evidence supports the
rejection: (a) WS-W measured the trained SYNC composition +7.9 pp ABOVE its own
float envelope (ledger §2F.2) — QAT finds weights whose staircase composition
beats naive quantization; (b) matched-cell, deployed LIF at S=8 (unpruned,
counts alphabet {0..8}) reads 0.9556 while sync at S=8 (pruned10, the same
θ/S value grid) reads 0.9603–0.9650 on the same BN mixer — no less capacity,
lower accuracy = an adaptation-method gap, not physics; (c) the commutation
theorem (`lif_deployment_exactness.md` Theorems 0–2) says deployed LIF ≡ an
integer-staircase ANN `clamp(F(T·z/θ),0,T)` — a training target the current
LIF recipe never optimizes directly. This memo adjudicates and develops the
central hypothesis: **LIF exact-QAT** — train the exact deployed composition
(per-hop integer staircase with the shipped half-step, θ trainable in the
loop, entry encode exact) with an STE backward, mirroring `sync_exact_qat`
which demonstrably handles L=9 within 0.7–1.1 pp of envelope.

**Headline results (all measured on the real t01_01-class pipeline artifact,
local rerun `generated/wsy_t01_01_phased_deployment_run`, seed 0, this box;
prototype scripts in scratch `wsy/`, §5.0):**

1. **The surrogate inventory (Phase 1a) confirms the mismatch, and sharpens
   it.** The shipped LIF recipe trains FIVE different objects in sequence —
   a float model perturbed by an unflagged mid-tread bias bake (Shift), a
   T-annealed constant-drive floor-staircase family (T = 32→16→16→8) with a
   per-cycle ATan-surrogate backward (ramp), the genuine cascade WITHOUT the
   half-step (LIF endpoint, 600 steps), then a one-shot +θ/(2T) fold that
   displaces the converged operating point (measured −2.06 pp here, −1.67 pp
   on the wave), then the genuine cascade again under weight-quantization
   reprojection (WQ endpoint). θ is trainable in NONE of them (`lif_theta_cotrain`
   defaults off; `per_channel_theta` disarmed post-QAT). Sync trains ONE
   object — the exact deployed ceil kernel under identity STE — through its
   whole ladder. §2 gives the file:line inventory.
2. **The trained LIF composition also sits ABOVE its own float envelope**
   (Phase 1b): deployed 0.9614 vs clamp-envelope twin 0.9494 on TEST
   (+1.20 pp) — the fifth measurement of the post-QAT-inversion meta-lesson,
   now on the LIF side. The optimization deficit is therefore measured
   against the PIPELINE envelope, matched-cell: LIF deployed sits −1.7 pp
   below its own AQ read (0.9614 vs 0.9785) where sync sits −0.6…−0.8 pp
   below its post-prune envelope on the same box — a 2–3× larger
   envelope-relative gap with a strictly-easier spec (no pruning).
3. **The commutation residual is real, small at the artifact, and
   training-steerable (the pivotal Phase-3 finding).** At the trained
   artifact, deployed-vs-staircase-twin reads −0.17 pp (argmax agreement
   0.966), growing along depth exactly as the temporal-A6 gauge predicts
   (per-hop mean |Δ| 0.03→0.31 θ/T grid-steps; total first-fire delay 23.3 vs
   window 8). But pure-staircase exact-QAT — which trains the L=9 mixer to
   **0.9757–0.9774 TEST in 3000 steps / ≈40 s** (θ frozen / θ in-loop),
   +1.4 pp above anything the shipped recipe reaches — deploys on the raw
   cascade at only 0.9505: aggressive staircase optimization AMPLIFIES the
   V3 back-loading term ~15× (−0.17 → −2.5 pp). **The pure staircase forward
   is not a faithful QAT object for deep single-segment LIF at low S; the
   commutation preconditions are load-bearing and Goodhartable.**
4. **Both exactness-restoring completions were built and measured.**
   (i) *Count-STE exact-QAT* (forward = the genuine per-cycle cascade
   bit-exactly, backward = the commutation-staircase STE at the window-count
   level): train-forward == deployed-forward **argmax parity 1.0000** at init
   and at the endpoint — the parity-by-construction property sync enjoys, now
   on LIF — and with θ in-loop reaches deployed 0.9562 TEST from the folded
   entry (0.9408), matching the shipped WQ-endpoint deployed read (0.9556
   wave) without weight quantization; θ visibly reallocates (starved hop 5:
   0.496 → 0.559 with per-channel spread 0.43–0.68; hop 6: 1.72 → 2.12).
   (ii) *Staircase-QAT + per-hop re-timed deployment* (the R5 transcode,
   count-preserving `round((c/T)·T)=c`): the SAME staircase-trained weights
   read **0.9747 TEST deployed** (θ frozen; −0.10 pp from their train
   forward) and **0.9751 TEST with θ in-loop** — +1.9…+2.0 pp over the
   shipped 0.9556 baseline and within 0.35 pp of the AQ envelope.
   The R5 blocker ("the NF twin has no per-hop counterpart",
   `conversion_policy.py:82-89`) dissolves: under exact-QAT the trained
   staircase IS the twin.
5. **The raw single-segment cascade at S=8 × depth 9 carries a ~1–2 pp
   V3 ceiling that no training method removed** (ATan-BPTT endpoint 0.9614;
   count-STE from the folded entry 0.9524/0.9562 and from the unfolded
   artifact 0.9562/0.9580 (θ frozen/in-loop), all at parity 1.0000; pure
   staircase 0.9505) while the same weights' staircase/re-timed reads sit at
   0.96–0.977. The mixer-S8 LIF residual is thus substantially a
   DEPLOYMENT-SEMANTICS term (back-loading under one 8-cycle window across 8
   hops), not model capacity — consistent with the mandate's rejection of
   the capacity framing, and repaired value-exactly by the re-timing option
   that already exists in the machinery.
6. **One-shot WQ projection on exact-QAT weights craters (0.48–0.81) —
   recovery is mandatory** (the pipeline already knows this: its own WQ
   projection entry was 0.840 → endpoint 0.9516, eval-subset domain). The integration plan (§6)
   runs the WQ endpoint THROUGH the exact forward; nothing here weakens the
   two-scale identity (R1, armed).

Prototype provenance: `wsy_probe_lif_envelope.py`, `wsy_exact_qat_proto.py`,
`wsy_exact_qat_cascade.py`, `wsy_e2_retimed_eval.py` + result JSONs + arm
state dicts in the session scratch directory
(`/tmp/claude-1005/-home-yigit-repos-research-stuff-mimarsinan/11224c9e-f926-4cb5-a527-2d0211f4bd25/scratchpad/wsy/`);
every forward goes through the repo modules
(`LIFActivation`, `SegmentForwardDriver`, `uniform_spike_train`,
`apply_lif_half_step_bias_compensation`, `NormalizationAwarePerceptronQuantization`);
the count-STE policy's fidelity is asserted against the persisted
`_ChipAlignedNFForward` (argmax parity 1.0000 on 2000+4096 samples). Seed 0,
RTX PRO 6000 under full contention; the t01_01 rerun reproduces the wave's
regime (pretrain 0.9837 vs 0.9820, adapt 0.9614 vs 0.9585 — the known
cross-hardware non-bit-identity, ledger §2F.2).

---

## 1. The deployed object (recap, code-anchored)

Fixed by the commutation memo and unchanged here. Per hop, with wire weights
`W~ = θ_in·W/θ_out`, per-cycle bias `b~ = b/θ_out`
(`transformations/perceptron/perceptron_transformer.py:106-125`), window T,
wire threshold 1:

```
Q = Σ_i w~_i c_i + T·b~            (terminal charge; +1/2 when the half-step is folded)
c_out = clamp(F(Q), 0, T)          (Theorem 2, up to V3/V4 timing violations)
```

`F(x) = ⌊x⌋` for `"<="`, `F(x) = ⌈x⌉−1` for the strict `"<"` the tier-0 lif
cells pin (`test_configs/tier0/t0_01_lif_mmixcore_wq_s4.json:54`; t01_01 the
same). Entry encode: `n = round(clamp(v/θ,0,1)·T)` Uniform spikes
(`models/spiking/hybrid/rate_forward.py:82`,
`chip_simulation/recording/spike_modes.py:32-41`). Readout: raw counts in
every measurement of this memo — during this session commit a63dddc5 made
the C2 membrane decode chip-realizable (nevresim final-membrane read port +
honesty gate), superseding the R8 exclusion where the knob is armed and
every accuracy backend is membrane-capable; C2 is a READOUT decode and
composes additively with everything here (§6.1 note 7). Timing: within one neural segment, arrivals are the
producers' actual fire cycles; the emission cap (1 spike/cycle) makes late
charge undeliverable — V3 back-loading (`lif_deployment_exactness.md` §3).
The t01_01 mixer trunk is ONE neural segment of 8 fc hops (permutes are
structural), so it never re-times internally; its temporal-A6 gauge FAILs
(total first-fire delay 23.3–24.6 vs window 8, stdout `[MBH-A6]
kind=temporal`).

## 2. Phase 1a — what the recipe actually trains (the surrogate inventory)

Every stage below verified in source; the table is the mismatch, stage by
stage, against the deployed object of §1.

| # | stage | forward trained | backward | T | half-step | θ | budget |
|---|---|---|---|---|---|---|---|
| S1 | Activation Shifting | FLOAT model after an **unflagged one-shot bias bake** `b_eff += (θ/2)/target_tq` on every perceptron (`tuning/tuners/activation_shift_tuner.py:40-54`; the `_use_ttfs` predicate at `:22-24` routes lif into the bake branch) + recovery training | float | — | bake #1 (Tq-domain) | frozen | recovery |
| S2 | Clamp / AQ ladders | **transform-inert**: in lif mode the clamp/quantize decorators are never installed (`tuning/orchestration/adaptation_manager.py:186-204`, `subsumes_decorators` includes `is_lif(spiking_mode)`) and the rungs train 0 steps (`adaptation_manager.py:16-31`); the step reads still move via the shared stabilization/recovery training on the FLOAT model (wave AQ +0.58 pp, this box +1.20 pp) | float | — | — | frozen | recovery only |
| S3 | LIF Adaptation ramp (`lif_tanneal`) | value-domain module forward; each hop = `LIFActivation` **constant-drive** multi-step (`models/nn/activations/lif.py:171-176,201-216`: x/θ repeated T times into an IFNode, mean×θ) ≡ the floor/strict staircase at the RUNG's T (Theorem A2) — rungs T = 32→16→16→8 (`tuning/orchestration/mbh_tanneal.py:17-25,60-72,102-117`; blend pinned at 1.0) | per-cycle ATan surrogate, BPTT over T identical cycles (`lif.py:51-87,121-151`; spikingjelly IFNode, subtractive reset `firing_strategy.py:111-114`) | 32→8 | **none** | frozen | fast ladder |
| S4 | LIF endpoint (P1″) | the **genuine cascade**: `_ChipAlignedNFForward` installed at finalize (`tuning/tuners/lif_adaptation_tuner.py:163-166`; ordering `smooth_adaptation_run.py:335,342-343` → `kd_blend_adaptation_tuner.py:296-313,182-193`) routes `model(x)` through `SegmentForwardDriver` + `LifSegmentPolicy` (`spiking/chip_aligned_nf.py:14-30`, `spiking/segment_policies.py:21-154`): per-cycle trains between hops, signed single-step IF, encoder host-run + uniform re-encode — deployment-faithful incl. V3 | ATan BPTT through T cycles and 9 hops | 8 | **none** | frozen | 600 steps |
| S5 | half-step fold | one-shot `b_eff += θ/(2T)` at WQ entry (`pipelining/pipeline_steps/quantization/weight_quantization_step.py:43,73-89` → `mapping/support/bias_compensation.py:73-109`, encoders skipped `:89`), i.e. **bake #2 on an operating point converged without it** | — | — | fold lands | frozen | 0 |
| S6 | WQ | projection-only rungs (0 steps, `tuning/tuners/normalization_aware_perceptron_quantization_tuner.py:40-45`) + endpoint recovery through `trainer.model(x)` = the persisted chip-aligned cascade (`model_training/basic_trainer_eval.py:111,126,143,166`; `tuning/forward_install.py:36-39,55-61` — the override pickles with the artifact) with float-aux reprojection | ATan BPTT | 8 | folded | frozen | ≤16k (used 4788 wave / measured ~80 ms/armed step) |

θ in-the-loop exists as machinery but never runs: the opt-in promotion seam
is `kd_blend_adaptation_tuner.py:108-132` → `spiking/theta_cotrain.py:19-40`,
gated by `lif_theta_cotrain` (default False, `lif_adaptation_plan.py:66`);
the post-hoc `per_channel_theta` arming is DISARMED by A/B on the trained
composition (`conversion_policy.py:67-76`) — the fourth inversion instance.

**Contrast — what sync trains (one object, end to end):** the exact deployed
ceil kernel `TTFSCeilStaircaseDecorator` under identity STE
(`adaptation_manager.py:237-259` → `models/nn/activations/autograd.py:57-70`,
forward IS the wire kernel `wire_semantics.py:37-53`), entry snaps on exactly
the deployed seams (`adaptation_manager.py:81-114`), the half-step folded
BEFORE the QAT and owned by it
(`bias_compensation.py:112-131`), mapping-time compensation skipped
marker-assertedly. Theorem 1 (sync memo §1.3) then gives train ≡ deploy
neuron-by-neuron; the seeded campaign realizes it at 1.0000/256 parity.

**The two verified inconsistencies beyond the family mismatch:**

- **Double half-step bakes.** S1's bake (+θ/(2·Tq), unflagged) and S5's fold
  (+θ/(2T), `LIF_HALF_STEP_FLAG`) both land in the effective bias; with
  Tq = T = 8 the composition receives two mid-tread shifts, each absorbed
  only by whatever training follows it. Neither is an error in isolation
  (training re-optimizes offsets), but each is a one-shot operating-point
  displacement: S1 read −1.72 pp at the Shift step; S5 measured **−2.06 pp**
  on this box (B0 → B0f, §5) and −1.67 pp on the wave (t01_01 fold-leg).
  This is the mechanism of the post-QAT inversion trap: exactness constants
  derived off-composition are injected into a composition that was optimized
  without them.
- **The gradient family.** Even the endpoints (S4/S6), which do train the
  deployed forward, use the per-cycle ATan surrogate composed over T cycles
  × L hops — a membrane-history-weighted backward whose per-step cost is
  O(T) (the measured ~80 ms/step at S=8 throttles the endpoint to ~5k steps
  in its wall) — where sync's backward is the identity STE (the float
  Jacobian of the clamp envelope), O(1) per hop.

## 3. Phase 1b/1c — trained-artifact measurements

Probe `wsy_probe_lif_envelope.py` on the LIF-Adaptation artifact (post-S4,
pre-fold; deployed step read 0.9614 reproduced exactly by the persisted
forward). Three arms on identical weights: **dep** = chip-aligned genuine
cascade (== SCM/HCM by the 21/21-lossless tail), **stair** = the commutation
staircase (unpatched module forward; constant drive ≡ `clamp(F(T·z/θ),0,T)`
by Theorem A2 — init parity vs the exact kernel asserted 1.0000 in §5),
**env** = the float clamp envelope with the entry quantizer bypassed (the
WS-W idiom, LIF side).

| arm | val (3000) | TEST (10k) |
|---|---|---|
| deployed (genuine cascade) | 0.9570 | 0.9614 |
| commutation staircase twin | 0.9633 | 0.9631 |
| float envelope twin | 0.9560 | **0.9494** |

**1b verdict:** deployed − envelope = **+1.20 pp** on TEST. The LIF QAT, like
sync's (+7.9 pp, ledger §2F.2), trains INTO the staircase: the float
envelope of the trained weights is not a valid reference at any hop (the
staircase-prefix sweep is monotone-increasing from 0.9560 to 0.9633 — every
staircased hop IMPROVES on its envelope; fifth measurement of the
meta-lesson). The optimization deficit is therefore stated
envelope-relative-to-pipeline, matched-cell on this box:

| cell | pipeline envelope | deployed | gap |
|---|---|---|---|
| lif t01_01 S=8 (unpruned) | AQ 0.9785 / pretrain 0.9837 | 0.9614 (pre-WQ) / 0.9556-class (post-WQ wave) | −1.7 … −2.6 pp |
| sync t0_21 S=8 (pruned10) | post-prune 0.9708–0.9726 | 0.9650 | −0.6 … −0.8 pp |

Same BN mixer, same θ/S value grid per hop, LIF unpruned — and 2–3× the
envelope-relative gap: the adaptation-method gap, directly measured.

**1c — the commutation residual per hop** (dep vs stair, 3000-sample
calibration, θ/T grid-step units; `node_value_recorder` on the segment walk
vs post-activation capture on the module walk):

| hop | name | E[dep−stair] | E|dep−stair| | E[stair−env] |
|---|---|---|---|---|
| 0 | patch_embed (enc) | 0.000 | 0.000 | −0.319 |
| 1 | blk0 fc1 | +0.028 | 0.037 | −0.175 |
| 2 | blk0 fc2 | −0.025 | 0.064 | −0.336 |
| 3 | blk1 fc1 | −0.002 | 0.075 | −0.261 |
| 4 | blk1 fc2 | −0.016 | 0.092 | −0.183 |
| 5 | blk2 fc1 (θ=0.496, A6-starved) | −0.055 | 0.063 | −0.264 |
| 6 | blk2 fc2 | −0.276 | 0.276 | −0.915 |
| 7 | blk3 fc1 | −0.232 | 0.310 | +0.149 |
| 8 | blk3 fc2 | +0.127 | 0.305 | −0.164 |

The dep−stair column is V3+V4 (timing) alone — near-zero at the entry,
growing to ~0.3 grid-steps by depth 7–8, with the sharpest onset immediately
after the starved hop (hop 6: −0.28 mean, i.e. late-arriving charge the
emission cap drops) — the temporal-A6 signature localized per hop. End-to-end
it costs −0.17 pp at THIS artifact (whose training saw the cascade), argmax
agreement dep↔stair 0.9661, mean|Δlogit| 2.62. §5 shows this residual is not
a constant of the weights: staircase-only optimization inflates it to
−2.5 pp.

**Matched against sync's per-hop picture** (ledger §2F.2, the same probe
idiom on t0_21): sync's trained-in kernel offsets reach ±1.8 grid-steps with
47 % starved channels at its hop 8, LIF's (stair−env, −0.16…−0.92) are
smaller — both QATs train INTO their kernels and away from the float
envelope. The structural difference is the extra column: LIF has a
per-weight-set TIMING term (dep−stair) with no sync analogue — sync's
whole-window grid snap has no crossing-time hazard (casc memo scope note),
which is precisely why sync's staircase QAT is deployment-complete and LIF's
is not (§4.1).

## 4. Phase 2 — the theory of LIF exact-QAT

### 4.1 The forward family and the parity trilemma

Define the per-hop exact-QAT kernel on the value domain (θ per channel,
window T, comparator per `thresholding_mode`):

```
q_θ(z) = clamp(F(T·z/θ), 0, T)/T ;   out = θ·q_θ(z)                     (1)
```

with the half-step carried as trainable bias mass (the shipped convention,
S5) so (1) stays the pure comparator staircase. Three deployable training
objects exist, ordered by how they close the train↔deploy identity:

- **(A) The staircase composition** — (1) composed hop-to-hop with the exact
  entry round (§4.3). Identical to the deployed cascade **iff** Theorem 2's
  preconditions hold per hop (window coverage, no terminal overshoot, chase
  completion). O(1) per hop per step. *Measured (§5): trains L=9 to 0.9774
  but the raw cascade deviates −2.5 pp — (A) alone is REFUTED as the deploy
  twin for single-segment chains at S=8; V3 is steerable and the optimizer
  steers it.*
- **(B) Count-STE on the genuine cascade** — forward = the deployed per-cycle
  executor semantics exactly (trains between hops, signed IF, subtractive
  reset, strict compare); backward = (1)'s STE anchored at the deployed
  counts: per hop, with Z the window-total preactivation from the
  differentiable path and `dep` the executor's decoded count,

  ```
  out = θ·q_θ(Z) + (dep − θ·q_θ(Z)).detach()                            (2)
  ```

  Forward value ≡ deployed (bit-exact; the timing residual rides the
  detached term), backward ≡ the staircase STE. This is the LIF twin of
  sync's Theorem-1 parity-by-construction — parity holds by CONSTRUCTION,
  not by commutation assumptions. Cost: T no-grad cycle passes + one
  value-domain differentiable pass ≈ 3.6× (A), still ~2× cheaper than the
  ATan-BPTT step and with the honest loss (Goodhart-immune: the optimizer
  sees the true deployed logits). *Measured: parity 1.0000 at init and
  endpoint.*
- **(A + R5) Staircase-QAT with per-hop re-timed deployment** — arm (A) as
  the trained object plus the mapping-level per-hop boundary
  decode/re-encode (count-preserving, `round((c/T)·T) = c`,
  `conversion_policy.py:82-89`). Re-timing resets arrival timing at every
  hop, so Theorem 2's preconditions hold per hop by construction (uniform
  arrivals, cycle-0 anchor, one-level-per-cycle chase) and the deployed
  composition ≡ (A) up to the per-hop V4 transient (≤7·10⁻⁴ rate units,
  lif memo). The historical R5 blocker was twin mismatch; under exact-QAT
  the trained network IS the per-hop twin. *Measured: 0.9747 TEST deployed,
  −0.10 pp from its train forward.*

The program's structural claim: **(B) and (A+R5) are the two sound exact-QAT
completions; (A) alone is not.** (B) optimizes the raw deployment as-is and
honestly measures its V3 ceiling; (A+R5) removes the V3 term value-exactly
and converts the staircase's full trainability into deployed accuracy, at a
mapping-visible latency/energy cost (§6.2).

### 4.2 The θ gradient through the staircase STE

With `r = z/θ`, STE surrogate `q'(r) := 1[0 < r < 1]` (clamp-gated identity),
and `out = θ·q(r)`:

```
∂out/∂z = q'(r) = 1[0 < r < 1]                                          (3)
∂out/∂θ = q(r) + θ·q'(r)·∂r/∂θ = q(r) − r·1[0 < r < 1]                  (4)
```

— the LSQ-family scale gradient, specialized to the count staircase:

- **In-band** (0 < r < 1): (4) = q(r) − r ∈ ±1/(2T) — the per-neuron signed
  quantization residual; θ descends toward per-channel grid placements that
  cancel accumulated rounding, which is exactly the estimator the post-hoc
  quantile/affine ladder tried (and inverted) to compute OUTSIDE the loop.
  In-loop, it is fitted on the live composition at every step: the
  inversion-trap premise ("the reference composition no longer exists by the
  time the constant lands") is dissolved by construction.
- **Count clamp at T** (r ≥ 1): (3) = 0, (4) = q = 1 — saturated channels
  push θ UP with the full downstream gradient signal: level reallocation for
  starved channels (the A6 gauge's starved_mass) becomes a first-class
  descent direction. Measured: hop 5 (θ 0.496, the one A6-starved hop)
  spreads to [0.43, 0.68] per channel; hops 6 and 8 grow +23–25 %, hop 7
  +10 % (§5, E5).
- **Count clamp at 0** (r ≤ 0): both zero — negative pre-activations are
  silent, as deployed. The DEAD-ZONE (0 < r < 1/(2T) rounding to count 0)
  still passes weight gradient by (3): **no dead-neuron trap.** This is
  where the fire-once cascade lesson does NOT transfer: the casc STE's
  pathology was the premature-fire/first-crossing discontinuity of a
  single-spike time code (whose gradient exists only at one crossing;
  revival was mandatory — stefast findings). Multi-spike LIF's count is a
  monotone T+1-level staircase of the window charge with full in-band
  gradient support; E1/E2 converge from 0.951 to 0.977+ in 3000 steps with
  no revival term anywhere.
- **Positivity**: θ enters (1) through `clamp(min)`, which zero-grads a θ
  driven under the floor — degenerate shrinkage is self-limiting (and none
  was observed; max drift ±25 %).

Per-channel θ is exact in the NF cascade by construction (spike·θ_c value
trains; `LifSegmentPolicy` divides per channel on the channel axis), and
exactly realizable on-chip on matching-axis edges via the existing
`per_input_scales` fold (R3; sync memo §4.2). The boundary/host seams that
mean-collapse vectors (`spiking/scale_aware_boundaries.py:31-40`,
`mapping/support/activation_scales.py:49-63`,
`mapping/mappers/scale_propagation.py:76-94`) bound which hops may go
per-channel today — the prototype keeps the externally-consumed trunk hop
scalar-trainable and the encoder frozen (§5), the same constraint set the
integration plan carries (§6.2).

### 4.3 The entry encode

The deployed entry is `n = round(clamp(v/θ_enc,0,1)·T)` Uniform spikes; the
QAT twin is `ChipInputQuantizer` — `round(x_norm·T)/T` under STE
(`models/nn/activations/autograd.py:102-124`), installed on encoding
perceptrons by the LIF tuner (`lif_adaptation_tuner.py:155-161`). Count-wise
the two are IDENTICAL (uniform placement affects timing only, and hop-1
window coverage holds by the latency invariant); under `subsume` placement
the encoder hop itself runs the constant-drive staircase on host
(`segment_policies.py:119-126`), which equals kernel (1) with no half-step
(the fold skips encoders, `bias_compensation.py:89` — consistent, since the
entry round is already mid-tread). Measured: exact-kernel-vs-LIF-node init
parity 1.0000 / mean|Δlogit| 9·10⁻⁵ (f32 ties only). The entry seam is
already exact; exact-QAT changes nothing here.

### 4.4 Depth-L training statistics — what transfers from sync, what doesn't

Sync's L=9 trainability rests on three properties of the identity-STE
staircase QAT, all of which kernel (1) inherits:

1. **The backward is the float Jacobian of the clamp envelope** — no
   surrogate sharpness constant, no membrane-history weighting, no T-fold
   depth multiplication of the backward graph. Gradient noise enters only
   through the forward anchor (loss evaluated at quantized activations),
   whose per-hop deviation from the envelope is bounded by θ/(2T) fresh
   quantization noise; the composed-error law is drift-then-saturate with
   the half-step cancelling the drift term (sync memo §3.2), so the anchor
   stays within a bounded tube of the envelope at any depth.
2. **Mid-tread entry from step 0** — the fold is INSIDE the trained object
   (bias mass, gradient-visible), not a post-hoc bake; the whole-population
   flooring mode (V1/A0) is structurally absent.
3. **The loss surface is piecewise constant but the STE surrogate is not** —
   keep-best on the deployed read + KD to the float teacher (the shipped
   recipe pair) selects among plateaus; measured convergence: 0.951→0.9777
   (θ frozen), →0.9817 val (θ in-loop) at L=9, monotone in keep-best, no
   divergence, ≈12 ms/step.

The LIF-specific deltas: (i) the T-anneal family (S3) is unnecessary from an
adapted basin — both prototype arms train directly at T=8; whether it stays
useful as a from-float warm start is an integration A/B (§6.5); (ii) the
count-STE anchor (2) adds the detached timing residual to the forward — a
forward-only perturbation whose magnitude is the V3 term itself (§3, ≤0.3
grid-steps/hop at the artifact); the backward ignores ∂(timing)/∂w, which is
exactly what makes it Goodhart-immune (the optimizer cannot chase timing
artifacts it cannot see in the gradient) but also means V3 reduction happens
only via the loss's selection pressure, not descent — consistent with the
measured plateau (§5: E4/E5 recover the fold displacement but do not lift
the raw-cascade ceiling).

### 4.5 The deployment identity and its preconditions

For arm (B) the identity is by construction; the preconditions reduce to the
bookkeeping set, each with an existing guard:

- **P-L1 (BN freeze).** Frozen normalization during QAT (`fast_ladder_freeze_bn`,
  recipe; `FrozenStatsNormalization` at WQ) — the prototype freezes stats in
  every arm.
- **P-L2 (counts readout, both sides).** R8 landed: deployed reads are
  counts-decode always; the QAT loss must consume the same decode (it does —
  the policy emits normalized counts to the host classifier exactly as
  `LifSegmentPolicy` does).
- **P-L3 (dtype ties).** f32 train / f64 SCM strict-compare ties — the
  existing torch↔deployed-sim parity gate is the monitor (21/21 lossless
  tail; prototype parity 1.0000 argmax on 4096).
- **P-L4 (join balance).** Depth-balancing relays (landed,
  `lif_depth_balancing_relays: True`) keep V6 = 0 so window coverage holds
  per hop; a gap>1 edge would break both the commutation anchor and the
  count-STE's per-hop Z alignment — keep the post-ChipLatency gap≤1
  assertion as the lock.
- **P-L5 (Novena exclusion).** Novena's zero reset breaks Theorem 0 (charge
  conservation), so kernel (1) is the WRONG backward anchor and the
  staircase twin the wrong reference; the existing
  `firing_strategy.require_chip_faithful_lif_forward` guard plus a mode gate
  keeps exact-QAT `Default`-reset-only. (The C4 expectation repair remains
  Novena's only sanctioned lever.)
- **P-L6 (fold ownership).** The half-step enters ONCE, before the QAT, and
  is trained (S5 moves to the exact-QAT install; the S1 Shift bake is
  removed under the exact arm — §6.1). The double-bake displacement (−1.7…
  −2.1 pp measured per bake) disappears.

For arm (A+R5), Theorem 2 applies per hop with re-timed inputs; the residual
identity gap is per-hop V4 (≤7·10⁻⁴ rate units) plus f32/f64 ties — the
measured train↔deploy gap was −0.10 pp on TEST (0.9757 → 0.9747), i.e. the
staircase NF is a valid parity twin for a re-timed deployment.

### 4.6 Back-loading (V3) as the binding physics term at S=8 × depth 9

Consolidating the §3+§5 measurements into the program's sharpest empirical
statement: on the raw single-segment deployment, six independent training
arms land in the same 0.950–0.962 deployed band (ATan-BPTT recipe 0.9614;
count-STE folded 0.9524/0.9562, unfolded 0.9562/0.9580;
staircase-then-deploy 0.9505), while the SAME weight sets read 0.963–0.982
as staircase compositions and 0.9747–0.9751 re-timed.
The ~1–2 pp band gap is the V3 emission-cap deficit of pushing 8 staircase
hops through ONE 8-cycle window (charge arriving in the window tail is
undeliverable; the temporal gauge measures 23.3 delay-cycles of demand
against 8 of supply). It is: (i) not capacity (the staircase with the same
alphabet trains 2 pp higher); (ii) not gradient quality (count-STE with
perfect parity and clean STE hits it too); (iii) not θ placement (θ in-loop
moves it +0.4 pp, not 2 pp); (iv) removed value-exactly by per-hop
re-timing, which is count-preserving and already implemented as the
boundary transcode. At S ≥ 16 the term shrinks (window supply doubles;
wave S=16 LIF loss −0.91 pp vs S=8 −1.69 pp) — the arming predicate should
be the existing temporal-A6 gauge, exactly as R5 proposed.

**The pairing is load-bearing in both directions (measured controls):**
re-timing the SHIPPED artifact gains only +0.04 pp (B0R: 0.9614 → 0.9618 —
its cascade training already priced the timing in, at the cost of staircase
quality), and staircase-QAT WITHOUT re-timing loses −2.5 pp. Neither lever
works alone; the pair is worth +1.3 pp over the shipped artifact and
+1.9 pp over the shipped WQ endpoint. This also explains why R5 measured
modestly in the correction-series era (+1.9 pp at S=4, ~+0.5 pp at S=8 on
chains, on cascade-adapted weights): re-timing's value scales with how
staircase-optimal the weights are, which only exact-QAT delivers.

## 5. Phase 3 — the prototype on the real pipeline objects

### 5.0 Setup

Artifact: `generated/wsy_t01_01_phased_deployment_run` (byte-identical
t01_01 config, seed 0, this box: pretrain 0.9837 → shift 0.9665 → AQ 0.9785
→ LIF adapt 0.9614; the WQ leg of this rerun was resumed via `start_step`
after a contention kill and completed on post-547f6fbd code — its read is a
completeness datum, not the arms' baseline; the arms' WQ reference is the
wave's 0.9556). Scripts (session scratch `wsy/`):
`wsy_probe_lif_envelope.py` (§3), `wsy_exact_qat_proto.py` (staircase arms),
`wsy_exact_qat_cascade.py` (count-STE + re-timing arms). Recipe constants
throughout: KD α=0.5 T=4 to the AQ-artifact teacher, AdamW lr 2e-3
(= `endpoint_floor_lr`, `tuning_policy.py:66`), θ-group lr 2e-4 wd 0, cosine
to 0.1×, grad-clip 1.0, batch 128, 3000 steps, keep-best every 100 steps on
the train-forward val read (== the deployed val read for the count-STE
arms), norms frozen, seed 0, full-10k TEST reads. θ
in-loop arms: per-channel on interior trunk hops
(`promote_activation_scale_per_channel`), scalar-trainable on the
externally-consumed hop 8, frozen encoder (§4.2 seam constraints).

### 5.1 The ledger

| arm | train fwd (what QAT sees) | TEST train-fwd | TEST deployed (raw cascade) | train↔deploy argmax parity | note |
|---|---|---|---|---|---|
| B0 = LIF-Adaptation artifact | — | stair-twin 0.9631 | **0.9614** | 0.9661 | the shipped pre-WQ state |
| B0R = B0 under re-timed deployment | — | 0.9631 | 0.9618 | — | control: re-timing alone ≈ +0.04 pp |
| B0f = B0 + half-step fold | — | 0.9510 | 0.9408 | — | the fold-leg: **−2.06 pp** one-shot |
| E1 staircase exact-QAT, θ frozen | staircase | **0.9757** | 0.9505 | 0.9460 | Goodhart: V3 inflated −0.17→−2.5 pp |
| E2 staircase exact-QAT, θ in-loop | staircase | **0.9774** (val 0.9817) | 0.9501 | 0.9438 | θ drift ≤ ±5 %; same V3 fate |
| E1R = E1 weights, re-timed deployment | staircase | 0.9757 | **0.9747** | ≈1 (−0.10 pp) | R5 transcode; the (A+R5) identity |
| E2R = E2 weights, re-timed deployment | staircase | 0.9774 | **0.9751** | ≈1 (−0.23 pp) | best deployed read of the program |
| E4 count-STE, θ frozen (from B0f) | genuine cascade (2) | 0.9524 | 0.9524 | **1.0000** | parity by construction |
| E5 count-STE, θ in-loop (from B0f) | genuine cascade (2) | 0.9562 | **0.9562** | **1.0000** | θ reallocates: hop5 0.496→0.559 [0.43,0.68]; hop6 1.72→2.12; hop8 2.05→2.57 |
| E4ⁿᶠ count-STE, θ frozen (from B0, no fold) | genuine cascade (2) | 0.9562 | 0.9562 | **1.0000** | the QAT learns its own offsets |
| E5ⁿᶠ count-STE, θ in-loop (from B0, no fold) | genuine cascade (2) | 0.9580 | **0.9580** | **1.0000** | best raw-cascade read of the arms |
| E3 = E2 + one-shot 5-bit two-scale WQ | — | 0.6109 | 0.4805 | — | projection without recovery craters |
| E6 = E5 + one-shot 5-bit two-scale WQ | — | — | 0.8144 | — | same lesson, milder (parity-trained weights sit nearer the grid) |
| reference: shipped pipeline (wave) | S1–S6 | — | 0.9556 (WQ=NF=SCM=HCM=Loihi) | 1.0 tail | ledger §1 |
| reference: AQ envelope (this box) | — | 0.9785 | — | — | the pipeline envelope |

Init parities (every arm): exact-kernel vs LIF constant-drive node 1.0000
(mean|Δlogit| 9·10⁻⁵); count-STE policy vs persisted chip-aligned deployed
forward 1.0000 (mean|Δlogit| 0.00000); policy-no-retime reads equal the
chip-aligned reads to 4 decimals on val AND test (0.9407/0.9505) — the
custom walk is deployment-faithful.

### 5.2 The acceptance question, answered

*Does exact-QAT close the gap toward the envelope the way sync's does?*

- **On the trained object: yes, decisively.** The exact staircase composition
  trains to 0.9757–0.9774 TEST (−0.1…−0.3 pp from the 0.9785 AQ envelope) at
  L=9, in 3000 steps ≈ 40 s — where the shipped five-object recipe lands at
  0.9556–0.9614. The optimization-deficit hypothesis is CONFIRMED: the
  deployed composition family was never the bottleneck; the training object
  was.
- **On the raw deployment: only up to the V3 ceiling.** Count-STE achieves
  the sync-style parity-by-construction (1.0000) and, from the unfolded
  artifact with θ in-loop, reaches 0.9580 pre-quantization (vs the shipped
  post-WQ 0.9556) — but every method stays inside the 0.950–0.962
  raw-cascade band (§4.6); the ATan endpoint's 0.9614 is not beaten on the
  raw object at this budget.
- **On the re-timed deployment: yes.** 0.9747 (θ frozen) / **0.9751**
  (θ in-loop) TEST on the re-timed genuine cascade (torch twin of the C3
  mapping option; SCM/HCM replication is the §6.4(iii) gate, open), +1.9…
  +2.0 pp over the shipped 0.9556, −0.35 pp from the AQ envelope, with the
  trained staircase as the parity twin (−0.10/−0.23 pp train↔deploy). Toward the mandate: the
  remaining −0.9 pp to pretrain (0.9837) is now envelope territory
  (AQ 0.9785 − pretrain = −0.52 pp, plus the WQ leg to be re-measured
  through the exact forward), not conversion territory.
- **The fold-displacement lesson generalizes.** Arms entered from the folded
  state recover to 0.9524/0.9562; the same arms from the UNFOLDED artifact
  reach 0.9562/0.9580 — even 3000 recovery steps do not fully pay back a
  one-shot +θ/(2T) bake on a converged basin. Under the integrated arm the
  fold lands once at INSTALL (before any exact-QAT training), where it is an
  identity-grounded initialization instead of a displacement (P-L6).

## 6. Phase 4 — integration plan

### 6.1 The recipe arm (`lif_exact_qat`)

One knob in `_LIF_RECIPE_KNOBS` (`tuning/orchestration/conversion_policy.py:46-99`),
default off until tier-0 A/B, predicate-gated
`is_lif(mode) and firing_mode == "Default" and cycle_accurate_lif_forward`
(P-L5). What it changes, stage by stage (compose, don't fork):

1. **Shift step:** skip the lif-branch bias bake
   (`activation_shift_tuner.py:37-54`) under the exact arm — the QAT owns
   offsets (P-L6). One predicate; the ttfs branch is untouched.
2. **AQ step (currently inert for LIF):** becomes the exact-QAT install +
   ladder, mirroring sync's shape: install kernel (1) as the quantization
   decorator (`LIFCountStaircase` twin of `TTFSCeilStaircaseDecorator`,
   the `autograd.py` STE idiom; strict/inclusive per `thresholding_mode`
   via the `wire_semantics` pair), fold the half-step ONCE
   (`apply_lif_half_step_bias_compensation`, encoders excluded), install
   `ChipInputQuantizer` entry snaps (already the LIF tuner's seam), promote
   θ per §6.2, mark per-model like `mark_sync_exact_qat`
   (`adaptation_manager.py:58-78`) so the WQ-entry fold skips
   marker-assertedly (idempotency flag already exists). Ladder + endpoint
   train through the staircase forward — O(1) steps, so the existing
   budget buys ~8× the steps.
3. **LIF Adaptation step:** under `(A+R5)` it reduces to finalize+verify
   (rebuild LIF activations — deployment-identical by A2 —, install the
   chip-aligned forward, run the torch↔deployed parity probe); under raw
   deployment it keeps a count-STE endpoint: the same
   `run_endpoint_recovery` seam with the policy forward of §4.1(B) installed
   instead of `_ChipAlignedNFForward` (the count-STE walk is a
   `SegmentForwardDriver` policy — `spiking/segment_policies.py` gains a
   third policy class; the driver is untouched).
4. **WQ step:** unchanged projection (two-scale stays armed, R1) but the
   endpoint recovery trains through the exact forward (staircase for
   re-timed deployments, count-STE for raw). One-shot projection craters
   without recovery (E3/E6) — the existing endpoint IS the recovery; only
   its forward changes.
5. **Mapping:** arm `lif_per_hop_retiming` TOGETHER with the exact-QAT
   marker (the R5 note requires exactly this pairing,
   `conversion_policy.py:82-89`): the parity gate compares the trained
   staircase NF against the re-timed SCM — expected exact modulo per-hop V4
   and dtype ties (measured −0.10 pp logit-level here; the gate is
   argmax/per-neuron and should be re-baselined on a re-timed cell). Gate
   the arming on the temporal-A6 FAIL predicate + single-segment topology
   (the cells where V3 binds; S≥16 cells keep the raw deployment).
6. **Subsumed levers:** `lif_distmatch`/DFQ and the post-hoc
   quantile/affine/per-channel-θ armings stay off — their estimator is now
   inside the loop (§4.2).
7. **Membrane readout (C2) composes.** With a63dddc5's chip-realizable
   decode armed, the QAT loss should consume the SAME θ·c+m readout it will
   deploy (one decode function shared by the policy walk and the SCM/HCM
   logits path) — then V2 never needs to be absorbed by training at all,
   and the P-L2 precondition generalizes from "counts both sides" to
   "one decode, both sides". All numbers in this memo are counts-decode;
   re-measure the arms under the armed decode before A/B.

### 6.2 θ export and the per-channel seams

Prototype-validated constraint set, carried into the arm: per-channel θ only
on hops whose output stays inside a neural segment on a matching-axis edge
(the R3 exact set — decode folded into consumer `per_input_scales`,
`scale_propagation.py:62-74`); scalar-trainable θ on externally-consumed
hops (host boundary mean-collapse seams, `scale_aware_boundaries.py:31-40`);
encoder θ frozen (boundary contract). Export via threshold groups
(`mapping/packing/canonical.py:96-101`) with the capacity dry-run oracle
(`capacity/dryrun.py`) as the enqueue gate — per-neuron thresholds fragment
packing groups (R3's known cost). The E4 plumbing (vector-aware ComputeOp
scales, lane-aware per_input_scales) widens the per-channel set later;
nothing in the arm depends on it.

### 6.3 Genericity

- **Any L:** the trained object is the composition itself; E1 converges at
  L=9 in 3000 O(1) steps and nothing in (1)–(4) carries a depth constant;
  the count-STE anchor's residual grows with single-segment depth exactly as
  the temporal gauge measures — which is the arming predicate, not a new
  constant.
- **Any vehicle:** the policies ride `SegmentForwardDriver`'s exec-graph walk
  (joins covered by P-L4 relays; multi-segment vehicles already re-time at
  ComputeOp boundaries, which is why the deepcnn/deepmlp/lenet LIF cells
  pass today — the single-segment mixer trunk is precisely where (A+R5)
  binds). Convs ride the same perceptron/staircase kernels.
- **Tier-1/2:** no workload constants anywhere (the kernels, the STE, the
  fold, and the gauges are data-free; LRs/budgets are the existing recipe
  constants). The wall cost SHRINKS vs the shipped arm (staircase steps are
  ~8× cheaper than the O(S) cycle-accurate steps that currently throttle
  the endpoints; count-STE ~2× cheaper).

### 6.4 Verification locks (new)

(i) exact-kernel ≡ LIF-node A2 lock (constant-drive bit-parity over both
comparators, T ∈ {4,8,16,32}); (ii) count-STE policy ≡ chip-aligned forward
per-neuron lock (extends `nf_scm_parity`); (iii) re-timed SCM ≡ staircase NF
gate for (A+R5) cells; (iv) fold-once assertion via the existing
`LIF_HALF_STEP_FLAG` + exact-marker (mirror
`soft_core_mapping_step.py:428-446`); (v) G7 engagement witnesses for every
arm (`lif_exact_qat: {installed, folded, theta_promoted, retimed}` reporter
line per cell).

### 6.5 Risk register

| risk | evidence | mitigation |
|---|---|---|
| Goodhart on the staircase proxy (raw deployment) | E1/E2: −2.5 pp train↔deploy | never gate keep-best on the staircase read for raw cells — deployed/count-STE read only; (A) reserved for re-timed cells |
| Re-timing latency/energy | R5: mapping-visible; sim_time ≈ S×hops for the trunk | gate on temporal-A6 FAIL + single-segment; report the mapping delta; S≥16 cells stay raw |
| WQ projection displacement | E3 0.48 / E6 0.81 one-shot | endpoint recovery through the exact forward is mandatory (already the pipeline shape); consider projected-STE rungs later |
| per-channel θ export fragmentation | R3 known | dry-run oracle at enqueue; scalar fallback per hop |
| from-float basins (prototype started from the adapted artifact) | untested | integration A/B: exact-QAT from the post-AQ float state vs post-adaptation state; keep the T-anneal ramp as a config-armable warm start until refuted |
| Shift-bake removal interaction (fp/no-AQ lif plans) | S1 bake predates AQ | scope the skip to the exact arm; fp plans keep today's path |
| dtype ties at strict "<" (P-L3) | parity 1.0000 measured, unguaranteed | existing parity gate; V9 lattice hazard note stands |
| Novena | Theorem 0 broken | excluded by predicate (P-L5) |

## 7. Relation to prior artifacts; refuted branches

- `lif_deployment_exactness.md`: supplies Theorems 0–3 and the V-ledger;
  this memo adds the measured statement that V3 is *training-steerable*
  (±15× across weight sets) — the commutation theorem is a per-weight-set
  identity, not a uniform bound, and exact-QAT must therefore anchor on the
  deployed forward (B) or restore the preconditions (A+R5).
- `sync_deployment_exactness.md`: Theorem 1 is the template; the LIF twin
  swaps the kernel pair and adds the timing anchor. The §3 composed-error
  laws transfer verbatim to arm (A)'s trainability argument (§4.4).
- `lossless_refinement_ledger.md`: R5 is re-adjudicated (the blocker
  dissolves under exact-QAT twinning — measured +2.4 pp on E1 weights);
  R3's inversion is dissolved in-loop (§4.2); the fold-leg loss line is
  reproduced and mechanistically explained (P-L6); R1/R8 unchanged.
- **Refuted here (do not re-derive):** pure-staircase exact-QAT as a raw-
  deployment lever (E1/E2, −2.5 pp train↔deploy); one-shot WQ projection on
  exact-QAT weights without recovery (E3/E6); "the LIF ramp needs the
  T-anneal to converge at target T" (both arms train directly at T=8 from
  the adapted basin); "θ in-loop destabilizes the staircase QAT" (E2/E5:
  monotone keep-best, bounded drift, +0.4 pp on the parity object).
- **Open (next round):** count-STE from the post-AQ float state; the WQ
  endpoint through the exact forward (the E3/E6 recovery leg); a re-timed
  tier-0 cell end-to-end through SCM/HCM (the §6.4(iii) gate); S=4
  re-adjudication after θ-in-loop (R10).

## 8. Addendum (2026-07-13): §6.5 follow-up measurements (WS-Z) + on-pipeline status

**Arming landed** (commit 6b379b07): tier-0 A/B 10/10 runnable cells hold or
gain (ledger §6). Two §6.5 open levers measured on the trained composition
(memo §5.0 protocol, budget-matched 3000 steps, 3 seeds, current-integration
seams: R3 θ promotion + entry snap + fold-once):

- **From-float basin entry: REFUTED.** Ties at S=8 (−0.85 SEd) and on the
  fc64 S=4 aux cell (+0.19 SEd); LOSES at the decisive fc128 S=4 cell
  (−3.0 SEd, −0.15 pp). Mechanically: from-float enters at 0.6141 staircase
  (vs 0.8721 adapted) at T=4 — the Shift/AQ legs buy staircase-compatible
  weight structure that from-float forfeits. The adapted-basin entry stands.
- **KD teacher at the exact endpoint: pretrain-float teacher WINS (small,
  consistent).** Positive in all three cells, >1 SE in both S=4 cells
  (fc128 S=4: 0.97113±0.00032 ctrl → 0.97190±0.00064; fc64 S=4 +2.29 SEd);
  no-KD is the WORST arm (−1.70 SEd below the AQ-teacher control).
  **Mechanism: the endpoint is pinned by its KD target** — with the AQ
  teacher the S=8 endpoint saturates at the AQ envelope (0.9783 ≈ teacher's
  0.9785); the pretrain teacher (0.9837) raises it, more where the residual
  is larger.
- **On-pipeline gap:** the AQ-hosted exact-QAT currently trains with PLAIN
  CE (`pipeline.loss`; the recipe's kd_ce_alpha/kd_temperature are consumed
  only by the KD-blend family, which the exact arm reduces to
  finalize+verify) — i.e. the pipeline runs the measured WORST KD arm. The
  `lif_exact_qat_kd` knob (reference-teacher snapshot step + AQ-tuner loss
  swap) is in implementation; probe A/B before recipe arming.
- **Residual decomposition after teacher fix (pre-WQ):** inherited float-leg
  AQ gap (−0.52 pp) + S=4 grid cost (~−0.66 pp) — further closure needs a
  different-type lever (not entry/teacher choice).
- **WQ endpoint through the exact forward (was Open): CONFIRMED WORKING
  on-pipeline** — the tier-0 A/B endpoints climb (t0_01 0.9300→0.9519,
  t0_04 0.9307→0.9558, t01_21 0.9446→0.9638; t01_19 reaches in 716 steps);
  the one frozen endpoint left is t0_03 (0.9896 entry==exit, G5).

## 9. Addendum (2026-07-13): KD-teacher lever REFUTED on-pipeline (6th inversion)

`lif_exact_qat_kd` implemented (commit 6ec3f034, default OFF): the AQ-hosted
exact-QAT distils to the post-structural float teacher (a Reference Teacher
Snapshot step, post-Scale-Migration). On-pipeline A/B (6 cells vs v7 baselines,
`scripts/_probes/exact_qat/xk_*.json`):

| cell | S | KD | v7 base | Δ |
|---|---|---|---|---|
| t0_01 mixer | 4 | 0.9578 | 0.9548 | +0.30 |
| t01_08 mixer | 4 | 0.9578 | 0.9548 | +0.30 |
| t01_21 mixer wb8 | 4 | 0.9546 | 0.9616 | −0.70 |
| t01_01 mixer | 8 | 0.9647 | 0.9698 | −0.51 |
| t01_02 mixer | 16 | 0.9736 | 0.9743 | −0.07 |
| t0_05 simplemlp (control) | 4 | 0.9825 | 0.9814 | +0.11 |

**Mean effect across the 5 mixers −0.14 pp; MIXED and net-negative.** The §8
isolated-harness WIN (3000-step budget-matched, no WQ endpoint, no draws)
INVERTS on the full composition — the 6th measured instance of the meta-lesson
(after affine fold, sync first-moment fold, per-channel θ on both modes,
sync-mixer envelope fold). Mechanism: on-pipeline the KD loss also flows through
the WQ endpoint-recovery leg and the best-of-N draws, and the float teacher's
decision boundary is one the staircase-constrained student cannot represent at
S≥8 (soft targets pull off the representable staircase); at the coarse S=4 grid
the softness sometimes regularizes (t0_01/t01_08 +0.30) but not reliably
(t01_21 −0.70). The control holds (+0.11), so the mechanism is correct and
non-destructive — just not a reliable lever. **VERDICT: do NOT arm in the
recipe** (fail-toward-measured); keep config-armable for research. The mixer
residual's binder is confirmed the AQ envelope position itself, not the KD
target — closing it needs a lever that changes the *representable* staircase
capacity, not the training target.
