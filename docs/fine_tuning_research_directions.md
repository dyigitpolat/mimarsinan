# Fine-Tuning Systems for Spiking Deployment — Research Directions

**Status:** living design document. Written 2026-06-08 after the cascaded
`ttfs_cycle_based` fine-tuning investigation (see
`docs/behavior_contract_unification.md` §9 for the incident record). This
document is forward-looking: it states the open problem precisely, gives the
measurements that bound it, and proposes concrete, prioritized research
directions for a better fine-tuning system. It deliberately does **not**
re-describe the shipped pipeline — that lives in the per-module
`ARCHITECTURE.md` files.

---

## 1. The problem, stated precisely

The deployment pipeline fine-tunes an ANN into a spiking network by ramping
each perceptron's activation from its continuous form toward an on-chip spike
node (`LIFActivation` / `TTFSActivation`), with KD recovery against a frozen
pre-step teacher (`KDBlendAdaptationTuner`). For most modes this reaches the
deployed accuracy: LIF (subsume/offload), TTFS, `ttfs_quantized`, and
**synchronized** `ttfs_cycle_based` all clear the ≥0.97 MNIST/mmixcore
acceptance gate.

**One configuration does not: cascaded `ttfs_cycle_based`** (both subsume and
offload), which plateaus at **0.94–0.96** simulation accuracy. After the §9
bug fixes (effective-bias SSOT, stale-mapping guard, gradual ramp,
LR-refind stabilization) the residual is no longer a seam or an implementation
error — it is an **optimization/representation gap intrinsic to training a
single-spike cascade through a value-domain proxy**. This document is about
closing that gap.

### 1.1 Why cascaded `ttfs_cycle` is special

| schedule | deployed inter-stage dynamics | differentiable proxy used by the ramp | proxy fidelity |
|---|---|---|---|
| LIF (cycle-accurate) | multi-spike rate code, signed IF | value-domain `BlendActivation` → chip-aligned forward at finalize | **high** (rate code ≈ mean) |
| synchronized `ttfs_cycle` | per-group analytical staircase composition | the ramped class forward **is** that composition | **exact** |
| **cascaded `ttfs_cycle`** | **single-spike, ramp-integrate, fire-once cascade** with sub-window spike timing | value-domain **staircase** (`TTFSStaircaseFunction`) | **low** — see §2 |

The deployed cascade is faithful to hardware (NF↔SCM per-neuron agreement is
exactly 1.0; SCM = HCM = nevresim = SANA-FE). The problem is that the function
the fine-tuner *optimizes* (the pointwise staircase) is not the function that
*deploys* (the timing-coded single-spike cascade), and at S=4 the two diverge
materially.

---

## 2. Measurements that bound the problem

All on mmixcore (`patch_n=patch_m=4`, `patch_c=128`, `fc_w=64/128`), S=T=4,
MNIST, seed 0, post-§9-fixes.

1. **Finalize cliff is large and high-variance.** `_finalize_cliff` =
   (proxy accuracy at ramp rate 1) − (genuine cascade accuracy immediately
   after the finalize swap), measured across runs: **0.23, 0.24, 0.42, 0.65**.
   The proxy optimum sits far from the genuine-cascade basin, and *how* far is
   itself model-draw-dependent. LIF's chip-aligned finalize cliff is ≈0.

2. **Per-layer proxy↔genuine divergence is structural, not noise.** Decoded
   per-perceptron values, genuine cascade vs staircase proxy on the same
   weights (permutation-aligned): mean-ratio **0.54–1.41**, mean |Δ| 0.06–0.19
   on values whose own mean is 0.11–0.41. The cascade systematically
   *attenuates* deep layers (back-loaded ramp integration: late-arriving
   single spikes contribute a shorter membrane ramp), compounding with depth.

3. **The deployed model carries a generalization gap, not just a transfer
   gap.** On one trained checkpoint: genuine-cascade **validation 0.972** vs
   **test 0.9345** — a 3.7 pp val→test gap on identical weights. The genuine
   cascade's harder decision surface (quantized to S=4 timing) overfits the
   tuning split.

4. **Error structure: pure optimization gap, not tie-breaking.** On 2000 test
   samples the deployed cascade has **zero argmax ties** and **zero
   1-quantum near-misses** — the errors are confident, not boundary flips. So
   tie-breaking / dithering at *inference* cannot help; the fix must be in
   *training*.

5. **What the levers bought (controlled, same 0.9824 pretrain via resume):**
   offload sim **0.916 → 0.954** (bias fix + stale-mapping fix + stabilization
   rounds); NF destruction (0.9468→0.9199) **eliminated**. Subsume FT
   0.9457 → 0.9616.

6. **What did NOT help (rejected with evidence):**
   - *Genuine-gradual ramp* (output blend `(1−r)·continuous + r·genuine`):
     holds ≥0.96 to r=0.9999 then r=1.0 is **catastrophic** — the
     ε-weighted continuous term is a margin oracle that resolves the cascade's
     thin logit gaps, so it cannot hand off at r=1. Net regression.
   - *Deployment-aware θ-calibration* (greedy per-layer `activation_scale`
     search through the genuine forward): +2.1 pp when selected on test
     (selection bias); honest val→test transfer **+0.08 pp**. Reverted.
   - *Pretraining best-val checkpoint*: regressed pretraining ~0.9 pp on the
     annealing cosine schedule (best-val epoch ≠ best-test epoch). Reverted.
   - *2× tuning budget*: within run noise for the FT step.

**Conclusion the measurements force:** the binding constraint is that
**gradient information about the deployed single-spike dynamics never reaches
the optimizer in a well-conditioned form.** The proxy is well-conditioned but
wrong; the genuine cascade is right but its surrogate gradients (single
fire-once Heaviside per neuron across T cycles) are sparse and noisy, landing
in a worse basin when descended directly (§8.3 of the contract doc).

---

## 3. Design principles for a better fine-tuning system

1. **Train the deployed function, conditioned well.** The endpoint of fine
   tuning must be the genuine cascade, but the *path* there must carry dense,
   low-variance gradients. A proxy is acceptable only insofar as it shares the
   genuine cascade's *local* gradient direction, not merely its forward value.

2. **Curriculum over the timing resolution, not just the blend rate.** The
   ramp today interpolates ANN→SNN *activation shape* at fixed S. An orthogonal
   and likely more powerful axis is interpolating *temporal resolution*
   (S=∞ continuous → S=4 deployed): a high-S cascade is nearly the staircase
   and well-conditioned; annealing S down walks the optimizer into the
   deployed basin while keeping each step small.

3. **Keep the contract SSOT.** Any new proxy/forward must route through the
   existing `SpikingDeploymentContract` / `SegmentForwardDriver` policies so
   NF↔SCM parity stays a hard gate. No new parallel walk.

4. **Measure on a held-out split, always.** §2.3/§2.6 show how easily a lever
   looks good on test via selection. Every proposed lever below must report
   honest val→test transfer with a fixed val split before it ships.

---

## 4. Research directions, prioritized

### D1 — Temporal-resolution curriculum (S-annealing) — **highest leverage**

**Idea.** Replace (or precede) the activation-blend ramp with a curriculum over
the cascade's timing resolution S. Train the genuine cascade at large S₀ (e.g.
32 or 64), where the single-spike ramp closely approximates the analytical
staircase and surrogate gradients are dense, then anneal S → S_deploy (4) in
stages, recovering at each S. The forward is the *genuine cascade at the
current S* throughout — no proxy, no finalize cliff.

**Why it should work.** §2.1's cliff is the distance proxy→genuine at S=4.
At S=64 that distance is ~0 (the staircase is the S→∞ limit). Annealing keeps
every step inside a recoverable basin while always optimizing the true
dynamics. It directly attacks §2's root cause (well-conditioned gradients about
the deployed function).

**What it needs.** `TTFSActivation` already takes T as a constructor arg; the
segment driver is S-parametric. Required: a schedule object that re-instantiates
the spike nodes / driver at each curriculum S with KD recovery between stages,
and a check that deploy-S is the last stage. Risk: cost (training at high S is
T× the cycles); mitigate by short recovery per stage and few stages
(64→16→8→4).

**Falsifiable test.** Cliff at each curriculum stage should be ≪ the one-shot
S=4 cliff (0.23–0.65); final test ≥ the proxy-trained baseline + 1 pp with
honest val selection.

### D2 — Distributional / soft-spike proxy with matched gradients

**Idea.** Build a differentiable proxy whose *forward* and *gradient* both match
the single-spike cascade's expected behavior under spike-time jitter — e.g. a
soft-argmax over fire cycle (temperature-annealed) rather than the hard
fire-once Heaviside, or a closed-form expectation of the ramp-integrate decode
over the input spike-time distribution. Unlike the staircase (§2.2), this proxy
models sub-window timing, so its optimum transfers.

**Why it should work.** The staircase ignores intra-window timing; that is the
entire §2.2 divergence. A timing-aware soft proxy removes the proxy↔genuine gap
*by construction* while staying smooth.

**Risk.** Deriving a faithful, cheap closed form for the cascade decode under
the actual input spike-time distribution is non-trivial; an approximate one may
just relocate the gap. Validate against the genuine cascade per-layer (§2.2
metric) before trusting it.

### D3 — Variance-reduced surrogate gradients for the genuine cascade

**Idea.** Keep the genuine forward but make its gradient usable: (a) multi-spike
relaxation during training (allow k>1 spikes, anneal k→1) so the Heaviside is
hit more often per neuron; (b) surrogate-width annealing (wide ATan early,
narrow late); (c) per-cycle gradient checkpointing with antithetic spike-time
sampling to cut variance.

**Why it should work.** §2's "worse basin when descended directly" is a
gradient-quality problem; these are standard SNN-training fixes
(Neftci/Zenke-style) not yet applied to this cascade.

**Risk.** Multi-spike relaxation changes the dynamics — must collapse exactly to
single-spike at k=1 (parity gate guards this).

### D4 — Stabilization as the primary training phase (SWA + longer genuine run)

**Idea.** Today the genuine cascade is trained only in post-finalize
stabilization (now multi-round with LR-refind). Promote it to a first-class
phase: Stochastic Weight Averaging across the stabilization rounds, a cosine
restart schedule, and a budget sized to the genuine phase (not 2× the ramp
budget). SWA specifically targets the §2.3 generalization gap (flatter minima).

**Why it should work.** Cheap, orthogonal to D1–D3, and directly addresses the
val→test gap (§2.3) rather than the transfer gap. Likely a 0.5–1 pp win on its
own.

**Risk.** Low. This is an incremental extension of the shipped stabilization
loop; the rollback guard keeps it safe.

### D5 — Regularize the genuine cascade toward robustness

**Idea.** During genuine-cascade training, add: label smoothing, input/spike-time
noise (`enable_training_noise` already exists as a hook), and a KD term against
the *analytical-staircase* output (not just the ANN teacher) so the cascade is
pulled toward the well-behaved proxy's decisions where they agree.

**Why it should work.** §2.3 says the deployed model overfits; §2.4 says errors
are confident. Both are classic over-confidence symptoms that smoothing/noise
address. The staircase-KD term injects the proxy's better-conditioned signal
without making the proxy the forward.

**Risk.** Over-regularizing caps the ceiling; sweep strengths with honest val.

### D6 — Architecture / mapping co-design (longer horizon)

**Idea.** The cap is partly that S=4 gives only 4 timing levels. Options:
(a) raise deploy-S where the power/latency budget allows and quantify the
accuracy/cost Pareto front; (b) prefer the **synchronized** schedule for
`ttfs_cycle` deployments (it already clears 0.97) and treat cascaded as the
power-optimized variant with a documented accuracy trade; (c) mapping-level
changes that shorten cascade depth (the §2.2 attenuation compounds with depth).

**Why it should work.** Directly enlarges the representational ceiling rather
than optimizing within it. (b) is available *today* as a deployment policy.

---

## 5. Recommended sequence

1. **D4 + D5 first** (cheap, low-risk, orthogonal, ~1–2 pp expected): SWA over
   stabilization, label smoothing, staircase-KD term. Each gated on honest
   val→test transfer.
2. **D1 (S-annealing curriculum)** as the main bet — it attacks the root cause
   and, if it works, makes D2/D3 unnecessary.
3. **D3** if D1's high-S stages are too costly (variance reduction lets a
   shorter/lower-S curriculum work).
4. **D2** only if D1+D3 still leave a gap — it is the most research-heavy.
5. **D6(b)** is the pragmatic fallback: ship synchronized for accuracy-critical
   `ttfs_cycle` deployments and document the cascaded power/accuracy trade.

## 6. Guardrails for any new work here

- Route every forward through `SpikingDeploymentContract` / `SegmentForwardDriver`
  policies; NF↔SCM per-neuron parity (currently 1.0) is a hard regression gate.
- Report honest val→test transfer with a fixed validation split; never select a
  hyperparameter or scale on the test set (§2.3/§2.6 are cautionary tales).
- Add contained unit tests for any new proxy/forward/curriculum BEFORE wiring it
  into a tuner (the project discipline), and a parity test against the genuine
  cascade per-layer decode (§2.2 metric) for any new proxy.
- Keep the `_finalize_cliff` and `_phase_seconds` instrumentation; they are the
  cheapest signal for whether a change closed the transfer gap.

---

## 7. Implementation status (2026-06-09)

What landed is the **shared-primitive refactor** of the tuning system; the D1
S-annealing *strategy* was prototyped, evaluated, found not to help on the target
config, and **removed** (kept only as the empirical result + the diagnostic
below).

**Refactor / SSOT primitives (kept).** The tuning mechanics are now shared,
axis-agnostic helpers rather than per-tuner copies — this is the genericity any
future transformation (including a hypothetical RRAM-noise tuner) reuses:
- `tuning/perceptron_rate.py` (`apply_manager_rate`, `rebuild_activations`,
  `set_blend_rate`) — one place to apply a rate across perceptrons, replacing 4+
  inlined `setattr; for p: update_activation` loops.
- `tuning/teacher.py` (`snapshot_frozen_teacher`, `freeze_module`) and
  `tuning/forward_install.py` (`LazyExecutorForward` with `_ensure_executor`,
  `CascadeForwardInstall`) — leaf modules the tuners share for KD-teacher capture
  and `model.forward` install.
- **Bug fix:** the pipeline-floor check (`_ensure_pipeline_threshold`) now lives
  once in `AdaptationRateTuner._after_run` — the activation-quant / noise tuners
  used to each re-add it, and the base path silently skipped it.
- LIF customizes finalize through ordered hooks (`_before/_after_finalize_rebuild`)
  instead of copying the base `_finalize` body.
- Continuous-rate axes (activation blend, decorator rate, and a future RRAM-noise
  ramp) remain generic via the existing `SmoothAdaptationTuner` + `AdaptationManager`
  rate-field extension point — a new axis is a thin `AdaptationRateTuner` subclass.

**D1 (S-annealing) — evaluated and rejected for cascaded offload mmixcore.** A
prototype (staircase-proxy ramp at a large `S₀`, genuine-cascade finalize, then
anneal `S → deploy` via a per-stage KD recovery) was run on
`mnist_mmixcore_ttfs_cycle_60_offload`, seed 0, deploy `S=4`:

| | baseline (no curriculum) | curriculum `[16,4]` |
|---|---|---|
| finalize cliff | **0.205** | **0.746** |
| deployed (Soft/Hard-core sim) | **0.946** | **0.918** |

D1 **hurt by −2.8 pp**, and the mechanism refutes its premise: the finalize cliff
(staircase-proxy minus genuine-cascade accuracy) **grew** with `S` (0.205 at S=4 →
0.746 at S=16) instead of shrinking toward 0. So annealing *down* from a higher-`S`
basin lands below training at deploy-`S` directly. This is consistent with the
encode/decode finding below and the §2.2 structural divergence. The prototype was
removed; the config knob, the curriculum machinery, and the RRAM glimpse were all
deleted (they were never requested as shipped features).

**Encode/decode diagnostic (kept).** A contained study (`tests/cascade_fixtures.py`
+ `tests/unit/tuning/test_cascade_encode_decode.py`, host-side compute ops
before/between/after neural segments) pins the genuine cascade's gradient contract
across segment boundaries:
- A **subsume** segment entry charges the decoded value directly → the boundary is
  **differentiable**, gradient crosses upstream.
- An **offload / host-ComputeOp** entry reads a re-encoded single-spike train
  (`round`-based TTFS encode, `segment_policy_ttfs._boundary_single_spike_train`)
  → gradient to upstream segments is **severed**.

So genuine-cascade training reaches all layers in a single subsume segment but only
the *last* segment once host ComputeOps cut the graph (the real mmixcore target).
This is the structural mechanism behind §2.2's deep-layer attenuation and behind
D1's failure here. The real next lever for cascaded multi-segment models is making
the boundary re-encode a straight-through/surrogate so the genuine backward trains
every segment (the D2/D3 work; parity-critical — the forward must stay the bit-exact
HCM encode). `TestEncodeDecodeGradientContract` pins the current contract so any
such change is deliberate. For accuracy-critical `ttfs_cycle` today, D6(b)
(synchronized, already ≥0.97) remains the pragmatic path.

## 8. Implementation status (2026-06-17)

The D2/D3 **boundary straight-through estimator** was built and **evaluated — refuted
as an accuracy lever** (kept default-off as the gradient-flow primitive, NOT retired):

- **What landed (`ttfs_boundary_surrogate`, default off; `_temp=1.0`).** A straight-
  through estimator on `segment_policy_ttfs._boundary_single_spike_train`: forward is
  the **bit-exact** `round` train; backward flows through a soft spike-time
  (`sigmoid((c − T(1−v))/temp)`), so the genuine cascade gradient reaches EVERY
  segment, not just the last. Threaded into both genuine training forwards
  (`_SegmentSpikeForward` for stabilization/deploy, `BlendedGenuineForward` for the
  blend ramp) via `TtfsSegmentPolicy.boundary_surrogate_temp`. Skipped under
  `no_grad`. Unit-tested: gradient now flows upstream past an offload boundary
  (`test_offload_boundary_surrogate_propagates_gradient_upstream`) and the forward is
  byte-identical (`test_surrogate_leaves_forward_bit_exact`). NF↔SCM parity stays 1.0.
- **Why refuted.** A/B on the mmixcore S=8 genuine-blend-fast ramp (resume from a
  fixed ANN=0.982 cache): **subsume** — deployed SCM {off 0.9321/0.9375, on
  0.9361/0.9313}, fully overlapping (subsume entries are already differentiable, so
  there is little severed gradient to recover). **offload** (severance maximal, every
  segment entry re-encoded) — deployed SCM {off 0.913, on 0.9094/0.9091}, also within
  noise (FT {off 0.8652/0.8738, on 0.8783/0.8724} overlapping). The severance is real
  and correctly un-severed, but it is NOT the binding accuracy constraint: the
  optimizer recovers enough via the last-segment gradient + the proxy/pretrained init,
  and the dominant gap is the §2.2 forward attenuation + §2.3 generalization gap, which
  a backward-only surrogate cannot touch. A *well-conditioned* timing-aware proxy (true
  D2) — not this soft-sigmoid STE — remains the open bet if the optimization gap is to
  be closed.
- **The lever that DID move accuracy: more genuine training (D4).** `ttfs_blend_fast_
  steps_per_rate` 120→300 lifted deployed **0.9365→0.9456 (+0.9 pp, above the ~0.5 pp
  run-to-run noise)** on the genuine-blend path — no new code. This is the cheapest
  real cascaded-TTFS accuracy lever measured to date; promote it for accuracy-critical
  runs (the proxy path at ~0.95 is the better starting baseline to apply it to).
- **LIF, for contrast (fully lossless).** Deployed *tracks* the ANN; more deployed-
  cascade training (`lif_blend_fast_stabilize_steps` 400→6000) climbs monotonically to
  **0.9784 ≥ ANN 0.9776**. DFQ per-neuron bias matching *hurts* LIF (0.972→0.962) —
  the LIF rate code has no death cascade to revive, unlike TTFS, so first-moment
  matching only perturbs the trained classifier. `lif_distmatch` shipped default-off.
  The shared DFQ core (`spiking.dfq_bias_correction`) is the Boy-Scout refactor TTFS's
  `distribution_matching` now reuses.

**Higher S is NOT a practical lever (clean test).** A fixed-ANN (0.982) proxy S-sweep
— resume the SAME pretrained model at Torch Mapping, vary only `simulation_steps`,
deploy → Soft Core Mapping — gives **S=8 → 0.9344, S=16 → 0.9366 (+0.22 pp for a 2×
timing-resolution doubling)**. Reaching LIF level (~0.975) from 0.934 would need ~18
doublings. Prior cross-S "evidence" (proxy S4 0.9501 / S8 0.9513; genuine S8 0.9365 /
S16 0.9338) was confounded by different ANN draws; this fixed-ANN test settles it.
Combined with the refuted STE and the modest more-training (D4, +0.9 pp) lever, the
single-spike cascade is **representation-limited** at ~0.93–0.95 — the §2.2 forward
attenuation (a deployed-dynamics property the bit-exact forward cannot change) caps it.

**Strategic conclusion.** For accuracy-critical `ttfs_cycle`, the **synchronized**
schedule (D6(b), already ≥0.97 = LIF level) is the path; cascaded is the
power-optimized variant carrying a documented ~2–3 pp accuracy trade. Closing that
trade needs a *fundamental* advance, not the levers tried here:
- (a) a timing-aware soft-spike proxy with matched gradients (true D2) — the crude STE
  proved "gradient reaches ≠ gradient well-conditioned"; a proxy modelling sub-window
  timing is the open bet for the optimization gap;
- (c) mapping co-design to shorten cascade depth (D6(c)) — directly attacks the §2.2
  depth-compounding attenuation, the binding *representation* limit.
The §2.3 generalization gap appears narrowed on the current pipeline (FT-val ≈
SCM-test on the genuine-blend runs), so D5/SWA is lower priority than D2/D6(c).
