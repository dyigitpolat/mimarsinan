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
