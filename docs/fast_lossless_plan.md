# Making the Non-Destructive Gradual Path FAST *and* LOSSLESS — A Research & Engineering Plan

**Author framing:** principal-ML strategy memo, written after four review rounds of the `mimarsinan` tuning subsystem.
**Companion docs:** `transformation_adaptation_spec.md` (the behavioral spec), the three review documents, and the team's own `tuning_accuracy_retention_report.md`.
**Thesis in one sentence:** *Lossless is an information-theoretic and conversion-calibration problem, not a controller problem; fast is a "schedule-not-search, reconstruct-don't-finetune" problem — and the previous work, though excellent, optimized the controller and the loss-blend while leaving the two governing levers (channel capacity and layerwise conversion fidelity) mostly untouched.*

---

## 0. Reframe: What We Are Actually Optimizing

Two goals are being conflated, and they are governed by different physics. Separate them cleanly before designing anything.

**LOSSLESS** = the *deployed* model (the real spiking cascade running on `nevresim`, evaluated on the **full** test set) retains the source ANN's accuracy to within evaluation noise. This is bounded by two things and nothing else: (i) the **channel capacity** of the spiking code at each layer — you cannot represent more information than the code carries; and (ii) the **fidelity of the value→spike conversion and the trainability of the discrete forward** — given adequate capacity, how close can you push the deployed forward to the ANN.

**FAST** = the wall-clock of the transformation *process* (calibration + any fine-tuning + evaluation + search). This is dominated today by (i) the **LR-finder** and per-cycle **recovery training**, (ii) **evaluation** passes, and (iii) the **serial discrete cycle/rollback** structure of the controller.

These are largely orthogonal. You can be fast-and-lossy (the default proxy ramp), slow-and-lossless-ish (the genuine controller ramp), or — the goal — fast-and-lossless. The plan attacks each governing lever directly.

### 0.1 The myopia diagnosis (said plainly, with respect)

The previous engineer did genuinely excellent work: the teacher→genuine `BlendedGenuineForward`, DFQ distribution matching, the scale-aware boundaries, the encoding-layer parity fix, the non-stalling ratchet, the de-fooled probe, and the deployment-parity gates are all correct and load-bearing. But the work stayed inside one paradigm — *"gradually deform a finished ANN, step by step, under an adaptive controller, recovering with fine-tuning"* — and inside that paradigm it optimized the **controller** (floors, rollback, LR tax) and the **training loss** (the blend + genuine-CE). Four things outside that frame went under-examined, and they are exactly where fast-and-lossless is won:

1. **Channel capacity (the S-step ceiling).** The team correctly *identified* a ~2.7 pp S=4 representational ceiling and then labeled it "orthogonal." It is not orthogonal — it is the **hard upper bound on lossless**. No controller, blend, or fine-tune can cross it. This must be led with, not parked.
2. **Conversion theory.** LIF converts *bit-exact* (0.958) while cascaded TTFS cliffs at ~0.26. That asymmetry is a textbook conversion-fidelity signal, not a tuning mystery — and the established ANN→SNN conversion toolkit (threshold/scale balancing, soft-reset, layerwise reconstruction) was largely bypassed in favor of end-to-end blended fine-tuning.
3. **The teacher as a layerwise oracle.** The ANN teacher is used only as an *output* KD target. It is a per-layer oracle: it can re-anchor every layer's statistics and stop the error-compounding "death cascade" via **layerwise reconstruction** and **feature distillation** — which are also far cheaper than end-to-end fine-tuning.
4. **Schedule vs search.** The team's own experimental fast path proved that, once the homotopy is well-conditioned, a *fixed* rate schedule with one optimizer reaches the same accuracy as the whole controller. That is the FAST headline, and it was left as an "experimental flag" instead of becoming the design principle.

The rest of this document develops these four into a concrete program.

---

## 1. The Lossless Ceiling: a Per-Layer Capacity / Rate-Distortion Budget

**Lead here, because it is non-negotiable physics.** Before any training, compute what is *achievable*, so effort is never spent fighting an information wall.

### 1.1 What each code can carry

- **TTFS with `S` time steps:** a single spike lands in one of `S` bins (or never), so it carries at most `log2(S+1)` bits per activation. At `S=4` that is **≈2.3 bits** — versus the ANN's effectively-continuous activations. Worse, TTFS is a *nonlinear, fragile* code (value↔time is monotone but the precision is non-uniform and timing errors compound across layers).
- **Rate-coded LIF with `T` steps:** firing count in `[0,T]` → `log2(T+1)` bits, but the code is a *linear, robust temporal average*. With threshold/weight balancing the conversion error is `O(1/T)` and trivially calibratable — which is precisely why LIF is bit-exact in this repo and TTFS is not. Same nominal bits, completely different effective fidelity.

The lesson: **bit count is not the binding constraint; the code's linearity and robustness are.** A 2-bit *robust linear* code can be near-lossless for many layers; a 2-bit *fragile nonlinear* code (TTFS@S=4) is not.

### 1.2 The budget procedure (a concrete, cheap pre-analysis)

For a given model + dataset, before transforming:

1. **Per-layer sensitivity to capacity.** Sweep each layer independently from fp → the candidate code at a few `S`/bit settings, *holding all other layers at fp*, and measure the end-to-end accuracy drop on a fixed calibration subset (this is the standard "per-layer quantization sensitivity" sweep; cheap, embarrassingly parallel). This yields a sensitivity curve `Δacc_ℓ(S)` per layer.
2. **Hessian-trace weighting (optional, sharper).** Where the sweep is ambiguous, use a Hessian-trace / curvature proxy per layer (the HAWQ-style sensitivity metric) to rank layers by how much a representational error there hurts the loss. Cheap via Hutchinson trace estimation.
3. **Allocate non-uniform capacity.** Solve a small knapsack: minimize total temporal cost `Σ S_ℓ` (the chip's time-step budget, which sets latency/energy) subject to `Σ Δacc_ℓ(S_ℓ) ≤ ε_lossless`. Early/encoding and high-curvature layers get more `S`; robust late layers get less. This is **mixed-temporal-resolution allocation** — the temporal analogue of mixed-precision quantization.
4. **Declare the achievable floor.** If no allocation within the chip's `Σ S` budget meets `ε_lossless`, then *lossless is impossible at this budget* — and the output of this step is a Pareto curve of `(Σ S, achievable accuracy)` that turns "we can't hit lossless" from a surprise at finalize into an explicit, up-front engineering decision (raise the budget, change the code, or accept the floor).

**This pre-analysis is the single most important addition.** It replaces the team's "park the S ceiling as orthogonal" with "quantify the ceiling, allocate against it, and only then start training." It also tells the FAST plan *where* to spend fine-tuning effort (the few high-sensitivity layers) and where calibration alone suffices.

### 1.3 Beyond uniform single-spike TTFS (raising the ceiling itself)

If the budget analysis says the ceiling is too low, the lever is the *code*, not the training. Options, in increasing hardware cost — each gated by what `nevresim`/the target chip actually supports:

- **Per-layer `S`** (from §1.2) — free if the chip allows heterogeneous time windows.
- **Multi-spike / burst codes** — allow `k>1` spikes per neuron per window; capacity rises to `log2(C(S,k))`. Often a firmware/mapping change, not silicon.
- **Graded / weighted spikes** (if the chip carries a payload per spike) — dramatically higher capacity per event.
- **Population coding** — represent one value across `m` neurons; trades neuron budget for precision.

The plan does not assume any of these is available; it requires the team to **state the chip's code capabilities as a capacity contract** (an extension of the `AdaptationAxis` contract, §6) so the budget analysis is grounded in real hardware, not wishful precision.

---

## 2. Why LIF Is Lossless and TTFS Cliffs — and What That Dictates

The diagnosis is not subtle once framed by conversion theory.

**Rate-coded LIF** is a robust linear code. ANN→SNN rate conversion is a mature, near-lossless procedure: replace ReLU with an integrate-and-fire neuron, set firing thresholds by **threshold balancing / weight normalization** (so the max firing rate matches the max pre-activation), use a **soft reset (reset-by-subtraction / residual membrane potential)** to avoid information loss at reset, and give it enough `T`. The error is `O(1/T)` and calibratable layer by layer. This is why LIF is bit-exact here without heroics.

**Cascaded TTFS** is a fragile nonlinear temporal code with three compounding problems:
1. **Low, non-uniform precision** (`log2(S+1)` bits, coarser near the timing extremes) — the capacity ceiling of §1.
2. **Cross-layer timing dependence** — a spike's time in layer `ℓ` depends on when its inputs arrived, so per-layer conversion errors *compound* down the cascade (the team's "death cascade").
3. **Acute scale/threshold sensitivity** — the value→time map depends on each layer's dynamic range; mis-calibration destroys it (exactly the encoding-layer bug, which was a single-layer instance of a *general* requirement).

**What this dictates** (and what the previous work only partially did):
- TTFS needs **explicit per-layer value→time calibration** (threshold/scale balancing adapted to timing codes), not just activation-distribution matching at the boundaries. The encoding-layer fix must generalize into a **calibration contract for every layer** (§4, L3).
- Error compounding is best fought **layerwise** — convert and *reconstruct* each layer to match the ANN's output for that layer (§4, L2), so errors are corrected locally before they cascade, rather than hoping end-to-end fine-tuning untangles a compounded mess.
- The discrete forward must be **in the training graph from the start of the genuine phase** with a **well-behaved annealed surrogate gradient** (the team's surrogate-alpha annealing is on the right track and should be grounded in the SNN surrogate-gradient literature, e.g. SpikingJelly's surrogate functions, which are already a submodule here).

The big strategic implication: **the cliff is a conversion-fidelity failure that calibration + layerwise reconstruction should largely eliminate *before* any expensive gradual fine-tuning** — and what remains after good conversion is a small residual that a short, well-conditioned anneal closes. That inverts the current emphasis (heavy controller-driven fine-tuning, light calibration).

---

## 3. Solution Architecture: One Adaptive Pipeline, Cheapest-Viable-Tool-First

The current system has two engines (the controller, and the experimental fast loop) and treats gradual deformation as the default tool. Replace that with **one pipeline that always runs the cheapest tool that achieves lossless-within-tolerance, escalating only when needed.** Conceptually a ladder:

```
  ┌─ Stage 0: CAPACITY BUDGET (§1) ── achievable floor + per-layer S/bit allocation ──┐
  │                                                                                    │
  ├─ Stage 1: CONVERT + CALIBRATE (§4 L3) ── threshold/scale balancing, soft-reset ────┤  no training
  │                                                                                    │
  ├─ Stage 2: LAYERWISE RECONSTRUCTION (§4 L2) ── per-layer match to ANN (AdaRound/    │  seconds–min,
  │            LSQ/BRECQ-style, + feature distill) ────────────────────────────────────┤  parallel, cheap
  │                                                                                    │
  ├─ Stage 3: SHORT ANNEALED GENUINE FINE-TUNE (§5 F1) ── one optimizer, scheduled ─────┤  minutes,
  │            rate/surrogate, deployed-forward loss + full distillation ───────────────┤  no cycles
  │                                                                                    │
  └─ Stage 4: CONTROLLER FALLBACK (existing) ── only for the residual on ill- ──────────┘  the expensive
             conditioned layers/axes that Stages 1–3 left short of tolerance              path, rarely
```

The decision rule after each stage: **evaluate the deployed forward on the full test set; if within `ε_lossless`, stop.** Most layers/axes should clear at Stage 1–2 (calibration + reconstruction is near-lossless for robust codes and well-conditioned quant), reach the rest at Stage 3 (a short anneal), and invoke Stage 4 only for genuinely hard residuals (e.g., the most timing-sensitive TTFS layers at low `S`). This is the resolution of the "two engines" fork from the last review: **the controller becomes the rarely-used top of a ladder, not the default.**

The remainder of the plan details the LOSSLESS levers (Stages 1–3 fidelity) and the FAST levers (Stages 1–3 cost), then folds them into the orchestrator (§6).

---

## 4. The LOSSLESS Program

Eight tracks, ordered by leverage. L1–L4 are the conversion-fidelity core; L5–L6 are allocation; L7 is the last mile; L8 is the upstream co-design that makes the whole thing easier.

### L1 — A faithful deployed forward in the training graph, with an annealed surrogate

Lossless is impossible if you train one forward and deploy another. The genuine cascade (the exact `nevresim`-equivalent forward) must *be* the training forward for the genuine phase, with:
- **Exact forward** (the team's `_SegmentSpikeForward` / `TTFSSegmentForward` is the right object — it must remain bit-identical to the deployed sim, enforced by the existing torch↔sim parity gate).
- **Annealed surrogate backward.** Use a smooth spike surrogate (sigmoid'/arctan'/triangular) whose sharpness anneals smooth→hard over the schedule (the team's `surrogate-alpha` annealing). Ground the surrogate choice and schedule in SpikingJelly's surrogate library rather than a bespoke one — it is already a submodule and is battle-tested.
- **Straight-through where appropriate** for the hard quantizers, with the surrogate reserved for the spiking nonlinearity.

This makes "cliff 0 by construction" *real*: at rate 1 the forward is exactly the deployed cascade and there is nothing left to drop. The `BlendedGenuineForward` already gives this at the output level; L2–L4 make the *interior* faithful too.

### L2 — Layerwise reconstruction against the teacher (the highest-leverage lossless lever)

This is the biggest under-exploited idea. Instead of (or before) end-to-end fine-tuning, **convert and reconstruct one layer/block at a time to minimize the deployed layer's output error against the ANN's activation on calibration data.** This is the spiking adaptation of the PTQ-reconstruction family:

- **AdaRound-style learned rounding** for the weight-quant axis: learn per-weight up/down rounding to minimize the layer output MSE vs fp — closes most quantization loss with a few hundred calibration batches and *no* end-to-end training.
- **LSQ-style learned step size** for activation/weight quant: make the quantization step a trainable parameter (the team already has "tunable_parameters" in the axis contract for exactly this — wire it).
- **BRECQ-style block reconstruction** for the spiking blocks: optimize each block's *genuine* output (the timing-coded cascade for that block) to match the ANN block's output, on cached ANN inputs. Because each block is corrected against the *true* ANN target, **errors do not compound** — directly defusing the death cascade.
- **QDrop-style stochastic quant during reconstruction** to avoid overfitting the calibration set.

Why this is both lossless *and* fast: layerwise reconstruction is local (small optimization per block), parallelizable across blocks, runs on cached activations (no full forward/backward through the whole cascade), and is the empirically near-lossless workhorse of modern PTQ. **For the quant axes it may eliminate fine-tuning entirely; for the spiking axes it gives the genuine fine-tune a near-converged starting point.**

### L3 — A per-layer calibration contract (generalize the encoding-layer fix)

The encoding-layer parity bug taught the general lesson: **every layer has a correct value↔spike scale dictated by a contract, and conversion must respect it.** Formalize:
- **Threshold/scale balancing** per layer for the spiking conversion (the rate-SNN threshold-balancing procedure; the TTFS analogue calibrates the value→time `x_max` per layer). This is the principled version of "distribution matching."
- **Soft reset / residual-potential** carry where the code supports it (kills reset-induced loss; standard in near-lossless rate conversion).
- **Pinned-contract layers** (encoding/input, and any layer whose scale is fixed by a hardware contract) are *excluded* from free retuning — the encoding-layer fix becomes a declared property (`is_scale_pinned`), not a special case.
- **DFQ cross-layer equalization + bias correction** (the team's DFQ) as the data-free pre-step before any data-driven calibration.

Calibration is *cheap* (forward-only on a calibration set) and is where most of the rate-SNN losslessness comes from — invest here before fine-tuning.

### L4 — Full distillation from the teacher (output **and** feature)

The ANN is the lossless oracle. Today only its *output* is distilled (KD + genuine-CE). Add:
- **Feature-map / intermediate distillation** (FitNets-style): match the SNN's per-block representation (decoded rate/timing) to the ANN's per-block activation. This re-anchors the interior and, like L2, arrests compounding.
- **Logit KD with temperature** against the ANN (already present) — keep.
- **Optionally attention/relational transfer** for transformer-style mixers (the repo has `mlp_mixer`/ViT cores).

The training objective for the genuine phase becomes "match the ANN" at multiple depths, not just "classify correctly." Since lossless *means* "SNN ≈ ANN," minimizing the SNN↔ANN divergence directly is the correct objective — task CE alone under-specifies it.

### L5 — Per-layer / per-axis sensitivity ordering (resurrect per-layer rate)

The team deleted per-layer rate as a "zombie flag." It was unused, not wrong — and a sophisticated approach *needs* it:
- Drive the ramp **per layer**, ordered by the §1.2 sensitivity: transform robust layers fully first (cheap, lossless), defer the sensitive ones and give them more budget/fine-tuning. This is the temporal-coding analogue of mixed-precision ordering.
- Where axes interact (activation-quant × TTFS-timing are coupled; weight-quant × threshold balancing), transform them **jointly per layer** rather than as separate global 0→1 passes — the current fully-sequential axis schedule ignores these couplings.

Reintroduce per-layer rate as a first-class scheduler capability (it is a vector rate; the `RateScheduler` already has the shape), driven by the sensitivity profile rather than a uniform scalar.

### L6 — Temporal-resolution allocation (the capacity lever, operationalized)

Operationalize §1.2/§1.3: the allocator outputs a per-layer `S_ℓ` (and code choice), the conversion respects it, and the budget knapsack is re-solved if a layer proves harder than its sensitivity sweep predicted. This is where you *buy* losslessness with latency/energy in a principled, Pareto-explicit way instead of discovering the ceiling at finalize.

### L7 — The last mile (rate 0.97 → 1.0)

Residual loss hides in the final stretch where the teacher is fully removed and the pure genuine cascade stands alone. Treat it explicitly:
- **Extended genuine-only fine-tune at rate = 1** with the ANN logits as the distillation target (the teacher is the lossless reference even after the blend is gone).
- **Recalibrate on genuine statistics** at rate 1 (BatchNorm/scale recalibration on the deployed forward's activation statistics — the distribution shifts when the teacher branch is dropped).
- **Reconstruction touch-up** (L2) on the most sensitive blocks at rate 1.
This is the difference between "≈0.94" and "within-noise-of-the-ANN," and it is cheap relative to the full ramp.

### L8 — Co-design / SNN-readiness preconditioning (upstream, highest ceiling)

The deepest reframe: a continuation method only reaches a lossless endpoint if a lossless endpoint exists *near the start*. If the pretrained ANN sits in a basin with no nearby SNN-friendly minimum, no gradual path recovers losslessly. So **shape the source**:
- **Constraint-aware pretraining / QAT-from-scratch:** pretrain the ANN with quantization (and ideally a spiking-surrogate penalty) already in the loss, so it lands in an SNN-friendly basin. This is the single biggest lever on the *ceiling* of losslessness.
- **Activation-range regularization:** penalize activation distributions that are hostile to the spiking code (heavy tails, large dynamic range) so the value→spike conversion is well-conditioned by construction.
- **Bounded/clamped activations from the start** (the clamp axis, but applied during pretraining) so the deployed clamp is not a deformation but the native behavior.

L8 is a longer-horizon investment and changes the upstream training recipe, but it is what turns "gradual repair of a hostile model" into "light conversion of a friendly one." It should be prototyped in parallel (§8) because it can dominate everything downstream.


---

## 5. The FAST Program

Seven levers, ordered by impact. The unifying idea: **the cheapest cycle is the one you never run** — eliminate search, eliminate end-to-end fine-tuning where reconstruction suffices, and eliminate redundant evaluation.

### F1 — Schedule, don't search (collapse the discrete cycle loop into one annealed run)

This is the FAST headline, and the team's own experimental fast path already proved it (genuine 0.41→0.9355 with a fixed schedule, one optimizer, no controller). Make it the **default policy for well-conditioned axes**:
- Drive the rate (and the surrogate sharpness) as **scheduled hyperparameters** over a single fine-tuning run — a curriculum/annealing, exactly like temperature annealing in Gumbel-softmax or bit-width annealing in QAT — with one optimizer and a cosine/one-cycle LR.
- **No probes, no bisection, no rollback, no per-cycle eval, no LR re-find.** Evaluate only at a few checkpoints to confirm progress.
- Validity condition: the homotopy is well-conditioned (monotone, cliff-free), which the **characterization phase already verifies** (the spec's §10, now wired). If characterization flags non-monotonicity/cliff on some layer, *that layer* escalates to the controller (Stage 4); the rest anneal.

This single change removes the dominant costs (the LR-finder tax and the serial cycle overhead) for the common case and is the proper home for the "fast hack" — folded into the orchestrator as a policy, not a bespoke loop (resolving the fork from the last review).

### F2 — Reconstruct, don't fine-tune (for the quant axes)

End-to-end fine-tuning is the heavy hammer. For weight/activation quantization, **PTQ reconstruction (L2: AdaRound/LSQ/BRECQ/QDrop) reaches near-QAT accuracy in seconds-to-minutes per block, on cached activations, with no end-to-end backward.** Reserve gradient fine-tuning for the spiking dynamics only. This can cut the quant-axis cost by 1–2 orders of magnitude while *improving* fidelity (reconstruction targets the true layer output).

### F3 — Kill the LR-finder tax (it is the measured dominant cost)

Profiling showed LR-finding (8 probes × 30 steps through the genuine forward, re-run on every target relaxation) dominates. Beyond F1 (which removes most re-finds):
- **One-time LR via a cheap proxy** (loss-slope on a few dozen batches, the spec's §7.1, or a closed-form estimate from the gradient/curvature), not an 8×30 sweep.
- **Schedule-free or trust-region optimizers** (e.g. schedule-free Adam, or a Lion/Adafactor variant) that are far less LR-sensitive, removing the *need* for a sweep.
- **Reuse one LR across the whole anneal** (already validated for the blend ramp; generalize to all well-conditioned axes).

### F4 — Evaluation economy (anchored to the deployed forward)

Evaluation is the other big cost, and the parity work showed it must be done on the **deployed forward, full test set** to be honest. Reconcile "honest" with "fast":
- **Decision evals** use a fixed, seed-stratified **subsample** of the test set (paired, McNemar SE — the spec's §6.2), reserving the **full-set deployed eval** for the final lossless certification and periodic checkpoints. (The team's `deployment_metric_full_eval` is the certification; the search should use the cheap subsample, *also on the deployed forward* so it is not fooled — the §A lesson.)
- **Cache** the deployed forward's intermediate activations once per checkpoint for reuse by reconstruction (L2) and feature distillation (L4).
- **Vectorize / batch** the spiking sim eval; **distribute** across the (acknowledged contended) GPUs.

### F5 — Checkpoint / recovery only on demand

The CheckpointGuard's scoped/pinned snapshots (from the D3 work) matter only when rollback happens. In the annealed default (F1), rollback essentially never fires, so the snapshot cost largely vanishes. Keep the guard for the controller fallback; do not pay it on the fast path. Scope snapshots to `tunable + transform params` (the frozen ANN backbone need not be cloned during fine-tune).

### F6 — Amortize across axes and layers (parallel + warm-start)

- **Parallel layerwise reconstruction** (L2) across blocks — independent optimizations on cached activations.
- **Warm-start** each axis/stage's optimizer from the previous stage's state (the persistent-optimizer work, scoped where param-sets are stable) so the anneal does not cold-start.
- **Fuse coupled axes** (L5) so weight-quant + threshold-balancing + TTFS-timing for a layer are calibrated together in one pass rather than three sequential 0→1 ramps.

### F7 — Scale to large models / datasets (so FAST holds at size)

- Reconstruction and calibration are **forward-mostly and layer-local**, so they scale far better than end-to-end fine-tuning — favor them at size.
- **Cache the calibration subset on device** (sized, seed-stratified), never the full set (the W8 lesson).
- For very large models, **FSDP/sharded** fine-tune with the sensor's correctness counts all-reduced so decisions are rank-invariant (the spec's distributed note); reconstruction is naturally shardable per block.

**Net FAST picture:** for a well-conditioned transformation, the process becomes *calibrate (forward-only) → layerwise-reconstruct (parallel, cached) → one short annealed genuine fine-tune (one optimizer, one LR, few evals)* — minutes, not the current many-minutes-to-hours of cycle/rollback/LR-find. The controller is invoked only for the residual hard layers.

---

## 6. Unify Into the One-Orchestrator Design (Transformation Contract v2)

Fold all of the above into the existing four-service architecture by **enriching the `AdaptationAxis`/transformation contract** so the driver can pick the cheapest viable policy per transformation. The contract gains:

```text
Transformation/Axis (v2) additionally declares:
  capacity_profile()        # bits/robustness of its code at given S/bit (§1) — for the budget
  calibrate(model, data)    # threshold/scale balancing, soft-reset, DFQ (§4 L3) — forward-only
  reconstruct(block, ref)   # layerwise PTQ-style reconstruction vs the ANN target (§4 L2)
  tunable_parameters()      # LSQ step sizes / learnable thresholds (wire the existing hook)
  surrogate()               # the annealed surrogate-gradient spec (§4 L1)
  conditioning -> {well_conditioned | ill_conditioned | unknown}   # from characterization (§5 F1)
  is_scale_pinned           # encoding/contract layers excluded from retuning (§4 L3)
  feature_targets()         # intermediate tensors to distill against (§4 L4)
```

The driver's **policy selector** (replacing the two-engines fork) reads these and routes:
- `well_conditioned` → **annealing policy** (F1): calibrate → reconstruct → short annealed fine-tune, no controller.
- `ill_conditioned` / `unknown` → **controller policy** (the existing scheduler + rollback + recovery), but only after calibration + reconstruction have done their cheap work.
- Every policy shares the **deployed-anchored sensor** (the genuine probe on the deployed forward) and the **capacity budget** (§1).

This keeps the spec's "one orchestrator, N transformations" principle intact while absorbing the genuine-TTFS recipe, the fast path, and the conversion/reconstruction toolkit as *first-class transformation behaviors* rather than bolted-on flags. The recovery-quality flag cluster collapses into the two named policies.


---

## 7. Metrics, Targets, and Validation

**Define "lossless" precisely, anchored to deployment.** Lossless = the **deployed forward** (the `nevresim`-equivalent spiking cascade, not a torch proxy) on the **full test set** is within the ANN's evaluation noise band:

`acc_deployed ≥ acc_ANN_full − k·SE`, with the paired SE from §6.2 and `k≈2`.

This is the spec's fixed-baseline anti-drift, corrected by the parity lesson: the baseline is the ANN on the full set, and the candidate is the *deployed* forward, so the two numbers are commensurable (the failure that manufactured the spurious "SCM drop").

**Primary metrics (track per run, per layer):**
- **Lossless gap:** `acc_ANN_full − acc_deployed_full`. Target: within `±k·SE` (i.e., ≤ ~0.3–0.5 pp on these models).
- **Finalize cliff:** `acc_ramp@1 − acc_deployed`. Target: ≈0 (the L1 "faithful forward" guarantee should make this structural).
- **Per-layer residual attribution:** the layerwise reconstruction error (L2) per block, so residual loss is *attributed*, not aggregate — you know which layer to spend more `S`/fine-tune on.
- **Capacity headroom:** achieved accuracy vs the §1 Pareto ceiling at the chosen `Σ S`. If you are at the ceiling, lossless requires more budget, not more training — a crucial diagnostic.

**Speed metrics:**
- **Wall-clock to certified-lossless**, decomposed into calibrate / reconstruct / fine-tune / eval / search (extend the team's existing `_phase_seconds`).
- **Eval passes** and **LR sweeps** per run (both should approach the F-program's near-zero for well-conditioned axes).
- **Peak VRAM** and **checkpoint bytes** (should drop sharply once fine-tuning is replaced by reconstruction).

**Validation infrastructure (build on what exists):**
- **The torch↔deployed-sim parity gate stays the gatekeeper** — every certification asserts bit-exactness (modulo documented WQ tie-flips) so "deployed accuracy" is trustworthy.
- **The mock-axis zoo** (the spec's IV.2) gets new archetypes for the policy selector: a `well_conditioned` axis must take the annealing path and never invoke the controller; an `ill_conditioned` axis must escalate; a `capacity_starved` axis must report the ceiling rather than spin.
- **Per-layer reconstruction unit tests:** reconstruction of a block must match the ANN block output to a documented tolerance on calibration data.
- **A regression gate on the cliff:** the cascaded-TTFS deployed gap must stay below the certified tolerance once Stage 1–3 land — the cliff must not silently return.
- **Capacity-budget golden:** the §1 sensitivity sweep + allocation is deterministic and snapshot-tested, so a model's achievable floor is reproducible.

---

## 8. Experiment Matrix (Ranked, with Hypotheses and Kill Criteria)

Run these roughly in order; each has a falsifiable hypothesis, the metric that decides it, and a kill criterion so dead ends die fast.

| # | Experiment | Hypothesis | Decide by | Kill criterion |
|---|---|---|---|---|
| E1 | **Capacity budget on cascaded TTFS** (§1) | The ~0.26 cliff is *partly* a capacity wall; per-layer sensitivity is highly non-uniform | Pareto curve `(ΣS, acc)`; per-layer `Δacc_ℓ(S)` | If the ceiling at the chip's `ΣS` is already ≥ lossless, capacity is *not* the issue → focus on L2/L3 only |
| E2 | **Calibration-only conversion** (L3): threshold/scale balancing + soft-reset + DFQ, **no fine-tune** | Most of the LIF↔TTFS gap is mis-calibration; calibration alone closes a large fraction | deployed gap after calibration vs the ~0.26 cliff | If calibration moves the gap <20% of the way, the value→time map needs reconstruction (E3), not just balancing |
| E3 | **Layerwise reconstruction** (L2) on the spiking blocks, on cached ANN activations | Block reconstruction against the ANN defuses the death cascade; near-lossless at PTQ cost | deployed gap + per-layer residual; wall-clock | If per-layer residuals don't compose (end-to-end gap ≫ Σ per-layer), cross-layer timing coupling needs joint treatment (E6) |
| E4 | **Annealed genuine fine-tune as the *only* training** (F1), single optimizer/LR, vs the controller | The controller adds cost but not accuracy on well-conditioned TTFS | deployed acc (anneal vs controller) at equal budget; wall-clock | If anneal is materially worse at any budget, the homotopy is ill-conditioned somewhere → keep controller for those layers |
| E5 | **PTQ reconstruction vs QAT fine-tune for the quant axes** (F2) | AdaRound/LSQ/BRECQ match fine-tune accuracy at a fraction of the cost | quant-axis deployed acc + wall-clock | If reconstruction underperforms by >0.5 pp, keep a short fine-tune touch-up (still cheaper than full QAT) |
| E6 | **Per-layer / joint-axis ordering** (L5): sensitivity-ordered, couple WQ×threshold×TTFS per layer | Joint per-layer beats sequential global passes on both fidelity and cost | deployed acc + total wall-clock vs sequential | If no gain over sequential, retain sequential (simpler) |
| E7 | **Full feature distillation** (L4) vs output-only KD | Re-anchoring interior features closes residual cascade error | deployed gap with vs without feature distill | If <0.2 pp gain, drop it (cost not worth it) |
| E8 | **Last-mile treatment** (L7): extended rate-1 genuine fine-tune + recalibration + KD-to-ANN | The final pp hides in rate→1; explicit last-mile closes it | deployed gap before/after last mile | If gap already within tolerance after E2–E4, skip |
| E9 | **Richer temporal code** (L6/§1.3): per-layer `S`, then multi-spike if the chip allows | Raising capacity on the few starved layers crosses the ceiling losslessly | deployed acc at the chip's feasible code set | Gated by hardware support; if unsupported, E1's ceiling stands |
| E10 | **Co-design preconditioning** (L8): QAT-from-scratch + activation-range reg in pretraining | A spiking-friendly basin makes Stages 1–3 near-trivial and raises the ceiling | deployed acc + total (pretrain+convert) cost vs convert-a-hostile-ANN | If preconditioned model isn't easier to convert, the basin hypothesis is wrong for these archs |

**Strategic reading of the matrix:** E1–E3 test the thesis that *calibration + reconstruction does most of the lossless work cheaply*. E4–E5 test that *schedule/reconstruction replaces the expensive controller/fine-tune*. E6–E8 are fidelity refinements. E9–E10 are the ceiling-raisers (hardware code, and upstream co-design) for whatever residual remains. If E1–E5 land as hypothesized, you have **fast-and-lossless for the common case** and a clear, attributed account of any residual.

---

## 9. Sequencing, Milestones, and Risks

**Phase 1 — Quantify and calibrate (weeks, low risk, high information).** Land the §1 capacity budget (E1) and the calibration contract (E2, L3). Deliverable: a Pareto ceiling per target model + the deployed gap after calibration-only. This alone tells you how much of the cliff is capacity vs calibration vs trainability, and it is mostly forward-only code reusing the existing scale-aware-boundary and DFQ machinery.

**Phase 2 — Reconstruct (weeks, medium risk, high leverage).** Land layerwise reconstruction for the quant axes (E5, F2) and the spiking blocks (E3, L2), with per-layer residual attribution. Deliverable: near-lossless quant axes at PTQ cost, and a near-converged starting point for the spiking fine-tune. This is where most of the FAST win and a large part of the LOSSLESS win land together.

**Phase 3 — Anneal and unify (weeks, medium risk).** Make the annealed single-run fine-tune the default for well-conditioned axes (E4, F1), fold the fast path and the controller into the policy selector (§6), and add the last-mile treatment (E8, L7). Deliverable: the one-orchestrator pipeline with the controller demoted to fallback; the recovery-quality flag cluster collapsed into two policies.

**Phase 4 — Ceiling-raisers (parallel, higher risk/reward).** Prototype richer temporal codes (E9, gated by hardware) and co-design preconditioning (E10) in parallel from the start, since either can dominate downstream effort and raise the achievable ceiling.

**Risks and mitigations:**
- **Reconstruction may not compose across the TTFS cascade** (timing coupling). *Mitigation:* E3's per-layer-residual composition check; escalate coupled layers to joint reconstruction (E6) or the controller fallback.
- **The annealed path may be ill-conditioned on some layers** despite good conversion. *Mitigation:* the characterization gate routes those layers to the controller; the architecture supports per-layer policy.
- **Hardware code capabilities may cap the ceiling** below lossless. *Mitigation:* E1 makes this explicit up front; the decision (more `ΣS`, richer code, or accept the floor) is a product call, not a silent finalize-time loss.
- **Co-design changes the upstream training recipe** (organizational cost). *Mitigation:* prototype on one model in parallel; adopt only if E10 shows a decisive convert-cost/ceiling win.
- **Regression of the cliff** as the default changes. *Mitigation:* the deployed-gap regression gate (§7) and the torch↔sim parity gate guard every change.

---

## 10. Summary — The Direction in Five Moves

1. **Lead with capacity.** Compute a per-layer rate-distortion budget and a Pareto ceiling *before* training (§1). Lossless below capacity is impossible; quantify it, allocate non-uniform `S`, and stop fighting walls. (The team's "S=4 ceiling" is the binding constraint, not an aside.)
2. **Win losslessness with conversion fidelity, not the controller.** Per-layer threshold/scale calibration + soft-reset (L3) and **layerwise reconstruction against the ANN teacher** (L2) — the established near-lossless ANN→SNN/PTQ toolkit — do most of the work cheaply and defuse the death cascade. The cliff is a conversion failure, and LIF's bit-exactness proves conversion *can* be lossless.
3. **Win speed by replacing search with schedule and fine-tuning with reconstruction.** A single annealed genuine fine-tune (one optimizer, one LR, few evals) for well-conditioned axes (F1) — the team's own fast path, generalized — plus PTQ reconstruction for the quant axes (F2), eliminates the dominant LR-finder and cycle costs.
4. **Exploit the teacher as a layerwise oracle** (feature distillation L4, reconstruction targets L2) and treat the **last mile** explicitly (L7) — that is where "≈0.94" becomes "within noise of the ANN."
5. **Unify into one adaptive pipeline** (§6): capacity budget → calibrate → reconstruct → short anneal → controller *only* for the ill-conditioned residual, all anchored to the **deployed forward on the full test set**. Consider **co-design preconditioning** (L8) as the upstream lever that raises the ceiling and makes the whole pipeline near-trivial.

The previous engineer made the gradual-deformation paradigm *work*; the expert move is to recognize that for most of the model the paradigm is unnecessary (calibration + reconstruction + annealing are faster and at least as lossless), that the residual is bounded by a capacity ceiling that must be quantified and allocated against, and that the controller is the rarely-needed top of a ladder — not the default road.
