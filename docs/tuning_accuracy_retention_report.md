# Tuning Accuracy-Retention — Analysis Report

**Subject:** the training-aware tuning subsystem (`src/mimarsinan/tuning/`) — how
it retains accuracy while transforming a continuous ANN into a deployable SNN,
where accuracy leaks, every mechanism that has been tried to close those leaks,
and the cost of each.

**Why now:** this conversation's mapping/fidelity work established that the
deployed sim is *bit-exact* to the torch forward for the value-deterministic
modes (see the deployment-fidelity work). That removes "the simulator is lying"
as an explanation for deployed-accuracy loss and points the remaining loss
squarely at **tuning**: the transformation the fine-tuner applies, and how much
of the proxy accuracy survives the genuine on-chip dynamics.

This report inventories every relevant code path and the architectural mechanism
behind it.

---

## 1. Executive summary

- **Tuning is a homotopy walk.** Every transformation (ANN→SNN activation,
  weight quantization, clamping, pruning, …) is applied along a *rate axis*
  `α ∈ [0,1]`: `α=0` is the original behaviour, `α=1` the full transform. A
  single control loop walks `α` toward 1 while recovering accuracy, rolling back
  cycles that regress beyond a statistically-derived tolerance.
- **For genuine cascaded TTFS, the ramp trains a *proxy*.** The KD-blend ramp
  advances each perceptron's value-domain `BlendActivation` (rate 0 == continuous
  teacher, rate 1 == the pointwise on-chip composition). The **genuine
  cross-layer single-spike cascade** is installed only at `_finalize()`. The gap
  between the proxy at rate 1 and the genuine deployed forward is the **finalize
  cliff** (~0.26 for cascaded TTFS; ~0.04–0.12 for LIF), historically recovered
  post-hoc by stabilization rather than by the ramp.
- **The diagnostic was fooled — now fixed (opt-in).** The full-transform probe
  used to measure the *value-domain* rate-1.0 accuracy (which converges nicely),
  hiding the genuine cliff. The landed `tuning_full_transform_probe` now measures
  *both* and keys the verdict on the **genuine** drop, surfacing `proxy_gap =
  value − genuine` and warning "the value-domain probe WAS FOOLED" when the proxy
  converges but the genuine forward does not.
- **The genuine ramp now exists (opt-in) and works on the bench.** Three
  approaches train *through* the genuine cascade for the whole ramp so the cliff
  is ~0 by construction: a **surrogate-annealed** ramp, a **teacher→cascade blend**
  ramp with distribution matching, and an **experimental fast** path that
  reproduces `generated/_genuine_ab/full_ramp.py` (genuine **0.41 → 0.9355**).
  They remain default-off pending a full-run non-regression gate vs the proxy
  baseline.
- **Three independent loss sources** cap retention: the finalize cliff (biggest,
  cascaded), ~2 pp ramp under-training, and a ~2.7 pp S=4 representational ceiling.

Baselines: LIF deployed **0.958** (bit-exact parity), cascaded TTFS **0.928**,
value proxy saturates ~**0.963** at S=4; higher S deployed 0.924/0.938/0.940
(S=4/8/16) — lifts hardware fidelity, not the tuning ceiling.

---

## 2. The problem: what "accuracy retention" means here

A deployed model loses ground relative to its pretrained/proxy accuracy at three
**independent** points:

| Loss source | Size | Mechanism |
|---|---|---|
| **Finalize cliff** (proxy↔genuine) | ~0.26 cascaded TTFS; ~0.04–0.12 LIF | the ramp trains the value-domain `BlendActivation` proxy; the genuine single-spike cross-layer cascade (`_SegmentSpikeForward`) is installed only at `_finalize`. The value composition ≠ the genuine timing-coded cascade. |
| **Ramp under-training** | ~2 pp | the controller exhausts its step budget before fully recovering accuracy as `α→1` (proxy 0.956 achieved vs 0.963 ceiling). |
| **S=4 representational ceiling** | ~2.7 pp | the S-step timing code caps the proxy itself (~0.963). Orthogonal to tuning; needs higher S or a better code. |

The first two are tuning's to fix; the third is representational. The mapping
work this session confirmed the deployed sim adds *no* further loss for LIF /
cascaded TTFS / ttfs_quantized (bit-exact), so the cliff is genuinely a
*training-target* mismatch, not a simulator artifact.

**The NF↔SCM "drop" was a metric artifact.** Earlier it looked like deployment
lost ~0.019; that was a full-10k NF metric vs a 500-sample-subsample SCM metric
(< 1 SE; `accuracy_se = 0.5/√n ≈ 0.022` at n=500). `deployment_metric_full_eval`
(default True) evaluates the deployment metric on the full test set, after which
SCM == NF exactly. Lesson baked into the tooling: the genuine cliff must be read
from a *value-vs-genuine* comparison on a *shared* eval, not from a subsample
accuracy delta.

---

## 3. Architecture of the tuning subsystem

The control core lives in `src/mimarsinan/tuning/` (root SSOT primitives +
`orchestration/` loop & services + `tuners/` concretes + `axes/` rate adapters).
Each `ARCHITECTURE.md` in those directories is the authoritative component map;
this section is the mechanism-level walk-through.

### 3.1 The tuner hierarchy and the dissolved control loop

```
TunerBase                                  (orchestration/tuner_base.py)
└── SmoothAdaptationTuner                   (= Cycle + Run + Base mixins)
    ├── AdaptationRateTuner                 (adaptation_rate_tuner.py — manager rates)
    │   ├── ActivationQuantizationTuner
    │   └── NoiseTuner
    ├── ClampTuner / ActivationAdaptationTuner / PruningTuner / PerceptronTransformTuner
    └── KDBlendAdaptationTuner              (orchestration/kd_blend_adaptation_tuner.py)
        ├── LIFAdaptationTuner              (tuners/lif_adaptation_tuner.py)
        └── TTFSCycleAdaptationTuner        (tuners/ttfs_cycle_adaptation_tuner.py)
```

- `orchestration/smooth_adaptation_run.py:SmoothAdaptationRunMixin.run()` —
  baseline calibration (`AcceptanceSensor.calibrate_baseline`) → `_run_with_scheduler`
  (the single `RateScheduler` loop) → `_finalize_run` (`_after_run` →
  `_stabilize_at_full_rate`).
- `orchestration/smooth_adaptation_cycle.py:SmoothAdaptationCycleMixin._adaptation`
  is now a **1-line delegate** to `AdaptationDriver.run_cycle`. The old
  `_adaptation` god-method was dissolved into named host **phase methods**:
  `_begin_cycle` (clone pre-state + pre-cycle eval), `_probe_instant`
  (`_update_and_evaluate` + catastrophic fast-fail), `_recover` (LR + recovery
  train), `_measure_post` (post eval + dual rollback gate), `_rollback_cycle` /
  `_commit_cycle`. One `DecisionRecord` is logged per cycle exit into a
  `DecisionTrace` (`tuning/trace.py`) — the JSON golden-trace contract that gates
  every refactor (`tests/unit/tuning/golden/`, `MIMARSINAN_RECORD_GOLDEN`).

### 3.2 The orchestrator + services (driver + N services)

`orchestration/adaptation_driver.py:AdaptationDriver` is the predictor→corrector→
commit/rollback skeleton. `run()` drives the scheduler from `committed` toward 1.0
over a per-cycle `attempt`, then `finalize`; `run_cycle(host, rate)` sequences the
host's phase methods and branches on `CycleContext` flags (catastrophic /
rolled_back). The decisions, snapshots and recovery are factored into services:

| Service | File | Responsibility |
|---|---|---|
| `RateScheduler` | `orchestration/rate_scheduler.py` | the rate-search *policy*: greedy-to-1.0 then bisect-the-gap (`greedy_to_one`), `uniform_ladder` (KD-blend's 0.125 steps), `one_shot_only`, `dense_grid` (non-monotone safe mode). The first jump always runs; `epsilon` bounds only the bisection. |
| `AcceptanceSensor` | `orchestration/acceptance_sensor.py` | the accept/reject/recovered *decision* math (see §3.3). Pure, bit-exact extraction. |
| `RecoveryEngine` | `orchestration/recovery_engine.py` | the *corrector*: centralizes `train_steps_until_target` + recovery-hook remove-in-`finally`; threads an optional owned optimizer (`tuning_persist_optimizer` keeps Adam moments across recovery calls). |
| `CheckpointGuard` | `orchestration/checkpoint_guard.py` | scoped rollback snapshots. `scope=full/location=device` is byte-identical to the old clone/restore; `scope=tunable` clones only `requires_grad` params; `location=cpu_pinned` is a real async d2h offload (frees ~1× VRAM, ~3–8× faster — `test_checkpoint_guard_cuda_benchmark`). |
| `characterize` | `orchestration/characterization.py` | optional pre-search α-grid profiling (`tuning_enable_characterization`): a non-monotone drop beyond the noise budget forces `dense_grid` safe mode; the steepest slope sets the bisection `epsilon_hint` so it never steps over a cliff. |

Building the driver fully standalone from the services is the remaining "V6
polish"; today the mixins host the phases and the driver sequences them.

### 3.3 The statistical decision model (`AcceptanceSensor` + `TuningBudget`)

Every threshold derives from one quantity:
`tuning_budget.py:TuningBudget.accuracy_se() = 0.5/√(eval_sample_count)` — the
Bernoulli worst-case standard error of a validation pass. From it:

- `_rollback_tolerance = 3·accuracy_se` — the per-cycle **noise** gate
  (`post_acc ≥ pre_cycle_acc − tol`).
- `absolute_floor = max(_validation_baseline·(1−_pipeline_tolerance),
  _pipeline_hard_floor)` — the **cumulative-drift** guard. Without it, each cycle
  could drop by `rollback_tolerance` relative to its predecessor and silently lose
  ~`N·tol` over N cycles. `_validation_baseline` is captured once at run start
  (mean of two rate-0 validations).
- `is_catastrophic` — a coarse, deliberately **non-SE-derived**
  `CATASTROPHIC_DROP_FACTOR` pre-recovery fast-fail (rejecting a statistically-wrong
  SE-derived "catastrophic" threshold was an explicit review decision).
- `ratchet_threshold(pre, margin, best, bound, floor)` — the NON-STALLING ratchet
  (`tuning_rollback_ratchet`): `max(pre − margin, best − bound, floor)`. The
  per-step relative bound keeps the ramp climbing (no best-anchor stall); the
  cumulative bound (`tuning_rollback_cumulative_bound`, default 0.05) caps total
  give-back below the best high-water mark and tightens as best ratchets up.
- **Paired McNemar gate (P2b)** — `paired_drop_se` / `paired_is_rollback` test the
  candidate vs the fixed baseline on a *shared* subsample's correctness vector:
  `(b10−b01)/N`, SE `√(b10+b01)/N` — several-fold tighter than the marginal
  `0.5/√n`. Behind `tuning_use_paired_sensor` (off → marginal path bit-exact),
  Monte-Carlo-calibrated in `test_paired_sensor_calibration`.

The `AdaptationTargetAdjuster` (`adaptation_target_adjuster.py`) relaxes the
target only after `_STUCK_STREAK_REQUIRED` (3) consecutive committed steps <1%,
and its decay is clamped to the same baseline-anchored floor.

### 3.4 LR discovery (the dominant per-cycle cost)

`learning_rate_explorer.py:LRRangeFinder` does a multi-step exponential LR sweep,
picking the *largest non-destructive* LR (highest LR that doesn't degrade beyond
`margin`, with `margin` derived from `accuracy_se`), centred on `pipeline_lr`
(`anchor_lr`, ±1 order of magnitude). Probes restore state after each step and are
tagged `(probe)` so they form a separate GUI trace. This sweep is the single
biggest per-cycle cost — the reason `use_paired_sensor` (+0.55% acc) costs 5.6× LR
time, and why `TunerBase._adaptation` skips LR discovery when `instant_acc` is
already near target and caches the LR across cycles (invalidated only on a
stuck-streak relaxation, or per `tuning_refind_lr_on_miss`).

### 3.5 The axes (rate application, the SSOT seam)

`axes/adaptation_axis.py:AdaptationAxis` is the homotopy-axis contract
(`attach`/`set_rate`/`calibrate`/`tunable_parameters`/`recovery_hooks`/`finalize`/…).
Each axis delegates math to `transformations/` and rate application to
`tuning/perceptron_rate.py`; it owns only the orchestration seam, and is a
transient per-run object (never pickled onto the model). The families:

- `axes/manager_rate_axis.py` — `AdaptationManager` rate fields
  (`quantization_rate`/`clamp_rate`/`activation_adaptation_rate` via an in-place
  shared `RateBuffer`, O(1) per step; `noise_rate` via the rebuild SSOT).
- `axes/blend_axis.py` — the KD-blend family: `BlendAxis.set_rate` →
  `perceptron_rate.set_blend_rate` (live `BlendActivation.rate`, no rebuild).
  **`TTFSGenuineAxis`** additionally anneals the spike surrogate `alpha`
  smooth→sharp on `_alpha_for_rate(r) = alpha_min·(alpha_max/alpha_min)**r` via
  `perceptron_rate.set_surrogate_alpha`. **`GenuineBlendAxis`** is *not* a
  `BlendAxis` — it mutates the installed `BlendedGenuineForward.rate` live (the
  blend is at the model *output*).

### 3.6 SSOT primitives (the leaf modules tuners share)

- `perceptron_rate.py` — `set_blend_rate` (every `BlendActivation.rate`),
  `apply_manager_rate` (one manager field + rebuild), `set_surrogate_alpha`,
  `rebuild_activations`. Replaced 4+ inlined `setattr; for p: update_activation`
  loops.
- `teacher.py` — `snapshot_frozen_teacher` (eval-mode, grad-frozen CPU deepcopy to
  distill against).
- `forward_install.py` — `LazyExecutorForward` (picklable build-once/drop-on-pickle
  base) + `CascadeForwardInstall` (single-owner install/remove) for installing a
  cross-layer NF as `model.forward`.
- `orchestration/genuine_probe.py` — `eval_forward_over_val(trainer, forward_obj,
  model, n, device)` (top-1 of an arbitrary forward over `iter_validation_batches`,
  never installed) and `genuine_acc_on_clone(model, device, *, prepare,
  build_forward, evaluate)` (genuine eval on a deepcopy, leaving live state /
  activation structure / installed forward untouched). Both callable-driven (no
  tuner imports), the basis for the cliff probe.

---

## 4. The KD-blend ANN→SNN ramp (the genuine-TTFS core)

`orchestration/kd_blend_adaptation_tuner.py:KDBlendAdaptationTuner` is the shared
base for LIF and TTFS:

- **Mechanism.** Snapshot a frozen teacher; install a `BlendActivation` (old→target
  by rate) on each perceptron's `base_activation`; ramp 0→1 with KD loss
  `α·CE + (1−α)·T²·KL` (T=3, α=0.3) recovering *during* the ramp.
- **Genuinely-gradual contract.** `_skip_one_shot=True` (no jump-to-1.0 probe) and a
  small uniform ladder (`_initial_ramp_step=0.125`, `_ramp_step_growth=1.0`,
  clamped to the budget's `min_step`) — the transform advances and recovers during
  the ramp instead of relabelling recovery as stabilization
  (`test_kd_blend_gradual_ramp.py`).
- **The default ramp is value-domain** (`_ramp_forward() → None`): rate 0 ==
  continuous teacher bit-exact, rate 1 == the pointwise on-chip composition. The
  genuine cross-layer dynamics are installed only at `_finalize()`.
- **probe ≡ deploy SSOT.** The deployed forward is built by the shared
  `_finalize_forward_for(model)` seam (`_finalize_forward() =
  _finalize_forward_for(self.model)`); LIF and TTFS override only that, and the
  genuine probe reuses it on a clone (via `_finalize_rebuild(model)`) so the probed
  forward is *identical* to the deployed forward.
- **Per-mode targets** (`tuners/`): LIF → `LIFActivation` + `_ChipAlignedNFForward`
  (`chip_aligned_segment_forward`); TTFS cascaded → `TTFSActivation` +
  `_SegmentSpikeForward` (the genuine single-spike cascade, kept installed through
  commit+recovery+downstream so the SCM gate doesn't trip — `[[ttfs_cycle_finetune
  _deploy_parity]]`); TTFS synchronized → no instance forward (the ramped blend's
  class forward *is* the analytical staircase) + a `TTFSInputGridQuantizer` STE on
  segment-entry perceptrons (the wire-contract grid snap).
- **The cliff is recorded.** `_after_run` records `finalize_cliff = ramp@1 −
  post_finalize` and `_report_cliff_probe_consistency` warns if the last per-commit
  `genuine_drop` and `finalize_cliff` (same gap, different anchor) diverge > 0.1.
  Post-finalize, `_max_stabilization_rounds=3` runs extra LR-restart stabilization
  while validation still improves — historically where a disproportionate share of
  cascaded deployed accuracy actually came from.

---

## 5. The finalize cliff: diagnosis and the genuine-ramp attempts

### 5.1 Stage 1 — the genuine full-transform probe (LANDED, opt-in)

`tuning_full_transform_probe` (default off), in `smooth_adaptation_cycle.py:
_probe_full_transform`: after each commit it measures BOTH
`value_full_acc` (`_value_full_transform_eval`, always a non-destructive
clone→rate-1.0→restore) AND `genuine_full_acc` (`_full_transform_eval`, base =
value, **overridden by `KDBlendAdaptationTuner`** to run the *deployed* forward at
rate 1.0 on a deepcopy via `genuine_acc_on_clone`). It logs
`{committed, committed_acc, value_full_acc, genuine_full_acc, value_drop,
genuine_drop, proxy_gap}` where `proxy_gap = value_full_acc − genuine_full_acc` is
the divergence the value proxy hides.

`smooth_adaptation_run.py:_log_full_transform_trend` keys the verdict on
`genuine_drop`: CONVERGING when `last.genuine_drop < first.genuine_drop`
(deployed cliff shrinking), else FLAT/DIVERGING — and when the genuine drop fails
to shrink while `value_drop` *did*, it warns **"the value-domain probe WAS
FOOLED"** and surfaces the `proxy_gap` trajectory. This is the instrumentation
that makes the cliff visible per-commit instead of once at finalize. It is the
prerequisite for trusting any cliff-closing change.

### 5.2 Stage 2a — surrogate-annealed genuine ramp (LANDED, opt-in)

`ttfs_genuine_annealed_ramp` (default off, cascaded only; `ttfs_ramp_alpha_min=0.5`,
`ttfs_ramp_alpha_max=2.0`), in `tuners/ttfs_cycle_adaptation_tuner.py`: train
*through* the genuine single-spike cascade for the WHOLE ramp instead of blending
the value domain. `_make_axis → TTFSGenuineAxis`; `_make_blend` returns the bare
`TTFSActivation` target (no value-blend ReLU side); `_after_install_blend` runs
`_finalize_rebuild` first so the segment policy finds genuine TTFS nodes;
`_ramp_forward()` installs the same `_finalize_forward_for(self.model)` cascade for
the whole ramp.

The key insight (`models/nn/activations/ttfs_spiking.py:_heaviside_surrogate`):
the surrogate forward is the **exact `pre>0` Heaviside**; `alpha` only shapes the
ATan *backward*. So annealing `alpha` changes gradient *conditioning* only — and
**rate=1 at `alpha_max` is bit-identical to the deployed cascade**, so the finalize
cliff is ~0 *by construction* and `_finalize` becomes a no-op.

### 5.3 Stage 2b — teacher→cascade blend ramp (LANDED, opt-in)

`ttfs_genuine_blend_ramp` (default off, cascaded only; mutually exclusive with the
annealed ramp — blend wins): shares the bare-`TTFSActivation` setup, then
`_calibrate_to_teacher_distribution` runs
`spiking.distribution_matching.match_activation_distributions(model, teacher,
cal_x, T, …)` on the deployed cascade (scale-aware `[0,1]` boundaries live-mutate
the perceptron scale `Parameter`s the bare nodes already reference — no second
rebuild). `_make_axis → GenuineBlendAxis`; `_ramp_forward()` installs
`BlendedGenuineForward(model, teacher, T, rate=0)` for the whole ramp (rate 0 =
frozen teacher exactly, rate 1 = genuine cascade exactly). `_make_kd_loss →
_BlendGenuineKDLoss` (KD+CE on the blend output + `ttfs_genuine_blend_ce_alpha`·CE
on the pure-genuine logits). `_finalize_forward` deploys the PURE
`_SegmentSpikeForward` (teacher dropped → cliff 0 by construction).

This is the principled successor to the *rejected* naive
`genuine_gradual_cascade_ramp` (a whole-model output blend that never made
intermediate reps genuine and never annealed conditioning, and lost final
accuracy): the blend is at the output but the *intermediate* reps are genuine
throughout, and distribution matching addresses the death cascade (§5.5).

### 5.4 The experimental fast genuine blend (LANDED, opt-in)

`ttfs_genuine_blend_fast` (default off, requires `ttfs_genuine_blend_ramp`;
`ttfs_blend_fast_steps_per_rate=120`, `ttfs_blend_fast_rates=[0.5,0.75,0.9,0.97,1.0]`):
`run()` bypasses the SmoothAdaptation control flow entirely. `_run_fast_genuine_blend`
builds ONE optimizer + a warmup(5%)/cosine LR over `len(rates)·steps_per_rate`
steps; for each fixed rate R it does `_set_rate(R)` + `steps_per_rate` training
steps with loss `CE((1−R)·teacher + R·genuine) + α·CE(genuine)`; then `_set_rate(1.0)`,
removes the ramp forward, runs `_finalize` (pure `_SegmentSpikeForward`). **No
`_adaptation` cycles, no RecoveryEngine, no rollback clone/restore, no
`_stabilize_at_full_rate`, no per-cycle LR find.** It reproduces
`generated/_genuine_ab/full_ramp.py`: **genuine 0.41 → 0.9355**. This is the
existence proof that a genuine-through-the-cascade ramp can recover cascaded
accuracy cheaply — the candidate for the "<2-minute" target — but it sidesteps the
controller's safety machinery, so it is experimental.

### 5.5 Death cascade and distribution matching (the enabling win)

The genuine ramp originally failed because a *cold* cascade kills neurons by depth
(each layer's quantized timing starves the next). `spiking/scale_aware_boundaries.py`
(opt-in θ_out + `input_scale = upstream`) plus DFQ per-neuron bias correction match
the genuine distribution to the teacher's: measured **cold cascade 0.10 → 0.41**,
after which the bottleneck shifts to ramp mechanics (floor / decay / LR) — which is
exactly what Stage 2 trains through. Encoding-layer scale pinning
(`scale_aware_boundaries.py`) separately fixed the NF↔SCM *deploy* parity
(0.656 → 1.0). These are "absolute wins" — active even under fast mode — not gated
behind the cliff-closing experiments.

---

## 6. The retention/cost landscape

### 6.1 Landed wins (default-on, here to stay)

| Win | Accuracy effect | Cost |
|---|---|---|
| scale-aware boundaries + DFQ bias correction | cold cascade 0.10 → 0.41 | one calibration pass |
| KD-blend gradual ramp (uniform 0.125, recover *during* ramp) | makes the ramp converge at all | the ramp |
| genuine forward kept installed through commit+recovery+downstream | prevents SCM-gate trip (0.9 → 0.6 failure) | none |
| encoding-layer scale pinning | NF↔SCM deploy parity 0.656 → 1.0 | none |
| LIF signed integrate-and-fire | exact per-neuron NF↔HCM parity | none |
| `_finalize_forward_for` SSOT (probe ≡ deploy) | correctness of the cliff measurement | none |

### 6.2 Speed / cost reductions (default-on)

| Change | Effect |
|---|---|
| LR cache + skip-when-near-target (`TunerBase._adaptation`) | cuts the dominant per-cycle LR-find cost |
| `CheckpointGuard` `cpu_pinned` async offload | ~3–8× faster snapshots, frees ~1× VRAM |
| subsample fast eval | faster; the one ablation flag that *improved* accuracy |
| `deployment_metric_full_eval` | killed the spurious NF↔SCM "drop" (metric artifact) |

### 6.3 Opt-in (benefit real, cost not worth default)

| Flag | Effect | Cost |
|---|---|---|
| `tuning_use_paired_sensor` (P2b McNemar) | +0.55% acc (tighter rollback) | 5.6× LR time |
| `tuning_rollback_ratchet` | bounds cumulative drift without stalling the ramp | none (gate only) |
| `tuning_stabilization_bounded` (+ratio 0.5) | single bounded cosine stabilization pass | replaces patience rounds |
| `tuning_persist_optimizer` | Adam moments survive across recovery | small state |
| `tuning_recipe_recovery` / `tuning_enable_characterization` / `tuning_tight_plateau` / `tuning_recovery_lr_plateau` / `tuning_refind_lr_on_miss` | recovery-quality knobs | varied |
| D4/D5 levers (label smoothing / SWA / staircase-KD) | small | config-gated off |

### 6.4 Reverted / refuted

- **Terminal stabilization** → hurt (deployed 0.924, SCM drop −0.027). Reverted.
- **`global_budget` floor** → backfired (default 0.0); now enforced non-negative.
- **D1 S-annealing + curriculum + RRAM** → removed (the rate-tuner extension point
  is the genericity, not a curriculum).
- **Naive `genuine_gradual_cascade_ramp`** (whole-model output blend) → lower final
  acc; superseded by §5.3.
- **DFQ-bias as the cliff cause** → refuted by a grad-norm probe (all layers got
  gradient 0.73–1.63).
- **Two statistically-wrong review recs** (a 0.005 D4 floor; an SE-derived
  catastrophic threshold) → rejected per ablation evidence.

### 6.5 The cost axis

- **LR finding dominates** per-cycle cost; everything that caches/skips it is the
  biggest speed lever.
- **Eval** is subsample (fast, ±0.02–0.04 noise) vs full (exact, slow); the
  statistical model (`accuracy_se`) sizes every tolerance off the subsample SE.
- **Wall-clock target:** a <2-minute session that recovers full accuracy, vs the
  temporary 30–40 s fast hack and the ~10-minute slow-but-correct path. The fast
  genuine blend (§5.4) is the leading candidate to hit it.

---

## 7. Flag taxonomy (every tuning-retention flag, default, effect)

All default-off → byte-identical to the prior path unless noted; the value-domain
proxy ramp is the default.

| Flag | Default | Effect |
|---|---|---|
| `tuning_full_transform_probe` | off | **Stage 1**: per-commit value + genuine cliff probe, `proxy_gap`, "fooled" verdict |
| `ttfs_genuine_annealed_ramp` (`ttfs_ramp_alpha_min`/`_max`) | off | **Stage 2a**: train through cascade, anneal surrogate α (cliff ~0) |
| `ttfs_genuine_blend_ramp` (`ttfs_distmatch_*`, `ttfs_genuine_blend_ce_alpha`) | off | **Stage 2b**: teacher→cascade output blend + distribution matching |
| `ttfs_genuine_blend_fast` (`ttfs_blend_fast_steps_per_rate`/`_rates`) | off | experimental fast path (0.41 → 0.9355), bypasses the controller |
| `ttfs_finetune_kd_against_rung2` | off | synchronized-only: KD teacher = IR-mapped rung-2 contract flow |
| `tuning_use_paired_sensor` | off | P2b paired-McNemar rollback gate (+0.55% acc, 5.6× LR) |
| `tuning_rollback_ratchet` (`tuning_rollback_cumulative_bound`) | off | non-stalling cumulative-drift cap |
| `tuning_stabilization_bounded` (`tuning_stabilization_ratio`) | off | single bounded cosine stabilization pass |
| `tuning_refind_lr_on_miss` / `tuning_recovery_lr_plateau` / `tuning_tight_plateau` | off | recovery-quality LR knobs |
| `tuning_persist_optimizer` | off | persistent optimizer across recovery calls |
| `tuning_enable_characterization` | off | pre-search α-grid profiling → scheduler epsilon/policy |
| `tuning_recipe_recovery` | off | STEP recovery honors the recipe optimizer + schedule |
| `deployment_metric_full_eval` | **on** | full-test-set deployment metric (kills subsample artifact) |

---

## 8. Open questions & next steps

1. **Promote a genuine ramp (the headline).** Stage 1 makes the cliff visible;
   Stage 2a/2b/fast can close it on the bench (0.41 → 0.9355 fast). The gate is a
   **full-run accuracy non-regression vs the proxy baseline** — run the genuine
   annealed and blend ramps to completion through `run.py`, confirm `finalize_cliff
   ≈ 0`, `proxy_gap ≈ 0` per-commit, and final acc ≥ the 0.928 cascaded baseline,
   then flip a default. The fast path needs the controller's safety machinery (or a
   justification to skip it) before it can be more than experimental.
2. **Bounded-recovery controller** for the ~2 pp ramp under-training inside a
   <2-minute budget — likely `tuning_stabilization_bounded` + `tuning_rollback_ratchet`
   + persistent optimizer, ablated together rather than singly.
3. **S-ceiling** (~2.7 pp) is orthogonal: higher S lifts it but costs sim time and
   doesn't help the cliff; revisit only after the cliff is closed.
4. **Driver standalone (V6 polish):** construct `AdaptationDriver` directly from the
   services rather than hosting phases on the mixins — unblocks reusing the loop
   outside the tuner classes.

---

## 9. Code-path index

**Control loop & services** — `tuning/orchestration/`:
`adaptation_driver.py` (orchestrator + `CycleContext`), `smooth_adaptation_run.py`
(`run`, `_log_full_transform_trend`), `smooth_adaptation_cycle.py` (phase methods,
`_probe_full_transform`), `smooth_adaptation_tuner.py` (composed tuner),
`tuner_base.py`, `acceptance_sensor.py`, `recovery_engine.py`, `rate_scheduler.py`,
`characterization.py`, `checkpoint_guard.py`, `tuning_budget.py`,
`adaptation_manager.py` / `adaptation_manager_factory.py`, `genuine_probe.py`,
`kd_blend_adaptation_tuner.py`.

**SSOT primitives** — `tuning/`: `perceptron_rate.py`, `teacher.py`,
`forward_install.py`, `trace.py`, `learning_rate_explorer.py`,
`adaptation_target_adjuster.py`, `adaptation_rate_tuner.py`, `shift_calculation.py`.

**Axes** — `tuning/axes/`: `adaptation_axis.py`, `blend_axis.py`
(`BlendAxis`/`TTFSGenuineAxis`/`GenuineBlendAxis`), `manager_rate_axis.py`,
`perceptron_transform_axis.py`, `pruning_axis.py`, `activation_shift_axis.py`.

**Concrete tuners** — `tuning/tuners/`: `lif_adaptation_tuner.py`
(`LIFAdaptationTuner`, `_ChipAlignedNFForward`), `ttfs_cycle_adaptation_tuner.py`
(`TTFSCycleAdaptationTuner`, `_SegmentSpikeForward`, `_BlendGenuineKDLoss`),
plus `activation_*`, `clamp_tuner.py`, `noise_tuner.py`, `pruning/`,
`perceptron_transform_tuner.py`, `normalization_aware_perceptron_quantization_tuner.py`.

**Genuine-cascade & distribution matching** —
`models/nn/activations/ttfs_spiking.py` (`TTFSActivation`, `_heaviside_surrogate`,
`set_surrogate_alpha`), `models/spiking/training/blended_genuine_forward.py`
(`BlendedGenuineForward`), `spiking/scale_aware_boundaries.py`,
`spiking/distribution_matching.py`, `spiking/chip_aligned_nf.py`.

**Deployment parity (the retention check)** —
`pipelining/core/nf_scm_parity.py`, `pipelining/core/simulation_factory.py`
(`run_trainer_metric`, `deployment_metric_full_eval`).

**Plans / prior reports** — `characteristics-investigation-the-mutable-babbage.md`
(the two-stage cliff plan), `docs/fine_tuning_research_directions.md`,
`docs/gradual_finetuning_report.md`.
