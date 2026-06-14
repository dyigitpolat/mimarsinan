# mimarsinan — Architecture Review & Refactoring Plan for Non-Destructive Gradual Model Transformation

**Repository:** `github.com/dyigitpolat/mimarsinan` (shallow clone, `src/mimarsinan`)
**Subsystem under review:** `tuning/` (with `models/nn/decorators/`, `model_training/`, and the `pipelining/.../adaptation` steps)
**Document type:** Architecture review + refactoring plan + test/scale strategy
**Companion:** the *Behavioral Specification* produced earlier; section references like *(spec §6)* point to it.

---

## 0. Executive Summary

The `tuning/` subsystem already *is* an implementation of non-destructive gradual model transformation, and a surprisingly complete one. The ANN→SNN conversion problem (clamp, activation shift, activation quantization, weight quantization, noise injection, pruning, LIF/TTFS spiking dynamics) is exactly a family of behavioral transformations driven from rate `0 → 1`, with recovery training and rollback. The core ideas from the spec are present: adaptive step sizing with rollback, statistically-derived tolerances, a fixed baseline with an anti-drift floor, validation-aware LR discovery with restore-after-probe, and a dataset-sized compute budget.

The weaknesses are not in the *ideas* but in their *factoring*. The same algorithm is implemented two-and-a-half times; the transformation abstraction is implicit and split across three incompatible application mechanisms; the controller is a ~120-line god-method mixin that fuses sensing, recovery, rollback, and target relaxation; the statistical gate is a marginal worst-case estimate evaluated on *unpaired* subsamples; and several hot paths (full-model checkpoint clones per probe, whole-val-set GPU caching, optimizer rebuilt per cycle) will not scale cleanly to multi-GB models or ImageNet-scale validation sets.

This report maps the concept onto the code, reviews the architecture candidly, then proposes a set of refactoring vectors organized around one idea: **make the `Transformation` a first-class object and reduce the tuner to a thin orchestrator of four small, independently testable services** — a rate scheduler, an acceptance sensor, a recovery engine, and a checkpoint guard. The same restructuring is what unlocks the scalability work.

---

## Part I — What Implements the Concept

The mapping from the spec's components to the code is direct. This is the single most important table in the report; the rest of the review is organized around the rows.

| Spec concept *(spec §)* | Implementation | Verdict |
|---|---|---|
| Predictor–corrector continuation loop *(§2, §5)* | `smart_smooth_adaptation.py::SmartSmoothAdaptation.adapt_smoothly` **and** `orchestration/smooth_adaptation_run.py::SmoothAdaptationRunMixin.{run,_continue_to_full_rate}` | Works, but **duplicated** across 2–3 loops |
| One round = predictor + corrector + commit/rollback *(§5.2)* | `orchestration/smooth_adaptation_cycle.py::SmoothAdaptationCycleMixin._adaptation` | Correct, but **god-method** |
| Transformation contract / interpolation semantics *(§4)* | `orchestration/adaptation_manager.py::AdaptationManager` + `models/nn/decorators/{adjustment,clamp_quantize,transforms}.py` | **Implicit**, three rival mechanisms |
| Interpolation modes: functional blend / stochastic mask *(§4.2)* | `MixAdjustmentStrategy` (blend), `RandomMaskAdjustmentStrategy` (mask), `NestedAdjustmentStrategy` | Good primitives, no declared contract |
| Adaptive LR discovery, restore-after-test *(§7.1)* | `learning_rate_explorer.py::{LRRangeFinder, find_lr_range_for_trainer, clone/restore_state_for_trainer}` | **Strong**; minor scale issues |
| Recovery / corrector loop with divergence guard *(§7.2–7.3)* | `model_training/basic_trainer_steps.py::train_steps_until_target` (patience, best-state, min-improvement) | **Strong** |
| Statistical acceptance gate *(§6)* | `tuning_budget.py::TuningBudget.accuracy_se` (`0.5/√n`), `_rollback_tolerance = max(3·SE, 3·empirical_noise)` | Present but **marginal & unpaired** |
| Probe vs commit separation *(§6.4)* | `validation_context("probe")` tagging; `progress_eval_batches` (cheap) vs `eval_n_batches` (decision) | Tagging only; **no independent confirm split** |
| Fixed baseline + anti-drift floor *(§8.1)* | `_validation_baseline` (captured once), `_absolute_post_acc_floor`, `AdaptationTargetAdjuster`, `pipelining/core/accuracy_budget.py::AccuracyBudget` | **Excellent** — this is exactly right |
| Probe/catastrophic gate *(§6.4)* | `tuner_base.py::CATASTROPHIC_DROP_FACTOR = 0.8` | Works; magic constant |
| Termination / partial result *(§5.4)* | `min_step`, `max_cycles`, `_attempt_recovery_if_below_floor`, warnings | Works; spread across files |
| Compute budget from data size *(§12, scalability)* | `tuning_budget.py::TuningBudget.from_dataset` (`check_interval=√SPE`, eval capped to ~5k samples) | **Strong** design instinct |
| Stabilization at full rate *(§5)* | `smooth_adaptation_run.py::_stabilize_at_full_rate` (rollback-bracketed) | Good |
| Checkpoint / rollback (non-destructive) *(§9 I1/I3)* | `clone_state_for_trainer` / `restore_state_for_trainer`, `_clone_state/_restore_state` | Correct; **not scale-aware** |
| Per-axis pipeline sequencing | `pipelining/pipeline_steps/adaptation/*.py` + `quantization/*` (each a `TunerPipelineStep`) | Clean step DAG; axes fully sequential |

**Reading of the design intent.** Comments throughout (`perceptron_rate.py`, `forward_install.py`) reference "higher-level `Transformation` axis objects" that should "share one application path." That abstraction is *aspirational in the comments but absent in the code* — the refactor below is largely about actually building it.

---

## Part II — Architecture Review

### II.1 Strengths worth preserving

These are not throwaway compliments; the refactor must *retain* each of them, and the test suite should pin them.

- **Statistically-grounded thresholds.** Every accuracy comparison is a multiple of `accuracy_se()` rather than a hardcoded percentage (`tuning_budget.py`, `_rollback_tolerance`). This is the right instinct and rare in practice.
- **Anti-drift via a fixed baseline.** `_validation_baseline` is captured once at run start and the rollback gate takes `max(relative, baseline·(1−tol), pipeline_hard_floor)` (`_absolute_post_acc_floor`). The `tuning/ARCHITECTURE.md` even articulates the cumulative-drift argument explicitly. This is the spec's §8.1 implemented correctly, including the cross-step `AccuracyBudget`.
- **Restore-after-probe discipline.** `LRRangeFinder.find_best_lr` clones state, sweeps, and restores in a `finally`; `train_steps_until_target` keeps the *best* state, not the last. Non-destructiveness is taken seriously.
- **Test-set isolation.** Tuner internals never call `trainer.test()`; exploratory evals are tagged `(probe)`. This prevents the classic leak of selecting against the test set.
- **Budget derived from data.** `check_interval = √(steps_per_epoch)`, eval capped so commit/rollback decisions stay "<30s," LR probes bounded. The system already thinks in terms of wall-clock per decision.
- **Large-backbone memory awareness exists in spots.** `SavedTensorDecorator` detaches, subsamples (deterministic linspace), and moves to CPU inside the forward — with a docstring explaining the ViT-B OOM it prevents. AMP autocast, fused Adam on CUDA, and optional grad clipping are all wired in `basic_trainer.py`.
- **Symmetric, single-owner forward patching.** `CascadeForwardInstall` asserts against double-patch and is idempotent on removal — exactly the discipline cross-layer forwards need.

### II.2 Weaknesses, by theme (with severity)

**W1 — No first-class Transformation abstraction (Severity: High).** Three rival mechanisms apply a rate:
1. `AdaptationRateTuner` family → set a global `AdaptationManager.<rate_attr>` field, then full per-perceptron rebuild via `apply_manager_rate → rebuild_activations → update_activation` (quant, noise, shift, activation-replacement, and clamp-via-manager).
2. `ClampTuner` → a bespoke `SmoothAdaptationTuner` subclass with its own learnable-scale parameter, regulariser, saturation probing, and freeze step — yet clamp *also* exists in the manager.
3. `KDBlendAdaptationTuner` family → `set_blend_rate` mutates `perceptron.base_activation.rate` directly, plus a `forward_install` override.

There is no interface that says "a transformation is a thing with `apply(model, α)`, `calibrate`, `interpolation_mode`, `monotonicity_guarantee`, `is_stochastic`, `tunable_parameters`, `finalize`." Consequently each new axis is a new *tuner subclass* re-implementing `_update_and_evaluate`, `_after_run`, `_recovery_training_hooks`, `_stabilization_budget`. The orchestration logic and the transformation logic are entangled.

**W2 — The rate-search algorithm is implemented 2–3 times (Severity: High).** `run()` first tries a one-shot `_adaptation(1.0)`; on failure it builds a `SmartSmoothAdaptation` (grow-on-commit ×1.5 / halve-on-rollback); and *then* `_continue_to_full_rate` runs yet another loop (step = remaining/4, its own attempt cap, its own halving). Three policies for one job, chained, with subtly different step math and termination conditions. This is the most bug-prone part of the system and the hardest to reason about. Note also that the grow/halve policy in `SmartSmoothAdaptation` differs from the spec's greedy-to-1.0 + bisect-the-gap (§5.2): `_continue_to_full_rate` is closer to the spec, which is itself evidence the team converged on the spec's policy but left the old one in place.

**W3 — `_adaptation` is a god-method (Severity: High).** One ~120-line method in `smooth_adaptation_cycle.py` performs: reporting, pre-state clone, pre-cycle eval (with a reuse optimization), probe eval, catastrophic gate, LR fetch/caching, recovery training (with hooks), the dual rollback gate, the stuck-streak target-relaxation state machine, and cycle logging. Each concern is individually reasonable; fused, they are nearly impossible to unit-test in isolation, and the control flow (early returns to "committed rate" as the rollback signal) is implicit.

**W4 — The acceptance test is marginal and unpaired (Severity: High; also a scalability lever).** `accuracy_se = 0.5/√n` is the Bernoulli *worst-case* (p=0.5) marginal SE; for a 95%-accurate model the true marginal SE is ~0.49× smaller, and the *paired* SE of the drop (only discordant examples contribute — spec §6.2) is smaller still. Worse, `pre_cycle_acc` and `post_acc` are measured on *different* batches (the `iter_validation_batches` cursor advances), so the comparison is unpaired and the difference carries the sum of both subsamples' variance. The consequence is an inflated `_rollback_tolerance`, which forces smaller steps and more cycles than necessary — directly slowing the tuning process.

**W5 — No independent confirmation split → multiple-testing inflation (Severity: Medium).** Many cycles each test "no regression" against the same rotating validation cache. Across dozens of cycles, the probability that some lucky subsample passes a step that is actually regressive grows (spec §6.5). There is no held-out *confirm* split distinct from the *search* signal.

**W6 — Full-model checkpoint clones on every probe and cycle (Severity: High at scale).** `clone_state_for_trainer` clones the entire `state_dict` on-device. This happens: once per cycle (pre-state), 8× inside each LR sweep (probe restore), and on every rollback. For a multi-GB model this pins ≥2× model memory and makes each LR discovery an 8-clone affair. The team deliberately keeps clones on-device ("no GPU↔CPU memcpy") — a speed choice that becomes a memory wall on large models.

**W7 — Optimizer state is discarded every call (Severity: Medium, speed).** `_get_optimizer_and_scheduler_steps` builds a fresh Adam in `train_n_steps`, `train_steps_until_target`, and `train_one_step`. Adam moments are thrown away between the LR sweep and recovery and between cycles, costing re-warmup and slowing convergence.

**W8 — Whole-validation-set GPU caching (Severity: High at scale).** `basic_trainer_eval._build_gpu_val_cache` materializes the *entire* validation loader onto the device, then `iter_validation_batches` rotates a cursor over it. Decisions are correctly capped to ~5k samples, but the *cache* is the full set. On ImageNet-scale val sets this OOMs; the cache should hold only the fixed decision subsample.

**W9 — Rate changes reallocate the decorator stack (Severity: Medium).** `apply_manager_rate` rebuilds every perceptron's `TransformedActivation` and decorator objects from scratch on each rate change, because the rate is a Python float baked into freshly-constructed decorators rather than a live buffer. During a ramp this is repeated allocation churn and prevents a clean "set α in place" semantics.

**W10 — Stochastic axes are treated as if monotone & deterministic (Severity: Medium).** The quantization path nests `RandomMaskAdjustmentStrategy`, and `NoisyDropout` injects noise; both are stochastic, so the same α yields different accuracy each eval, and monotonicity of distortion in α is not guaranteed. The controller's bisection assumes monotone feasibility, and decisions are not reproducible (no per-decision seeding; the rate=0 short-circuit even notes it "shifts global RNG state"). There is no characterization phase (spec §10) to detect non-monotonicity or to average the gate over noise draws.

**W11 — Probe sensitivity data is logged but unused (Severity: Low/Medium, speed).** `_cycle_log` records `(rate, instant_acc, post_acc, outcome)` but the next step size is chosen by fixed grow/halve, not by the observed sensitivity curve (spec §5.6). The information needed to take larger, smarter steps is collected and discarded.

**W12 — Axes are fully sequential 0→1 passes (Severity: Low; design tradeoff).** The pipeline runs clamp → shift → activation-quant → weight-quant → noise → … each as an independent full continuation with its own recovery. This is robust and debuggable but is N sequential continuations; there is no shared-recovery or interleaved-axis option for when wall-clock matters most.

---

## Part III — Refactoring Vectors

The organizing principle: **one orchestrator, four small services, N transformation objects.** Today the tuner *is* the algorithm and each transformation is a tuner subclass. The target inverts this — the algorithm is a fixed, well-tested orchestrator parameterized by a `Transformation` and four injected collaborators:

```
                    ┌────────────────────────────────────────┐
                    │            AdaptationDriver             │  (was SmoothAdaptationTuner)
                    │  one loop, no transformation knowledge  │
                    └───┬───────┬────────────┬────────────┬───┘
                        │       │            │            │
                 RateScheduler  │      RecoveryEngine  CheckpointGuard
                (greedy+bisect) │      (LR + train +   (scoped, async,
                                │       divergence)     diffable)
                          AcceptanceSensor
                          (paired, probe vs confirm)
                                │
                                ▼
                        Transformation  ◄── apply(model, α) / calibrate / finalize
                        (Clamp, ActQuant, WeightQuant, Noise, Blend, LIF, TTFS, Pruning…)
```

Each vector below states the problem, the current code, the proposed shape (illustrative — not drop-in), the migration path, and the risk. They are ordered so that earlier vectors unblock later ones.

### V1 — Introduce the `Transformation` protocol (resolves W1, W9, W10)

**Problem.** No interface unifies the three rate-application mechanisms; each axis is a tuner subclass.

**Proposed shape.** A single protocol every axis implements. It is the spec's Transformation Contract (§4) made concrete for this codebase.

```python
class Transformation(Protocol):
    name: str
    interpolation_mode: Literal["functional_blend", "parameter_path", "stochastic_mask"]
    monotonicity: Literal["guaranteed", "expected", "none"]
    is_stochastic: bool

    def attach(self, model) -> None:
        """Build the (rate-buffered) decorator stack ONCE. Idempotent."""

    def set_rate(self, alpha: float) -> None:
        """Set α in place — no reallocation. Writes a registered buffer."""

    def calibrate(self, model, batches) -> None:
        """Data-dependent params (quant scales, clamp p99). Deterministic."""

    def tunable_parameters(self) -> Iterable[nn.Parameter]:
        """Transform-owned params that join recovery (e.g. learnable clamp scale)."""

    def recovery_hooks(self, alpha) -> list:            # was _recovery_training_hooks
    def finalize(self, model) -> None:                  # was _after_run / freeze / forward-install
    def set_decision_seed(self, seed: int) -> None:     # reproducible stochastic eval (W10)
    def descriptor(self) -> str:                        # stable hash for caching/golden tests
```

`AdaptationManager` becomes the *registry/host* of attached transformations rather than a bag of float fields, and the rate is a **registered buffer mutated in place**:

```python
class RateBuffer(nn.Module):
    def __init__(self): super().__init__(); self.register_buffer("alpha", torch.zeros(()))
    def set(self, a): self.alpha.fill_(float(a))
# RateAdjustedDecorator reads self.rate.alpha instead of a baked float -> set_rate is O(1), no rebuild
```

**Migration.** Wrap the existing decorators behind `Transformation` adapters first (zero behavior change): `ClampTransformation`, `ActQuantTransformation`, `WeightQuantTransformation`, `NoiseTransformation`, `BlendTransformation`, `LIFTransformation`, `TTFSTransformation`, `PruningTransformation`. Each adapter delegates to today's `update_activation`/`set_blend_rate`/forward-install. Then, incrementally, move the rate from a baked float to the `RateBuffer` so `set_rate` stops rebuilding (W9). The `tuners/*.py` files collapse into these transformation objects + config.

**Risk.** Low if adapters are introduced first; the decorator math is unchanged. The in-place-rate change (W9) is the only behavior-affecting step and is guarded by the smoothness tests already present (`test_clamp_smoothness.py`).

### V2 — Collapse the rate search into one `RateScheduler` (resolves W2)

**Problem.** Three loops (`_adaptation(1.0)` one-shot, `SmartSmoothAdaptation`, `_continue_to_full_rate`) implement one algorithm.

**Proposed shape.** A single strategy object implementing the spec's policy (§5.2): each round greedily attempts the full jump to 1.0, bisects the *remaining gap* on failure, commits, and repeats — with the optional sensitivity-guided first step (V8).

```python
class RateScheduler:
    def __init__(self, *, epsilon: float, policy="greedy_to_one", history=None): ...

    def run(self, committed: float, attempt: Callable[[float], CycleResult]) -> float:
        while committed < 1.0 - 1e-6:
            gap = 1.0 - committed
            step = self._first_step(gap)                 # gap (greedy) or sensitivity-predicted
            took = False
            while step >= self.epsilon:
                res = attempt(committed + step)          # ONE cycle: probe→recover→confirm
                self.history.record(res)                 # feeds V8
                if res.committed:
                    committed = committed + step; took = True; break
                step /= 2.0
            if not took:
                return committed                          # partial result (spec §5.4)
        return committed
```

`AdaptationDriver.run()` becomes: calibrate baseline → `scheduler.run(0.0, self._cycle)` → `transformation.finalize()` → `stabilize()`. The one-shot is just the first iteration with `step == gap`; no special case. Delete `SmartSmoothAdaptation` and `_continue_to_full_rate`.

**Migration.** Land `RateScheduler` behind a feature flag; assert (golden-trace test, V-test in Part IV) that for the existing mock/integration cases the committed trajectory and final rate match within tolerance, then remove the old loops.

**Risk.** Medium — this is the control core. Mitigated by golden-trace equivalence tests and by the fact that `_continue_to_full_rate` already encodes the target policy.

### V3 — Extract `AcceptanceSensor`; make it paired (resolves W3 partly, W4, W5)

**Problem.** Threshold logic is inlined in `_adaptation`; the gate is marginal and unpaired; no confirm split.

**Proposed shape.** A service that owns *all* accept/reject/recovered decisions and the eval splits.

```python
class AcceptanceSensor:
    def __init__(self, trainer, budget, *, k_commit=2.0, k_probe=None,
                 search_split, confirm_split):           # disjoint fixed subsamples (W5, W8)
        ...
    def reference(self, model) -> RefStats:              # captured once (spec §8.1)
    def probe_drop(self, model) -> Decision:             # cheap, search_split, paired vs reference
    def paired_se(self, ref_correct, cand_correct):      # McNemar: sqrt(b01+b10)/N  (spec §6.2)
    def recovered(self, model) -> bool:                  # vs fixed baseline - k·SE
    def confirm(self, model) -> Decision:                # confirm_split, paired, authoritative
```

The key change is paired evaluation on **fixed** example sets: cache per-example correctness vectors for the reference once, and at each probe evaluate the candidate on the *same* examples, deriving the drop's SE from discordant counts only. This typically shrinks the SE several-fold, which lets `_rollback_tolerance` shrink, which lets steps grow — fewer cycles for the same safety (a speed win as well as a correctness win).

Replace `accuracy_se = 0.5/√n` with the paired SE for *decisions*; keep `0.5/√n` only as a conservative fallback when correctness vectors are unavailable.

**Migration.** Introduce `AcceptanceSensor` wrapping today's `validate_n_batches` first (unpaired, to prove the extraction), then switch the internal estimator to paired correctness vectors. Partition the validation set deterministically into `search` and `confirm` once at run start.

**Risk.** Low–medium. Paired stats are strictly more powerful; the main care item is ensuring the candidate and reference see identical examples (fix the subsample seed and disable shuffling for these passes).

### V4 — Extract `RecoveryEngine` with persistent optimizer state (resolves W3 partly, W7)

**Problem.** Recovery training, LR fetch, and divergence handling are inlined; Adam is rebuilt every call.

**Proposed shape.** A service that owns LR discovery + the train-to-target loop + divergence/rollback signaling, and an *optimizer-state policy*.

```python
class RecoveryEngine:
    def __init__(self, trainer, budget, lr_finder, *, optimizer_policy="persist_within_cycle"):
        ...
    def discover_lr(self, transformation) -> float:      # wraps LRRangeFinder; noise-averaged (V7)
    def recover(self, target, *, params) -> RecoveryResult:   # train_steps_until_target + guards
```

Persist Adam moments at least within a recovery cycle and across the LR sweep→recovery handoff (configurable reset on large α jumps). `params` comes from `model.parameters()` *plus* `transformation.tunable_parameters()` — this is how the learnable clamp scale and any future LSQ-style quantizer parameters join recovery uniformly, replacing `ClampTuner`'s bespoke wiring.

**Migration.** Move `train_steps_until_target` invocation and the catastrophic/divergence guards out of `_adaptation` into `RecoveryEngine.recover`. Change `basic_trainer_steps` to accept an externally-owned optimizer (optional) so state can persist.

**Risk.** Medium — persistent optimizer state changes dynamics. Gate behind config; default to current behavior until the perf/quality tests (Part IV) show the win.

### V5 — Extract `CheckpointGuard`: scoped, async, diffable (resolves W6, partly W3)

**Problem.** Full on-device `state_dict` clones per probe/cycle/rollback.

**Proposed shape.** A guard that snapshots only what can change, off the critical device memory, with a fast path for probes.

```python
class CheckpointGuard:
    def __init__(self, model, *, scope="tunable", location="cpu_pinned", mode="snapshot"):
        # scope="tunable": clone only params with requires_grad (skip frozen backbone)
        # location="cpu_pinned": async d2h copy; restore via h2d (frees 1× model of VRAM)
        # mode="cow"/"diff": optional — store only params mutated since snapshot
    def snapshot(self) -> Handle
    def restore(self, handle: Handle) -> None
    @contextmanager
    def bracket(self): ...    # snapshot on enter, restore on rollback signal
```

For the LR sweep specifically, snapshot **once** before the 8 probes and restore from that single handle between probes, instead of cloning per probe. For frozen-backbone tuning (common when only late layers adapt), `scope="tunable"` alone can cut checkpoint memory by 10–100×.

**Migration.** Replace `clone_state_for_trainer`/`restore_state_for_trainer` call sites with `CheckpointGuard`. Keep on-device snapshot as an opt-in `location="device"` for small models (preserves today's speed choice).

**Risk.** Low. Async d2h needs a stream sync before restore; the existing `MIMARSINAN_CUDA_DEBUG` sync hook shows the team already handles this pattern.

### V6 — Reduce the tuner to `AdaptationDriver` (resolves W3 fully)

With V1–V5 in place, the god-method dissolves. The driver is short and reads like the spec's pseudocode:

```python
class AdaptationDriver:
    def __init__(self, model, transformation, sensor, recovery, guard, scheduler, budget): ...

    def run(self) -> float:
        self.transformation.attach(self.model)
        self.sensor.reference(self.model)                 # fixed baseline (spec §8.1)
        committed = self.scheduler.run(0.0, self._cycle)
        self.transformation.finalize(self.model)
        self._stabilize()
        return committed

    def _cycle(self, alpha) -> CycleResult:
        with self.guard.bracket() as snap:
            self.transformation.set_rate(alpha)           # O(1), in place (V1)
            self.transformation.calibrate(self.model, self.budget.calib_batches)
            if not self.sensor.probe_drop(self.model).ok:         # probe gate
                return CycleResult.rejected(alpha)
            lr = self.recovery.discover_lr(self.transformation)
            rec = self.recovery.recover(self.sensor.target,
                                        params=self._params())
            if rec.diverged or not self.sensor.confirm(self.model).ok:  # confirm split (V3)
                snap.rollback();  return CycleResult.rejected(alpha)
            return CycleResult.committed(alpha, rec)
```

Every concept from the spec is now one named call. The target-relaxation state machine (stuck-streak → `AdaptationTargetAdjuster`) moves into the `AcceptanceSensor`'s target policy, where it belongs, and is unit-testable on its own.

### V7 — Make LR discovery scale- and noise-aware (resolves part of W6, W10; speed)

`find_lr_range_for_trainer` is already anchored (±1 order around `pipeline_lr`) and validation-aware — keep that. Add: (a) **single baseline snapshot** for the sweep (via V5) instead of per-probe clones; (b) for `is_stochastic` transformations, **average the probe metric over R noise draws** with a fixed decision seed so the chosen LR is not a noise artifact; (c) optionally replace full-validation scoring of each probe with a cheaper *loss-slope* or *gradient-noise-scale* signal for the coarse pass, reserving validation accuracy for the final 2–3 candidates. (c) is the biggest wall-clock lever on large models because it removes most eval passes from the inner LR loop.

### V8 — Use the sensitivity history to choose steps (resolves W11; speed)

`_cycle_log` already holds `(rate, drop)` points. Feed them to the `RateScheduler` as a monotone secant/Lipschitz estimate of `drop(Δα)` and set the round's *first* step to the largest predicted-feasible increment, falling back to bisection on miss (spec §5.6). On smooth axes this stays greedy; on cliff-like axes (low-bit quant, hard clamp) it stops wasting a full probe-and-bisect cascade every round. Cache probe results keyed by `(transformation.descriptor(), committed_hash, alpha)` so an α is never re-probed against an unchanged committed model.

### V9 — Add a characterization phase (resolves W10; robustness)

Before the search, run the spec's §10 profiling on a handful of α values: measure the paired drop curve (monotonicity → if a significant decrease appears, downgrade `monotonicity` to `none` and switch the scheduler to dense-grid safe mode), estimate max local slope (→ set `epsilon`), check surrogate-gradient health for non-differentiable axes (quant STE via `StaircaseFunction`, clamp via `DifferentiableClamp`), and measure stochastic spread (→ set R for V7). `ClampTuner._probe_clamp_saturation` is a partial version of this; generalize it to all transformations and store the profile with the run for reproducibility and golden tests.

### V10 — Optional: shared-recovery / interleaved axes (addresses W12; advanced)

Today each axis is an independent 0→1 continuation. Two optional accelerations, presented with their risk: (a) **warm-start** each axis's recovery from the previous axis's converged optimizer state instead of a cold Adam; (b) **interleaved multi-axis continuation** — a vector rate `α ∈ [0,1]^K` advanced jointly, with the scheduler stepping the *least-sensitive* axis first. (b) is a research-grade change (spec Appendix C) and should be gated behind strong tests; (a) is low-risk and reuses V4's persistent optimizer state.


---

## Part IV — Testing & Validation Strategy

The repo already has 339 test files, including a strong `tests/unit/tuning/` set (`test_smart_smooth_adaptation.py`, `test_rollback.py`, `test_commit_guard.py`, `test_cumulative_floor.py`, `test_adaptation_target_adjuster.py`, `test_clamp_smoothness.py`, `test_learning_rate_explorer.py`, …). The strategy below *builds on* that suite; the new service boundaries make most of it faster and more focused.

### IV.1 The refactor's safety net: golden-trace equivalence

Before deleting any of the three rate loops (V2) or changing the gate (V3), record a **golden decision trace** — the sequence `(alpha_proposed, outcome, committed_after, lr_chosen)` — for a fixed seed on each existing mock and small integration case. The new `RateScheduler`/`AcceptanceSensor` must reproduce these traces within tolerance. This converts "did the refactor change behavior?" from a judgment call into a test. `_cycle_log` already contains nearly everything needed; promote it to a structured, serialized artifact.

### IV.2 Mock-transformation zoo (the core controller test — spec §14.2)

The single highest-value addition. Replace the model+training with cheap analytic `Transformation` mocks exposing a known `accuracy(alpha, recovery_progress)` surface, so the *driver/scheduler/sensor* logic runs in milliseconds with no GPU. Archetypes, each asserting invariants I1–I7 hold throughout and the committed trajectory matches expectation:

| Mock | Surface | Asserts |
|---|---|---|
| `SmoothMonotone` | gentle linear drop | reaches α=1 within round bound |
| `Cliff` | sharp drop past α* | bisection finds α* within ε, never commits past it |
| `PlateauThenDrop` | flat then cliff | greedy big step rejected then bisected |
| `Stochastic` | drop + injected noise | false-accept/reject within design (ties to IV.3) |
| `NonMonotone` | drop dips then rises | controller detects, switches to safe mode (V9) |
| `RecoveryLimited` | recovers slowly/partially | budget/abort/rollback exercised |
| `AdversarialTiming` | infeasible early, feasible later | still converges to α=1 |

These tests are what give confidence that *one* controller is stable across heterogeneous transformations — they span the profile space by construction, deterministically, in seconds. Today's `test_smart_smooth_adaptation.py` and `test_adaptation_stress.py` are the seeds of this; generalize them to the protocol.

### IV.3 Statistical validation of the gate (spec §14.3)

Monte-Carlo the `AcceptanceSensor` against Bernoulli correctness with a *known* true drop: measure empirical false-accept and false-reject rates vs nominal across many simulated runs; verify the paired (McNemar) SE estimator against ground truth; repeat with injected transform stochasticity to validate the noise-averaged SE (V7). Calibrate `k_commit`, the split sizes, and R to hit target error rates. This is pure-Python and fast. It also produces the numbers that justify shrinking `_rollback_tolerance` (V3) — close the loop by feeding the calibrated tolerance back into the budget.

### IV.4 Transformation conformance tests (spec §14.1)

For every real `Transformation`, assert the contract: `set_rate(0)` reproduces original outputs within fp tolerance; `set_rate(1)` equals the full transform; `attach` is idempotent; `set_rate` does **not** reallocate (assert object identity of the decorator stack — pins V1/W9); declared gradients are finite where promised and the surrogate matches finite differences within tolerance (quant STE, clamp); `calibrate` is deterministic; `set_decision_seed` makes stochastic outputs reproducible. Existing `test_napq_rate_aware.py`, `test_weight_quantization*.py`, `test_clamp_learnable_scale.py`, `test_per_layer_rate.py` cover pieces of this; the protocol lets them run as one parameterized suite over all axes.

### IV.5 Metamorphic tests (spec §14.7)

Relations that must hold without a ground-truth optimum: tightening `k_commit` never increases final committed α; increasing `recovery_budget` weakly increases it; reducing `epsilon` weakly increases it for cliff mocks; larger eval splits reduce decision variance; identical seeds/config produce identical traces (I6). These guard against silent regressions that pointwise tests miss.

### IV.6 Checkpoint/rollback chaos tests (spec §14.8)

Force a recovery divergence and assert exact restore (bitwise on tunable params) — extend `test_rollback.py`/`test_commit_guard.py` to the `CheckpointGuard` scopes (`tunable`, `cpu_pinned`, `cow`). Kill a process mid-recovery and assert the on-disk "current" model is never worse than the last commit (I1/I3). Feed a `NonMonotone` mock and assert safe-mode engagement.

### IV.7 Scalability / performance gates (CI budget assertions)

Turn the existing "<30s per decision" intent into enforced gates. On a fixed small-but-real integration (e.g. the ViT clamp probe already in `tests/integration/`), assert: total checkpoint bytes allocated ≤ budget; LR-discovery wall-clock ≤ budget; probe count per round ≤ `ceil(log2(gap/ε))`; peak VRAM ≤ budget. These catch scalability regressions (a stray full-model clone, a whole-val-set cache) at PR time rather than on the next ImageNet run.

### IV.8 Determinism harness

A single fixture that seeds torch/numpy/python, snapshots and restores RNG around probes, and asserts trace reproducibility (I6). This is the prerequisite for golden traces (IV.1) to be meaningful for stochastic axes.

---

## Part V — Scaling to Large Datasets & Large Models with Fast Cycles

"Fast adaptation cycles + large data + large models" decomposes into four cost centers: **evaluation**, **checkpointing**, **LR discovery**, and **recovery training**. Each has a concrete target below; several are direct consequences of Part III.

### V.1 Evaluation cost (large datasets)

- **Cache only the decision subsample on GPU, not the whole val set (W8).** Replace `_build_gpu_val_cache`'s full-loader materialization with a fixed, seed-stratified subsample sized to `eval_sample_count` (~5k for SE≈0.007). This is the single change that makes the system safe on ImageNet-scale validation. Stream anything larger; never pin the full set.
- **Paired correctness vectors (V3).** Evaluating reference and candidate on the *same* fixed examples both tightens the SE (fewer cycles) and lets the sensor cache the reference correctness vector once and reuse it for every probe — turning each probe into a single candidate pass with no separate reference eval.
- **Two-tier eval, enforced.** Keep `progress_eval_batches` (16) for in-loop convergence checks and the larger split only for commit/confirm — this is already the design; V3 just makes the commit split *independent* (W5).

### V.2 Checkpoint cost (large models)

- **Scope to tunable params (W6/V5).** If the backbone is frozen during an axis's tuning, never clone it. For partial fine-tuning this alone removes most checkpoint memory.
- **Offload to pinned CPU, async (V5).** Frees ~1× model of VRAM versus on-device clones; the d2h/h2d copies overlap with compute. Keep on-device as an opt-in for small models.
- **One snapshot per LR sweep, not per probe (V5/V7).** Removes 7 of every 8 clones in LR discovery.
- **Diff/COW snapshots (optional).** For very large models, store only params mutated since the snapshot; recovery typically moves a small fraction far.

### V.3 LR-discovery cost (per cycle)

- **Cheaper coarse signal (V7).** Score the coarse sweep by loss-slope / gradient-noise-scale and reserve full-validation scoring for the top 2–3 candidates; this removes most of the 8 validation passes per sweep.
- **Cache across cycles (already present), invalidate on relaxation (already present).** Keep `_get_cached_lr`'s reuse; extend it to reuse the persistent optimizer state (V4) so a cache hit skips warmup too.

### V.4 Recovery training cost (per cycle)

- **Persistent optimizer state (V4/W7).** Stop discarding Adam moments between the LR sweep and recovery and between cycles; this shortens each recovery.
- **Larger justified steps → fewer cycles.** The paired sensor (V3) and sensitivity-guided stepping (V8) both reduce the *number* of cycles, which is the highest-order term in total wall-clock. This is the most important scalability point: the cheapest cycle is the one you don't run.
- **Keep AMP + fused Adam + grad clip** (already in `basic_trainer.py`); ensure they're on the recovery path for all axes.
- **Activation memory (large models).** `SavedTensorDecorator`'s CPU subsampling is the right pattern; make it the default everywhere stats are collected (calibration, saturation probing, characterization V9) so no axis pins full activation tensors across the forward.

### V.5 Multi-device / very large models (forward-looking)

The controller is implicitly single-process. For models that require sharding, three semantics need definition before a distributed run: (1) **checkpoint/rollback under DDP/FSDP** — `CheckpointGuard` must snapshot/restore sharded params consistently (FSDP `state_dict` with full/sharded policy); (2) **sensor all-reduce** — paired correctness counts must be reduced across ranks so every rank makes the same accept/reject decision (decision must be rank-invariant for I6); (3) **rate application** — `set_rate` writing a registered buffer broadcasts cleanly, which is another reason to prefer the buffer over rebuilt decorators (W9/V1). None of this needs to be built now, but the service boundaries in Part III are what make it tractable later: only `CheckpointGuard` and `AcceptanceSensor` become distribution-aware; the driver, scheduler, and transformations do not change.

### V.6 What *not* to change for scale

Resist the urge to push every axis to a single joint continuation (V10b) purely for speed — the sequential per-axis design is what makes failures debuggable and rollbacks clean. Spend the scalability budget first on evaluation, checkpointing, and cycle-count reduction (V.1–V.4), which are pure wins with no robustness cost.

---

## Part VI — Phased Roadmap & Risk

Ordered for incremental, always-green delivery. Each phase ships behind a flag with the golden-trace and mock-zoo tests gating the flip.

| Phase | Vectors | Outcome | Risk | Gate |
|---|---|---|---|---|
| 0 | IV.1, IV.8 | Golden traces + determinism harness | none | n/a |
| 1 | V1 (adapters only) | `Transformation` protocol wrapping existing axes; zero behavior change | low | conformance IV.4 |
| 2 | V3 | `AcceptanceSensor` extracted, then paired + confirm split | med | IV.3 calibration matches |
| 3 | V5, V4 | `CheckpointGuard` (scoped/async), `RecoveryEngine` (persistent opt) | med | IV.6 + perf gates IV.7 |
| 4 | V2, V6 | One `RateScheduler`, `_adaptation` god-method dissolved | med-high | golden-trace equivalence |
| 5 | V1 (in-place rate), V8, V9 | buffer-rate, sensitivity stepping, characterization | med | mock zoo IV.2 + smoothness |
| 6 | V7, V.1–V.4 scale work | large-model/dataset readiness | med | scalability gates IV.7 |
| 7 | V10 (optional) | shared/interleaved recovery | high | research-grade test bar |

**Top risks and mitigations.** (1) *Behavioral drift when collapsing the loops (V2)* — mitigated by Phase 0 golden traces and by `_continue_to_full_rate` already encoding the target policy. (2) *Paired-gate miscalibration (V3)* — mitigated by the Monte-Carlo suite (IV.3) before the tolerance is tightened. (3) *Persistent optimizer state changing convergence (V4)* — gated by config, default-off until perf/quality tests show the win. (4) *Async checkpoint races (V5)* — mitigated by the existing CUDA-debug sync pattern and explicit stream syncs before restore.

**The one-sentence version.** Make `Transformation` a real object, reduce the tuner to a thin driver over a scheduler / paired sensor / recovery engine / checkpoint guard, and the system becomes simultaneously more elegant (one loop, named concepts), more correct (paired stats, confirm split, characterization), and more scalable (scoped async checkpoints, subsampled eval, fewer cycles) — without discarding the genuinely good instincts already in the codebase (SE-derived tolerances, fixed-baseline anti-drift, restore-after-probe, data-sized budgets).
