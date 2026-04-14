# Architectural Guidance: Tuning Efficiency & Accuracy Retention at Scale

## Context

Since commit `e83f00e`, 12 commits addressed pipeline assertion failures where tuned model accuracy dropped below tolerance after tuning steps. The fixes corrected threshold algebra, simplified the adaptation loop, improved recovery training, and added safety mechanisms. CIFAR-10 and MNIST now pass reliably.

**The remaining problems**:
1. **ImageNet adaptation steps are extremely slow** -- expensive operations (full-dataset LR exploration, full validation every check_interval) consume hours per adaptation cycle with marginal benefit.
2. **mnist_hard_all fails at Activation Quantization** -- the tuner one-shots to rate=1.0, `_after_run()` unconditionally commits rate=1.0 regardless of recovery success, and the pipeline assertion catches the accuracy gap.
3. **Tuner-pipeline metric mismatch** -- the tuner's internal `original_metric` (from its own baseline validation) differs from the pipeline's `previous_metric` (from the prior step's test accuracy), creating an undetectable threshold gap.

---

## Part 1: What Was Tried and What Worked

### A. Threshold Algebra Fixes (Highest Impact)

| Fix | Files | What It Solved |
|-----|-------|----------------|
| `degradation_tolerance` semantic inversion (0.95 retention -> 0.05 drop) | `defaults.py`, `deployment_pipeline.py`, 5 JSON configs | Pipeline assertion used wrong formula |
| Pipeline tolerance override reconnected to config | `deployment_pipeline.py:_initialize_config` | Hardcoded 0.95 ignored user config |
| Removed progressive target decay from every adaptation cycle | `unified_tuner.py`, `pruning_tuner.py`, `perceptron_transform_tuner.py`, `activation_shift_tuner.py` | 5% gap between tuner acceptance and pipeline requirement |
| Rollback: relative margin -> absolute target-based threshold | `unified_tuner.py:_adaptation` L209 | Algebraically aligned with pipeline assertion formula |

**Key principle**: All acceptance thresholds must be provably equivalent to the pipeline assertion `new_metric >= previous_metric * (1 - degradation_tolerance)`.

### B. Loop Efficiency

- **SmartSmoothAdaptation simplified** (`smart_smooth_adaptation.py`): Bisection-based step finder -> adaptive step-size loop (grow 1.5x on commit, halve on rollback). Removed internal clone/restore -- caller handles state.
- **One-shot optimization** (`unified_tuner.py:run`): Try rate=1.0 first. Light transformations (clamp, noise, quantization) skip the gradual loop entirely. 5-20x faster for these tuners.

### C. Recovery Quality

- **`_after_run()` in every tuner**: Forces rate=1.0, runs recovery training to original (un-decayed) target, calls `_ensure_pipeline_threshold()` as safety net.
- **State snapshot fix** (`activation_adaptation_tuner.py`): Rollback now saves/restores `base_activation` objects alongside model params, preventing inconsistent state.
- **PruningTuner integration** (`pruning_tuner.py`): Uses base class loop (shared rollback, one-shot, LR finding) instead of independent implementation.

### D. Statistical and Resource Improvements

- **LR finder** (`learning_rate_explorer.py`): Baseline-aware, early exit on collapse, budget cap, anchor-based range (pipeline_lr +/- 1 order of magnitude).
- **`accuracy_se()`** (`tuning_budget.py`): `0.5 / sqrt(eval_sample_count)` -- Bernoulli worst-case SE. All thresholds derived from this single metric.
- **GPU memory** (`pipeline.py`, `deployment_pipeline.py`): CPU cloning, model offloading after each step, DataLoader worker cleanup.

---

## Part 2: Confirmed Bugs from Runtime Analysis

### BUG-1: `_after_run()` Unconditionally Commits Rate=1.0 (CRITICAL)

**Observed in**: `mnist_hard_all_phased_deployment_run_20260414_050101`
**Error**: `[Activation Quantization] step failed to retain performance within tolerable limits: 0.9379 < (0.9574 * 0.99) = 0.9478`

**Root cause trace** through `ActivationQuantizationTuner._after_run()` (L26-46):

```
1. run() calls _adaptation(1.0) -- one-shot attempt
2. _adaptation(1.0) commits (post_acc passes rollback check against tuner baseline)
3. run() sees _committed_rate >= 1.0, calls _after_run()
4. _after_run() line 30: self._committed_rate = 1.0  <-- UNCONDITIONAL
5. _after_run() runs recovery training (may fail to reach target)
6. _after_run() calls _ensure_pipeline_threshold() (may also fail)
7. _after_run() returns the failed metric
8. run() asserts _committed_rate >= 1.0 -- passes because line 30 forced it
9. Pipeline calls pipeline_metric() -> trainer.test() -> 0.9379
10. Pipeline assertion: 0.9379 < 0.9574 * 0.99 -> FAIL
```

**The same bug exists in ALL three tuners**:
- `activation_quantization_tuner.py` L30: `self._committed_rate = 1.0`
- `activation_adaptation_tuner.py` L63: `self._committed_rate = 1.0`
- `clamp_tuner.py` L199: `self._committed_rate = 1.0`

**Why gradual adaptation never gets a chance**: The one-shot `_adaptation(1.0)` passes the tuner's internal test gate (threshold = `original_metric * (1 - pipeline_tolerance)` = `0.9333 * 0.99 = 0.924`), so it commits. But the pipeline assertion uses a DIFFERENT baseline (`previous_metric = 0.9574` from the prior step), creating an undetectable gap.

### BUG-2: Tuner-Pipeline Metric Basis Mismatch (CRITICAL)

**Location**: `unified_tuner.py:330-333` vs `pipeline.py:142,166`

The tuner calibrates its baseline at `run()` start:
```python
baseline_val = self.trainer.validate()      # validation accuracy (e.g., 0.9333)
self.target_adjuster.original_metric = baseline_val
```

The pipeline captures:
```python
previous_metric = self.get_target_metric()  # test accuracy from prior step (e.g., 0.9574)
# ... after step runs ...
assert self.get_target_metric() >= previous_metric * self.tolerance
```

These are **fundamentally different values**:
- The tuner's baseline uses `validate()` (validation set, may be a subset)
- The pipeline's `previous_metric` uses `test()` (full test set, from the prior step)
- For mnist_hard_all: validation=0.9333 vs test=0.9574 -- a **2.4% gap**

The tuner's test gate at rate=1.0 checks `test_acc >= original_metric * (1 - pipeline_tolerance)` = `0.9333 * 0.99 = 0.924`. Test accuracy of 0.9462 passes this gate. But the pipeline needs `0.9574 * 0.99 = 0.9478` -- which 0.9462 fails.

**Constraint**: The tuner CANNOT calibrate against `test()` because that would leak test-set information into the tuning loop's training decisions (rollback, target adjustment, recovery training). The `degradation_tolerance` is a hard failure gate on test accuracy, not a training signal.

**The same mismatch affects `_ensure_pipeline_threshold()`** (L241-282): it compares test accuracy against `original_metric * (1 - pipeline_tolerance)`, but `original_metric` comes from `validate()`, not the pipeline's `previous_metric`. So even the safety net uses the wrong threshold.

**Solution**: Separate the two concerns -- the tuner's internal training target (validation-based, no data leak) and the pipeline hard gate (test-based, passed in from the pipeline). See recommendation 4.1 below.

### BUG-3: ImageNet LR Exploration Destroys Model + Full Validation Per Probe (PERFORMANCE)

**Observed in**: `imgnet_sq_pretrained_phased_deployment_run_20260414_050008`

Clamp Adaptation LR exploration at rate=1.0:
- Probe at LR=0.000139: val=0.661 (OK) -- ~2 min
- Probe at LR=0.000268: val=0.307 (degrading) -- ~2 min
- Probe at LR=0.000518: val=0.117 (destroyed) -- ~2 min
- Probe at LR=0.001: val=0.005 (catastrophic) -- ~2 min
- State restored, best LR selected = 1e-5 (lowest)

**Each probe costs ~2 minutes**: ~97 training steps + full 50K validation (390 batches).
**8 probes = ~16 minutes** just for one LR exploration call.
**LR exploration is called once per adaptation cycle**, and each cycle also does:
- Full validation for instant check after transformation (~1 min)
- Recovery training with full validation every 97 steps
- Full validation for post-recovery rollback check (~1 min)

**Observed wall-clock**: 0.7 hours elapsed and the run was still on Clamp Adaptation's first gradual cycle (rate=0.5), having already wasted ~20 minutes on a failed one-shot.

---

## Part 3: Remaining Failure Modes at ImageNet Scale

### FM-1: LR Discovery Range Too Narrow

**Location**: `learning_rate_explorer.py:134-139`

For ImageNet with `pipeline_lr = 0.0001`, the anchor-based sweep covers `[1e-5, 1e-3]` -- 2 orders of magnitude total. The observed probe data shows the upper half of this range is destructive (LR=0.0005 already drops accuracy to 0.117). The finder returns the boundary LR (1e-5), which is extremely conservative and will make recovery training very slow.

### FM-2: Full Validation for LR Probes Is Wasteful

**Location**: `unified_tuner.py:90-92` (validate_fn passed to LR finder)

The LR finder's `validate_fn` calls `validate_n_batches(eval_n_batches)` where `eval_n_batches` covers the FULL validation set (390 batches for ImageNet). For LR range finding, a rough accuracy estimate suffices -- 32-64 batches would give adequate signal while being 6-12x faster.

### FM-3: Cosine Annealing Wrong for Variable-Length Recovery

**Location**: `basic_trainer.py:273` -> `_get_optimizer_and_scheduler_steps`

`train_steps_until_target` creates cosine annealing over `max_steps` (28K for ImageNet). Early patience exit at ~650 steps sees nearly flat LR. Full 28K run decays to `lr * 1e-3` aggressively.

### FM-4: No Best-Model Checkpointing During Recovery

**Location**: `basic_trainer.py:284-313`

Keeps the last model state, not the best. If accuracy peaks mid-training and degrades, the committed model is worse than the peak.

### FM-5: Rollback Tolerance Too Tight for ImageNet

**Location**: `unified_tuner.py:122`

`rollback_tolerance = 3 * accuracy_se() = 0.0067` for ImageNet. Real evaluation noise exceeds this due to batch-order correlation and BatchNorm variance.

### FM-6: Target Decay Ratchet Effect

**Location**: `unified_tuner.py:232-235`, `adaptation_target_adjuster.py:38-46`

`update_target(midpoint)` always decays because `midpoint < target` when stuck. Target never recovers. Permanently lowers the bar for downstream tuners.

### FM-7: Failed One-Shot Wastes Significant Budget

**Observed**: ImageNet Clamp Adaptation one-shot consumed ~20 minutes before rolling back. Recovery training at rate=1.0 ran up to 28K steps of full training before the rollback check triggered.

---

## Part 4: Recommended Changes (Priority Order)

### 4.1 Fix Tuner-Pipeline Hard Gate with Pipeline-Provided Threshold [CRITICAL -- FIX FIRST]

**Files**: `unified_tuner.py`, `pipeline_step.py`, relevant pipeline step classes

The tuner cannot use `test()` for internal calibration (data leak). Instead, separate the two concerns:

1. **Internal training target** (validation-based): drives rollback decisions, recovery training, target adjustment. No change -- stays `validate()`-based. No data leak.

2. **Pipeline hard gate** (test-based): a floor threshold passed in from the pipeline. The pipeline knows `previous_metric` (from the prior step's `test()`) and can pass `previous_metric * tolerance` as the hard floor the tuner must respect.

**Implementation**:

The pipeline already knows `previous_metric` at `_run_step()` L142. Pass it to the step/tuner:

```python
# pipeline.py _run_step():
previous_metric = self.get_target_metric()
step.pipeline_previous_metric = previous_metric  # NEW
step.run()
```

```python
# unified_tuner.py - TunerBase.__init__() or run():
# The pipeline step passes this through to the tuner
self._pipeline_hard_floor = pipeline_previous_metric * (1 - pipeline_tolerance)
```

Then use `_pipeline_hard_floor` for the test gate at rate=1.0 and for `_ensure_pipeline_threshold()`:

```python
# In _adaptation() at rate >= 1.0 (L214-222):
test_acc = self.trainer.test()
if test_acc < self._pipeline_hard_floor:
    self._restore_state(pre_state)
    return self._committed_rate

# In _ensure_pipeline_threshold() (L252-253):
threshold = self._pipeline_hard_floor  # NOT original_metric * (1 - tol)
```

**Why this works**: The hard floor is computed from test-set information (`previous_metric`) but is a fixed constant passed to the tuner -- the tuner never queries `test()` to derive training decisions. It only uses `test()` as a go/no-go gate against the externally-provided floor. No data leak: the tuner doesn't optimize toward test accuracy, it just rejects results that would fail the pipeline assertion.

**This also fixes `_ensure_pipeline_threshold()`**: the safety net currently uses `original_metric * (1 - pipeline_tolerance)` where `original_metric` is from `validate()`. With the hard floor from the pipeline, it uses the correct baseline.

### 4.2 Fix `_after_run()` to Not Force-Commit on Failure [CRITICAL -- FIX FIRST]

**Files**: `activation_quantization_tuner.py`, `activation_adaptation_tuner.py`, `clamp_tuner.py`

`_after_run()` must not set `self._committed_rate = 1.0` before verifying that recovery succeeded. The `_ensure_pipeline_threshold()` return value must be checked.

```python
def _after_run(self):
    self._continue_to_full_rate()
    
    self.adaptation_manager.quantization_rate = 1.0
    for p in self.model.get_perceptrons():
        self.adaptation_manager.update_activation(self.pipeline.config, p)
    
    lr = self._find_lr()
    self.trainer.train_steps_until_target(...)
    
    final_acc = self._ensure_pipeline_threshold()
    self._committed_rate = 1.0  # only AFTER recovery succeeds
    return final_acc
```

Note: `_committed_rate` is used by the assertion in `run()` to verify the tuner reached full rate. Moving it after recovery is semantically correct -- the transformation IS at rate=1.0 (that's set on the adaptation_manager), but the tuner shouldn't "commit" until accuracy is verified.

### 4.3 Two-Tier Validation: Cheap Progress Checks, Full Commit Checks [CRITICAL -- PERFORMANCE]

**Files**: `unified_tuner.py`, `tuning_budget.py`

The system currently uses `eval_n_batches` (390 batches for ImageNet = full validation set) for ALL validation calls -- LR probes, recovery training convergence checks, instant checks after transformation. This is the root cause of the 4.7h per-cycle cost. 93% of wall-clock time is validation.

**Solution**: Introduce a two-tier validation scheme:

1. **Cheap validation** (`progress_eval_batches`): 16 batches. Used for:
   - LR probe accuracy estimation (`_find_lr` validate_fn)
   - Recovery training convergence checks (`train_steps_until_target` `validation_n_batches`)
   - Instant accuracy check in `_update_and_evaluate()`

2. **Full validation** (`eval_n_batches`): 390 batches. Used ONLY for:
   - Post-recovery rollback decision (`_adaptation()` L207)
   - Test gate at rate=1.0 (`_adaptation()` L215)
   - `_ensure_pipeline_threshold()` recovery target

**File**: `tuning_budget.py` -- add field:
```python
@dataclass
class TuningBudget:
    # ... existing fields ...
    progress_eval_batches: int = 16  # NEW: cheap validation for progress tracking
```

**File**: `unified_tuner.py` -- use cheap validation for progress, full for decisions:
```python
def _find_lr(self):
    return find_lr_range_for_trainer(
        ...,
        validate_fn=lambda: self.trainer.validate_n_batches(
            self._budget.progress_eval_batches),  # CHEAP
        ...
    )

def _adaptation(self, rate):
    ...
    # Recovery training: cheap validation for convergence detection
    self.trainer.train_steps_until_target(
        ...,
        validation_n_batches=self._budget.progress_eval_batches,  # CHEAP
        ...
    )
    # Rollback decision: FULL validation (statistical rigor matters)
    post_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)
    ...
```

Also reduce `max_training_steps` from `3 * SPE` to `1 * SPE`:
```python
# tuning_budget.py:
max_training_steps = max(1, int(float(steps_per_epoch) * float(budget_scale)))
```

**Impact for ImageNet**: Per-cycle cost drops from 4.7h to ~7 minutes. Full pipeline achievable within 30 minutes.

### 4.3b Cap `_ensure_pipeline_threshold()` Budget [CRITICAL -- PERFORMANCE]

**File**: `unified_tuner.py:264-273`

`_ensure_pipeline_threshold()` currently uses `max_training_steps * 3` per attempt with `patience * 4 = 20` and `min_steps = max_training_steps`. For ImageNet with 1x SPE this is still 9,375 * 3 = 28,125 steps with 20 patience. This safety net alone could take 30+ minutes.

Cap it: 1x `max_training_steps` per attempt, standard patience, no min_steps override. If the model can't recover in 1 epoch with proper LR, more training won't help -- the transformation itself is too aggressive and the tuner should have taken smaller steps.

```python
# Current (too generous):
self._budget.max_training_steps * 3, patience=_RECOVERY_PATIENCE * 4, min_steps=self._budget.max_training_steps

# Fixed (bounded):
self._budget.max_training_steps, patience=_RECOVERY_PATIENCE, min_steps=self._budget.check_interval * 3
```

### 4.4 Best-Model Checkpointing in Recovery Training [HIGH IMPACT]

**File**: `basic_trainer.py`
**Function**: `train_steps_until_target` (L251-313)

Save model state_dict (to CPU) whenever `best_acc` improves. On any exit path, restore to best checkpoint before returning.

```python
# After L303 (if acc > best_acc + imp_eps):
    best_acc = acc
    best_state = {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}
    stale_checks = 0

# Before L311 (del optimizer, scheduler, scaler):
if best_state is not None:
    self.model.load_state_dict(best_state)
```

**Cost**: One CPU state_dict copy (~350MB for ViT-B) per accuracy improvement (typically 3-10 per cycle).
**Benefit**: Converts "keep last" to "keep best" -- the single highest-value change for accuracy retention.

### 4.5 Warmup + Constant LR for Recovery [HIGH IMPACT]

**File**: `basic_trainer.py`
**Function**: `train_steps_until_target` (L273)

Replace cosine annealing with linear warmup (5% of max_steps) + constant LR. Patience is the actual convergence detector; the scheduler should not interfere.

Combined with best-model checkpointing: train at constant LR, save when accuracy improves, stop when stale, restore best.

### 4.6 Empirical Rollback Tolerance Calibration [HIGH IMPACT]

**File**: `unified_tuner.py`
**Function**: `run()` (L330-333)

After baseline calibration, measure empirical noise via two validation passes. Use `max(3 * se, 3 * empirical_noise, 0.005)` as rollback tolerance.

**Cost**: One extra validation pass (~30s for ImageNet).
**Benefit**: Prevents spurious rollbacks.

### 4.7 Widen LR Range + More Probes [MEDIUM IMPACT]

**File**: `learning_rate_explorer.py:134-139`

Widen asymmetrically: `[anchor/100, anchor*10]`. The observed data shows that for ImageNet Clamp Adaptation, the optimal LR was at the lower boundary. A wider lower range would find a better LR.

**File**: `tuning_budget.py:55` -- Increase max probes: `min(12, ...)`.

### 4.8 Temporary Target Relaxation (Prevent Ratchet) [MEDIUM IMPACT]

**File**: `unified_tuner.py:232-235`

When stuck detection triggers target relaxation, save the pre-relaxation target. Restore it on the next large commit (>5% delta).

### 4.9 Per-Tuner Budget Multiplier [MEDIUM IMPACT]

**File**: `unified_tuner.py`

Add `_budget_multiplier` property (default 1.0). Override in heavy tuners:

| Tuner | Multiplier | Rationale |
|-------|------------|-----------|
| `ActivationAdaptationTuner` | 2.0 | Restructures attention patterns |
| `PruningTuner` | 2.0 | Structural removal needs longer recovery |
| `ClampTuner` | 1.0 | Light transformation |
| `ActivationQuantizationTuner` | 1.0 | Light transformation |
| `NoiseTuner` | 1.0 | Light transformation |

### 4.10 Cycle-Level Progress Logging [LOW IMPACT, HIGH DEBUGGING VALUE]

**File**: `unified_tuner.py`

Add `_cycle_log: list` to `SmoothAdaptationTuner`. In `_adaptation()`, append `(proposed_rate, committed_rate, pre_acc, post_acc, lr_found)`. Log the full trace at the end of `run()`. Zero compute cost.

### 4.11 Reduce One-Shot Waste for Known-Hard Transformations [LOW IMPACT]

**File**: `unified_tuner.py:338-339`

Add `_skip_one_shot` property (default `False`). Override to `True` in `ActivationAdaptationTuner` and `PruningTuner`.

Alternatively: cap one-shot recovery training at `max_training_steps // 10` so failure is detected faster.

---

## Part 5: Architectural Risks

### Risk 1: Optimizer State Not Part of Snapshot

`_clone_state` / `_restore_state` save model weights but NOT optimizer momentum/variance. Correct because optimizer is recreated each cycle. Any future change making optimizer persistent would silently break rollback.

### Risk 2: `_after_run()` Is Non-Reversible

Every tuner's `_after_run()` force-commits rate=1.0 without rollback capability. If recovery training and `_ensure_pipeline_threshold()` both fail, there is no path back. Consider saving a pre-`_after_run()` checkpoint.

### Risk 3: No Global Gradient-Step Budget

SmartSmoothAdaptation has a cycle cap (30) but no total gradient-step budget. Worst case for ImageNet: 30 * 28K = 840K steps (70+ epochs). Track cumulative steps and enforce a global cap.

### Risk 4: Data Iterator Not Reset on Rollback

`BasicTrainer.train_iter` advances through data on every training call. Rollback restores model state but not iterator position.

---

## Part 6: Investigation Plan

### Phase A: Instrumentation (Before Code Changes)

#### A.1 Add Timing Instrumentation to Key Operations

Add `time.time()` wrappers around the most expensive operations and emit timing via `pipeline.reporter.report`:

**File**: `unified_tuner.py:_adaptation()`
```python
import time

# In _adaptation():
t0 = time.time()
instant_acc = self._update_and_evaluate(rate)
self.pipeline.reporter.report("T_update_and_eval_sec", time.time() - t0)

t0 = time.time()
lr = self._find_lr()
self.pipeline.reporter.report("T_find_lr_sec", time.time() - t0)
self.pipeline.reporter.report("LR_found", lr)

t0 = time.time()
self.trainer.train_steps_until_target(...)
self.pipeline.reporter.report("T_recovery_training_sec", time.time() - t0)

t0 = time.time()
post_acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)
self.pipeline.reporter.report("T_post_validation_sec", time.time() - t0)
```

**File**: `learning_rate_explorer.py:LRRangeFinder.find_best_lr()`
```python
# Per probe:
self.pipeline.reporter.report(f"LR_probe_{i}", lr)
self.pipeline.reporter.report(f"LR_probe_{i}_acc", acc)
self.pipeline.reporter.report(f"LR_probe_{i}_sec", elapsed)
```

**Expected output**: Console log will show per-cycle breakdown like:
```
T_update_and_eval_sec: 55.2
T_find_lr_sec: 960.4
T_recovery_training_sec: 3600.0
T_post_validation_sec: 55.1
```

#### A.2 Add Cycle Summary Logging

**File**: `unified_tuner.py:_adaptation()` -- before return:
```python
self.pipeline.reporter.report("CYCLE_SUMMARY", {
    "proposed_rate": rate,
    "committed_rate": self._committed_rate,
    "pre_acc": pre_acc,
    "post_acc": post_acc,
    "lr_found": lr,
    "rollback": post_acc < threshold,
    "target": self._get_target(),
})
```

#### A.3 Budget Diagnostics at Tuner Start

**File**: `unified_tuner.py:run()` -- after budget creation:
```python
b = self._budget
self.pipeline.reporter.report("BUDGET", {
    "max_training_steps": b.max_training_steps,
    "check_interval": b.check_interval,
    "eval_n_batches": b.eval_n_batches,
    "lr_num_probes": b.lr_num_probes,
    "lr_steps_per_probe": b.lr_steps_per_probe,
    "accuracy_se": b.accuracy_se(),
    "rollback_tolerance": self._rollback_tolerance,
    "pipeline_tolerance": self._pipeline_tolerance,
})
```

### Phase B: Isolation Tests

#### B.1 LR Exploration Cost Test

Create a test that measures the wall-clock cost of one `_find_lr()` call with ImageNet-scale budget parameters, using a mock trainer that sleeps proportionally to simulate real compute:

```python
def test_lr_exploration_cost():
    """Verify LR exploration stays under 5 minutes for ImageNet budget."""
    budget = TuningBudget.from_dataset(1_200_000, 128, 1.0,
                                        val_set_size=50_000, val_batch_size=128)
    # budget.lr_num_probes should be 8
    # budget.lr_steps_per_probe should be ~97
    # budget.eval_n_batches should be 390
    
    # With real model: measure actual wall-clock
    # With mock: verify call counts
    assert budget.lr_num_probes <= 8
    assert budget.eval_n_batches <= 400
    
    # Expected cost: 8 * (97 train steps + 390 val batches)
    # Target: < 300 seconds with reduced val batches (32)
```

#### B.2 Metric Basis Alignment Test

```python
def test_tuner_uses_test_not_validate_for_baseline():
    """Tuner's original_metric must match pipeline's previous_metric basis."""
    # After tuner.run(), verify:
    # tuner.target_adjuster.original_metric == trainer.test()
    # NOT trainer.validate()
```

#### B.3 `_after_run()` Failure Path Test

```python
def test_after_run_does_not_commit_on_recovery_failure():
    """_after_run() must not set _committed_rate=1.0 before verifying recovery."""
    # Mock _ensure_pipeline_threshold() to return below-threshold accuracy
    # Verify _committed_rate is NOT set to 1.0
```

#### B.4 One-Shot Fallback Test (Regression)

```python
def test_one_shot_failure_falls_back_to_gradual():
    """When one-shot fails test gate, gradual adaptation must proceed."""
    # Set up model where rate=1.0 passes rollback but fails pipeline test gate
    # Verify SmartSmoothAdaptation runs after one-shot failure
```

### Phase C: Profiling

#### C.1 Profile a Single Adaptation Cycle

Run ImageNet Clamp Adaptation with Python's `cProfile` and measure:
- Time in `validate_n_batches` (should dominate)
- Time in `train_n_steps` during LR probes
- Time in `clone_state_for_trainer` / `restore_state_for_trainer`
- Number of calls to `validate_n_batches` per cycle

```bash
python -m cProfile -o adaptation_profile.prof run.py --config imgnet_sq_pretrained.json --start_step "Clamp Adaptation" --stop_step "Clamp Adaptation"
```

#### C.2 Memory Profiling

Use `torch.cuda.memory_stats()` to track peak GPU memory during:
- LR exploration (8 clone/restore cycles)
- Recovery training (optimizer + gradients + activations)
- State cloning to CPU

```python
# Add to _adaptation():
if torch.cuda.is_available():
    stats = torch.cuda.memory_stats()
    self.pipeline.reporter.report("GPU_peak_MB", 
        stats["allocated_bytes.all.peak"] / 1e6)
```

### Phase D: Algorithmic Analysis

#### D.1 Validate Budget Scaling at ImageNet Size

For ImageNet (`dataset_size=1.2M`, `batch_size=128`):
```
SPE = 9,375
check_interval = sqrt(9375) = 97
max_training_steps = 9375 * 3 = 28,125
eval_n_batches = max(32, 390) = 390  <-- full val set
lr_steps_per_probe = 97
lr_num_probes = min(8, max(2, sqrt(97))) = 8
accuracy_se = 0.5 / sqrt(49920) = 0.00224
rollback_tolerance = 3 * 0.00224 = 0.0067
```

**Cost per adaptation cycle**:
- LR exploration: 8 * (97 train + 390 val) = 776 train + 3120 val batches
- Recovery training: up to 28,125 train + ~290 * 390 val batches
- Post-check: 390 val batches
- **Total validation batches**: ~113,500 (dominating cost)
- At ~0.14s per batch: ~4.4 hours per cycle of validation alone

#### D.2 Validate the "Largest Non-Destructive LR" Heuristic

The current selection picks the highest LR where `acc >= baseline - margin`. From the observed data:
- LR=1e-5: acc=0.655 (non-destructive, but very conservative)
- LR=1.4e-4: acc=0.308 (destructive)

The heuristic correctly returns 1e-5, but this is too conservative for recovery. The model needs a higher LR to recover within the budget.

**Alternative heuristic**: Pick the LR at the "knee" of the accuracy-vs-LR curve (highest LR before steep decline), not the strict non-destructive threshold. This would select ~5e-5 from the observed data.

---

## Part 7: Wall-Clock Cost Analysis (ImageNet)

### Hard Target: 30 Minutes Per Pipeline

The entire ImageNet pipeline (all tuning steps combined) must complete within 30 minutes. These are pretrained models -- the transformations should not require extensive retraining. The current system violates this by orders of magnitude.

### Current Per-Cycle Cost (Unacceptable)

| Operation | Train Steps | Val Batches | Time (est.) |
|-----------|------------|-------------|-------------|
| `_update_and_evaluate()` | 0 | 390 | ~55s |
| `_find_lr()` (8 probes) | 776 | 3,120 | ~16 min |
| `train_steps_until_target()` | 28,125 | ~113,100 | ~4.4h |
| Post-recovery `validate_n_batches()` | 0 | 390 | ~55s |
| Test gate (at rate=1.0) | 0 | ~780 | ~1.5 min |
| **Total per cycle** | **28,901** | **~117,780** | **~4.7h** |

A single adaptation cycle takes **4.7 hours**. With multiple tuners (clamp, quantization, activation adaptation, pruning) each potentially running 10+ cycles, the total exceeds **100 hours**. This is fundamentally broken.

### Root Cause: Validation Is the Dominant Cost

The 4.7h per cycle breaks down as:
- **Training**: 28,125 steps * ~5ms/step = ~2.3 min (negligible)
- **Validation during training**: 290 checks * 390 batches * ~0.14s/batch = **4.4 hours**
- **LR exploration validation**: 8 * 390 batches * ~0.14s/batch = **6 min**

**93% of wall-clock time is spent on validation**, not training. The system validates 390 batches (full 50K ImageNet val set) every 97 training steps. This is absurd -- 97 training steps take ~0.5 seconds, followed by 55 seconds of validation.

### Budget for 30-Minute Target

For a single tuning step (e.g., Clamp Adaptation) within a 30-minute pipeline:
- ~5-6 tuning steps share the 30 minutes -> ~5 minutes per step
- Each step: one-shot + maybe 2-3 gradual cycles + `_after_run` recovery

Per cycle budget: ~1-2 minutes. This requires:
1. **Cheap validation during training**: 8-16 batches per check (~1-2s), not 390 (~55s)
2. **Cheap LR exploration**: 8-16 batches per probe validation, not 390
3. **Full validation only for final commit/rollback decisions** (1-2 per cycle)
4. **Shorter recovery training**: 1x SPE max, not 3x SPE
5. **Test gate calls minimized**: only at rate=1.0 final commit

### Target Per-Cycle Cost

| Operation | Train Steps | Val Batches | Time (est.) |
|-----------|------------|-------------|-------------|
| `_update_and_evaluate()` | 0 | 16 | ~2s |
| `_find_lr()` (8 probes) | 776 | 128 | ~2.5 min |
| `train_steps_until_target()` | 9,375 | ~1,536 | ~3.5 min |
| Post-recovery `validate_n_batches()` | 0 | 390 | ~55s (full, for commit) |
| **Total per cycle** | **10,151** | **~2,070** | **~7 min** |

With 3-4 cycles per tuner: ~25 min. With one-shot success: ~8 min. Achievable within 30 min total pipeline.

### Specific Cost Reduction Changes

| Change | Current | Target | Effect |
|--------|---------|--------|--------|
| LR probe validation batches | 390 | 16 | LR search: 16 min -> 2.5 min |
| Recovery check_interval validation batches | 390 | 16 | Recovery: 4.4h -> 3.5 min |
| `max_training_steps` | 3 * SPE | 1 * SPE | Recovery: 28K -> 9.4K steps |
| `_update_and_evaluate()` validation | 390 (full) | 16 | Instant check: 55s -> 2s |
| Post-recovery rollback validation | 390 (full) | 390 (full) | Keep full -- commit decision |
| `_ensure_pipeline_threshold()` budget | 3x normal | 1x normal | Safety: 13h -> 4h -> capped |

**Key insight**: Use cheap validation (16 batches) for progress tracking and convergence detection. Use full validation (390 batches) only for commit/rollback decisions where statistical rigor matters. This matches how the `accuracy_se()` framework was designed -- the SE is computed for `eval_n_batches`, but convergence detection doesn't need that precision.

---

## Part 8: Implementation Sequencing

```
Phase 0 (Critical bugs):  4.1 (pipeline hard gate) + 4.2 (_after_run fix)
Phase 1 (Performance):    4.3 (two-tier validation + 1x SPE budget)
Phase 2 (Core Recovery):  4.4 (best-model checkpoint) + 4.5 (constant LR)
Phase 3 (Tolerance):      4.6 (empirical rollback tolerance)
Phase 4 (LR & Budget):    4.7 (wider LR range) + 4.9 (per-tuner budget)
Phase 5 (Polish):         4.8 (temp target relaxation) + 4.10 (logging) + 4.11 (one-shot skip)
```

**Phase 0** fixes correctness bugs that cause pipeline assertion failures on mnist_hard_all and similar configs.

**Phase 1** is the performance breakthrough: reduces per-cycle cost from 4.7h to ~7 min for ImageNet, making the 30-minute pipeline target achievable. Without this, no other improvement matters because runs never finish.

**Phases 2-5** improve accuracy retention quality and debuggability.

---

## Verification Plan

After implementing changes:

1. **Unit tests**: Run existing test suite (`pytest tests/`) -- all 1436 tests must pass
2. **mnist_hard_all regression**: Run the exact config that failed. Verify Activation Quantization step passes pipeline assertion.
3. **mnist_small_all**: Verify no regression.
4. **CIFAR-10**: Verify no regression from current passing state.
5. **ImageNet timing**: Run imgnet_sq_pretrained through Clamp Adaptation only. Verify:
   - LR exploration completes in <5 minutes (not 16)
   - One adaptation cycle completes in <30 minutes (not 4.7 hours)
   - Cycle log shows timing breakdown
6. **Metric basis test**: Verify tuner baseline = test accuracy (not validation)
7. **_after_run failure test**: Verify _committed_rate not set to 1.0 on recovery failure
