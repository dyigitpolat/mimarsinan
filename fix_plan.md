# Fix Plan: Accuracy Degradation & Pipeline Errors After Refactoring

## Context

A major refactoring removed Supermodel/InputCQ, replaced epoch-based tuning with step-based tuning, replaced the old LR search with an exponential sweep, and unified tuner base classes. This introduced several interrelated bugs causing: (1) ~10% pruning drops on MNIST, (2) 0% accuracy in weight quantization, (3) AssertionError in simulation step, and (4) accuracy discrepancy between Torch Mapping and downstream steps.

---

## Fixes (in implementation order)

### Fix 1: SimulationStep 'model' entry never accessed
**File:** `src/mimarsinan/pipelining/pipeline_steps/simulation_step.py`
**Line 8:** Remove `"model"` from `requires` list.

```python
# Before:
requires = ["hard_core_mapping", "scaled_simulation_length", "model"]
# After:
requires = ["hard_core_mapping", "scaled_simulation_length"]
```

**Why:** `process()` never calls `self.get_entry("model")`, so `PipelineStep.run()` (pipeline_step.py:16) raises AssertionError because the required entry was not accessed.

---

### Fix 2: Dual target adjuster in PerceptronTransformTuner (0% weight quantization root cause)
**File:** `src/mimarsinan/tuning/tuners/perceptron_transform_tuner.py`
**Lines 92-101:** Pass `target_adjuster` and `update_target_after_cycle=False` to SmartSmoothAdaptation.

```python
adapter = SmartSmoothAdaptation(
    self._adaptation,
    clone_state,
    restore_state,
    evaluate_model,
    interpolators=[BasicInterpolation(0.0, 1.0)],
    target_metric=self._get_target(),
    initial_tolerance_fn=initial_tol_fn,
    min_step=min_step_for_smooth_adaptation(self.pipeline, self._budget),
    target_adjuster=self.target_adjuster,       # ADD
    update_target_after_cycle=False,             # ADD
)
```

**Why:** Without these, SmartSmoothAdaptation creates its OWN internal `AdaptationTargetAdjuster(original_target, decay=0.999)`. The step search in `_find_step_size()` uses this internal adjuster (nearly stationary target), while `_adaptation()` updates the tuner's external adjuster. The step search never sees target relaxation, takes only tiny steps, exhausts the training budget, and the model converges to 0% accuracy.

---

### Fix 3: Same dual adjuster bug in PruningTuner (~10% pruning drop root cause)
**File:** `src/mimarsinan/tuning/tuners/pruning_tuner.py`
**Lines 185-195:** Add `target_adjuster` and `update_target_after_cycle=False`.

```python
adapter = SmartSmoothAdaptation(
    _adaptation,
    lambda: copy.deepcopy(self.model.state_dict()),
    lambda state: self.model.load_state_dict(state),
    _update_and_eval,
    [BasicInterpolation(0.0, 1.0)],
    self.target_adjuster.get_target(),
    before_cycle=before_cycle,
    initial_tolerance_fn=initial_tol_fn,
    min_step=min_step_for_smooth_adaptation(self.pipeline, self._budget),
    target_adjuster=self.target_adjuster,       # ADD
    update_target_after_cycle=False,             # ADD
)
```

**Why:** Same as Fix 2. The pruning tuner's `_adaptation` function updates `self.target_adjuster` but the step search uses an internal separate adjuster.

---

### Fix 4: Step-size initialization halved (affects all adaptation)
**File:** `src/mimarsinan/tuning/smart_smooth_adaptation.py`
**Line 43:** Revert to old `(1-t)*2` initialization.

```python
# Before:
step_size = 1.0 - t
# After:
step_size = (1.0 - t) * 2
```

**Why:** The while loop's first operation is `step_size /= 2`. The old code started at `2*(1-t)` so the first probe tests the full remaining range `(1-t)`. The new code starts at `(1-t)` so the first probe only tests half the remaining range `(1-t)/2`, making adaptation systematically more conservative, requiring more cycles, and wasting training budget on unnecessary intermediate steps.

---

### Fix 5: Self-terminating training with convergence detection
**Files:** `src/mimarsinan/tuning/tuning_budget.py`, `src/mimarsinan/model_training/basic_trainer.py`

**Problem:** The sqrt formula gives fixed, insufficient step counts (21 for MNIST, 100 for ImageNet). Any static formula either undertunes small datasets or overtunes large ones. The right budget depends on how hard recovery is -- something we can't predict upfront.

**Key insight:** Instead of predicting the right budget, **let training decide when to stop**. Set a generous max budget (1 epoch) and add convergence detection: stop when the model either reaches the target OR stops improving (patience exhausted). The `sqrt(spe)` value becomes the check interval (how often to evaluate progress), not the budget itself.

| Dataset  | SPE   | Check interval | Easy (2 checks) | Hard (patience=3) | Max  |
|----------|-------|----------------|------------------|--------------------|------|
| MNIST    | 468   | 21 steps       | 42 steps         | ~210 steps         | 468  |
| CIFAR10  | 390   | 19 steps       | 38 steps         | ~190 steps         | 390  |
| ImageNet | 10009 | 100 steps      | 200 steps        | ~1000 steps        | 10009|

**Step 5a: Reshape TuningBudget** (`tuning_budget.py`)

```python
@dataclass
class TuningBudget:
    max_training_steps: int   # generous max = steps_per_epoch
    check_interval: int       # progress check frequency = sqrt(spe)
    validation_steps: int     # batches per validation check
    lr_search_steps: int      # LR sweep steps

    @staticmethod
    def from_dataset(dataset_size, batch_size, budget_scale=1.0):
        bs = max(1, int(batch_size))
        steps_per_epoch = max(1, int(dataset_size) // bs)
        check_interval = max(1, int(math.sqrt(float(steps_per_epoch))))
        max_training_steps = max(1, int(float(steps_per_epoch) * float(budget_scale)))
        validation_steps = max(1, min(32, check_interval))
        lr_search_steps = max(50, min(500, check_interval * 4))
        return TuningBudget(max_training_steps, check_interval, validation_steps, lr_search_steps)
```

**Step 5b: Add convergence-aware early stopping to `train_steps_until_target`** (`basic_trainer.py`)

Replace the current every-step validation with periodic checks + patience:

```python
def train_steps_until_target(
    self, lr, max_steps, target_accuracy, warmup_steps=0,
    *, validation_n_batches=1, check_interval=1, patience=3,
):
    optimizer, scheduler, scaler = self._get_optimizer_and_scheduler_steps(lr, max_steps)
    # ... warmup setup ...
    
    best_acc = 0.0
    stale_checks = 0
    n_val = max(1, int(validation_n_batches))
    interval = max(1, int(check_interval))
    
    for step_idx in range(total):
        # --- training step (unchanged) ---
        x, y = self._next_train_batch()
        self._optimize(x, y, optimizer, scaler)
        scheduler.step()
        
        # --- periodic progress check ---
        if (step_idx + 1) % interval == 0 or step_idx == total - 1:
            acc = self.validate_n_batches(n_val)
            if acc >= target_accuracy:
                # Target reached: do 2 extra steps and break
                for _ in range(2):
                    x, y = self._next_train_batch()
                    self._optimize(x, y, optimizer, scaler)
                    scheduler.step()
                break
            if acc > best_acc + 1e-3:
                best_acc = acc
                stale_checks = 0
            else:
                stale_checks += 1
                if stale_checks >= patience:
                    break  # converged, no further improvement
    
    self.test()
    return self.validate_n_batches(n_val)
```

**Step 5c: Update callers** (unified_tuner.py, perceptron_transform_tuner.py, pruning_tuner.py)

Replace `train_steps_until_target(lr, self.training_steps, target, 0, validation_n_batches=self.validation_steps)` with:

```python
self._get_trainer().train_steps_until_target(
    lr,
    self._budget.max_training_steps,
    self._get_target(),
    0,
    validation_n_batches=self.validation_steps,
    check_interval=self._budget.check_interval,
    patience=3,
)
```

**Why this is the right design:**
- **Self-terminating**: training stops when it should, not when an arbitrary budget runs out
- **No prediction needed**: generous max (1 epoch) is a safety net, not a target
- **Universally efficient**: easy recovery stops in 2 checks (~42 steps MNIST, ~200 steps ImageNet), hard recovery uses what it needs, convergence detection prevents waste
- **Clean semantics**: `check_interval = sqrt(spe)` = "natural granularity for progress evaluation", `patience = 3` = standard convergence criterion in ML
- `budget_scale` scales the max budget (generous already, user can tighten if needed)

---

### Fix 6: (Subsumed by Fix 5)

The per-step validation issue is resolved by Fix 5's `check_interval` parameter -- validation now happens every `sqrt(spe)` steps instead of every step. No additional changes needed.

---

### Fix 5d: Update `min_step_for_smooth_adaptation`
Since `TuningBudget.training_steps` is renamed to `check_interval`, update the formula:
```python
def min_step_for_smooth_adaptation(pipeline, budget: TuningBudget) -> float:
    m = max_total_training_steps(pipeline)
    return max(0.001, float(budget.check_interval) / float(m))
```
This gives MNIST: 21/4680 ≈ 0.004, ImageNet: 100/100090 ≈ 0.001. Both reasonable.

Also update all callers that access `budget.training_steps` to use the new field names (`budget.max_training_steps`, `budget.check_interval`).

---

## Non-issues (confirmed correct)

- **`update_target_after_cycle=False`** in unified tuners: This is correct. The old code double-updated the target (once in `_adaptation()`, once in `adapt_smoothly()`). The new code updates once in `_adaptation()` which is sufficient.
- **Dynamic target decay from validation size**: Reasonable improvement. Once the dual-adjuster bug is fixed, this becomes a single coherent adjuster.
- **LR range finder**: Non-destructive exponential sweep is sound. Old heuristic (`pipeline_lr/200`) was fragile.
- **Accuracy discrepancy (68 vs 62)**: Likely explained by pruning drops (Fix 3) or single-batch validation noise. Should resolve after Fixes 3-5.

---

## Verification

After all fixes, run MNIST phased pipeline end-to-end:
```bash
cd /home/yigit/repos/mimarsinan && source env/bin/activate && python run.py examples/mnist_run.json
```

Check:
- Pruning adaptation: accuracy drop under 3% (was ~10%)
- Weight quantization: nonzero accuracy (was 0%)
- Simulation step: no AssertionError
- Overall pipeline completes successfully
