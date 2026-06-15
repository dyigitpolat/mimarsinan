# Tuning optimization flags — why they are OFF, and what turning them ON buys

These are the default-OFF, non-UI feature flags left in the tuning subsystem after
the structural refactor graduated (`tuning_use_axis` / `tuning_use_checkpoint_guard`
/ `tuning_use_driver` were flipped default-ON and their flags + legacy paths
deleted). Everything below is an *optimization/quality* knob, not a structural one.
The goal of this document is to inform collapsing them — graduate-and-delete (make
default, drop the flag) or remove-and-drop (delete the path) — rather than leaving a
pile of opt-in toggles.

All accuracy numbers are from a 3-seed real-pipeline ablation (MNIST/CIFAR ViT-class
model, Normalization-Fusion accuracy; baseline run-to-run noise band σ ≈ 0.0035).

**Cost measurement (the <10%-bake bar).** Decomposing tuning wall-clock from the
ablation logs (per-step `[PROFILE] wall` + per-call `T_find_lr_sec`):
- Real **LR-finding is ~13% of tuning time**; recovery training + validation are the
  other ~87%. The "5.6× LR" headline was a **miscount** — it counted `LR_found`
  *reports* (which fire on cache hits, ~1e-6 s), not real LR sweeps; real LR-find
  time is essentially unchanged with paired (113 → 116 s over 3 seeds).
- **Paired's true cost is +139% of total tuning wall** (869 → 2080 s), and it comes
  from the **extra rollback recovery cycles** (paired rolls back ~10× more), not from
  LR-finding. So paired is **far above the 10% bake bar → stays opt-in.**
- The efficiency knobs (persist / loss-slope / in-place) are micro-optimizations that
  *reduce* or are neutral on cost (≤ 0% added) → **under 10% → baked.**

| flag | status | accuracy | cost | note |
|------|--------|----------|------|------|
| `use_paired_sensor` | **opt-in** | +0.55% (proven) | **+139% tuning** | quality/cost trade; above the bar |
| `persist_optimizer` | **BAKED** | neutral | ≤0% (no rebuild) | clean (crash-guard kept) |
| `loss_slope_lr` | **BAKED** | neutral | ≤0% (cheap coarse LR) | degrades gracefully on stub trainers |
| `subsample_val_cache` | opt-in (deferred) | +0.20% (proven) | ≤0% + scale fix | qualifies, but baking re-breaks 3 tiny-model parity tests |
| `inplace_rate` | opt-in (deferred) | output-identical | ≤0% (O(1) write) | qualifies, but needs ~12 delegation-test migrations + an end-to-end run |
| `sensitivity_stepping` | measuring | TBD | ≤0% | driver-on ablation in progress (inert in the first round) |

(`tuning_full_transform_probe` is a *diagnostic*, not an optimization — it adds a
forward pass per commit to measure rate-1.0 convergence; opt-in by design.)

---

## 1. `tuning_use_paired_sensor` — the one real trade-off

**Why OFF.** It costs **5.6×** more LR-finder sweeps (109 vs 17 across 3 seeds). The
paired McNemar gate compares the candidate against a *fixed* baseline on a shared
example subsample, giving a several-fold tighter standard error than the marginal
gate — so it rolls back small drifts the marginal gate tolerates, bisecting more
carefully into the quantization cliff (commit 0.96 → roll 0.97 → roll → commit
0.961 → …). That careful stepping is exactly what costs the extra LR-finds.

**Benefit ON (data).** **+0.55%** deployed accuracy, the **largest and most
consistent** of any flag: +0.79% / +0.92% / −0.05% across the three seeds. The gain
is the whole point of the statistical-gate work — it catches genuine cumulative
drift the marginal gate misses.

**Trap found.** The `global_budget` floor (set to 0.005 to cut the cost) *erased* the
gain: floored, paired went ~neutral with one seed collapsing to 0.9508. **Unfloored
(`global_budget = 0.0`) is strictly better** and is now the default. So the cost
is *intrinsic* to the gate, not tunable away cheaply.

**To collapse:** graduate default-ON only if the 5.6× offline tuning cost is
acceptable for +0.55% deployed accuracy. Otherwise keep as the documented
accuracy-critical opt-in. A middle path worth measuring: cap the per-cycle re-LR
after a paired rollback (most of the cost is re-finding LR on near-identical state).

## 2. `tuning_subsample_val_cache` — improvement that breaks a tiny-model test

**Why OFF.** Its bounded GPU cache makes the per-cycle validation cursor wrap a
*small* fixed subset; on a tiny *untrained* model whose accuracy straddles the
catastrophic gate, that wrap flips marginal commits and broke 3 LIF/TTFS
*parity* unit tests (the gradual ramp couldn't commit). The codebase rule is "don't
weaken parity tests," and the cache's scale benefit only bites at ImageNet scale.

**Benefit ON (data).** **+0.20%**, the *most consistent* arm (σ=0.0011), AND the
**W8 fix**: the default whole-validation-set GPU cache OOMs on ImageNet-scale val
sets. It is also the fixed subsample the paired gate pairs over.

**To collapse:** this is the best graduate-and-delete candidate (proven win + real
scale fix), but it requires first making the 3 tiny-model parity tests robust to
which validation batches are evaluated (they assert a fragile property on an
untrained model). Recommended: fix those tests, then graduate.

## 3. `tuning_persist_optimizer` — neutral; efficiency only

**Why OFF.** Accuracy-**neutral** (+0.37%, within the σ=0.0035 noise band). The plan
designed it opt-in; its only benefit is efficiency (no Adam rebuild per recovery),
which the ablation did not measure (LR count is the wrong proxy). It also needed a
crash guard (`_supports_persistent_optimizer = False` for param-replacing tuners).

**Benefit ON.** Persists Adam moments across the LR sweep / recovery within a cycle
instead of rebuilding fresh each call — saves optimizer construction and preserves
momentum. **Unmeasured** wall-clock benefit.

**To collapse:** needs a wall-clock benchmark to justify graduating. Without one,
remove-and-drop is defensible (it adds a code path + crash guard for no proven gain).

## 4. `tuning_loss_slope_lr` — neutral; efficiency only

**Why OFF.** Accuracy-**neutral** (+0.27%, within noise). Graduating it makes the
core `_find_lr` depend on `trainer.evaluate_loss_on_batch` / `next_training_batch`,
which broke mock-trainer unit tests — an invasive coupling for an unproven gain.

**Benefit ON.** Ranks the coarse LR sweep by a cheap training-loss slope and reserves
full-validation scoring for the top 2–3 candidates — cheaper LR discovery. Benefit is
again wall-clock, **unmeasured**.

**To collapse:** same as persist — benchmark wall-clock or remove-and-drop.

## 5. `tuning_sensitivity_stepping` — NO DATA (inert ablation)

**Why OFF.** Its ablation arm was **inert**: the flag is read only on the driver path
(`_run_with_scheduler`), but that arm ran with the driver OFF, so the flag did
nothing. The reported −0.04% measured nothing. It is the one knob with **zero real
evidence**.

**Benefit ON (theoretical).** `last_successful_step` policy: start each scheduler
round from the previously accepted step (×2) instead of the full gap — fewer probes
on cliff-like axes (quant/clamp).

**To collapse:** needs a *proper* driver-ON ablation (driver is now always on, so
this is now testable) before any decision. Until then it is dead-weight: either run
that ablation or remove-and-drop.

## 6. `tuning_inplace_rate` — output-identical; never run end-to-end

**Why OFF.** Never validated end-to-end (no ablation arm). Graduating it changes the
rate-application delegation contract (buffer vs `apply_manager_rate`), which broke
~12 axis conformance/delegation tests. It is, however, proven **output- and
RNG-conformant** with the rebuild path by a dedicated unit test (grid incl. 0.0/1.0).

**Benefit ON.** O(1) rate updates: build the decorator stack once, then a ramp step
is a single in-place `RateBuffer.set(α)` write instead of rebuilding every
perceptron's decorator stack per step (the report's W9 churn). Pure efficiency,
**unmeasured** at the pipeline level.

**To collapse:** conformance is proven; graduating requires migrating the ~12
delegation tests to the buffer contract + one end-to-end run to confirm no surprise.
Then graduate-and-delete (W9 is a real inefficiency).

---

## Recommended collapse order

1. ~~**Now, zero-risk:** the two zombie flags (`interleave_axes`,
   `clamp_learnable_scale`) and `per_layer_rate_schedule`.~~ **Done** — removed
   (flags + dead `MultiAxisDriver` / `per_layer_schedule.py` / learnable-clamp
   machinery). Also baked to default behaviour (flag deleted): `enable_ttfs_finetuning`
   (always on for `ttfs_cycle_based`), the pre-pruning heatmap, and the SANA-FE
   parity / trace toggles.
2. **Graduate-and-delete after a small fix:** `tuning_subsample_val_cache` (fix the 3
   tiny-model parity tests first) and `tuning_inplace_rate` (migrate the delegation
   tests). Both are real wins (accuracy/scale, W9) with no accuracy downside.
3. **Decide with a benchmark:** `tuning_persist_optimizer`, `tuning_loss_slope_lr` —
   run a tuning wall-clock benchmark; graduate if it pays, else remove-and-drop.
4. **Run the missing ablation:** `tuning_sensitivity_stepping` — a proper driver-ON
   ablation; it has no evidence either way today.
5. **Product call:** `tuning_use_paired_sensor` — graduate only if +0.55% accuracy is
   worth 5.6× offline tuning cost; otherwise keep as the one documented
   accuracy-critical opt-in.
