# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use the `RateScheduler` to progressively apply
specific transformations while maintaining accuracy.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `../unified_tuner.py` | `TunerBase`, `SmoothAdaptationTuner`, `_RECOVERY_PATIENCE` | `TunerBase`: shared infrastructure (pipeline, model, trainer, budget, target adjuster, LR finder with `anchor_lr`). `SmoothAdaptationTuner`: baseline calibration from `validate_n_batches` at rate 0.0, one-shot with test gate, rollback tolerance `pipeline_dt + 3*se`, all `min_improvement` from `accuracy_se()` |
| `perceptron_transform_tuner.py` | `PerceptronTransformTuner` | Extends `SmoothAdaptationTuner`; uses `PerceptronTransformTrainer`; stochastic mixing of previous/new perceptron transforms with a **per-cycle frozen mask** (a fresh Bernoulli mask at probability `rate` is drawn lazily per `(perceptron_id, param_name)` inside each `_mixed_transform(rate)` closure, then reused across every training step and validation within that probe -- see "Per-cycle frozen mask" below); delegates `_adaptation()` to base class (test gate, min_improvement, hooks); `_after_run()` forces rate=1.0 transform, recovery training, `_ensure_pipeline_threshold()` |
| `activation_adaptation_tuner.py` | `ActivationAdaptationTuner` | Gradually blends non-ReLU activations toward ReLU; `_after_run()` commits to LeakyGradReLU and caches metric via `trainer.test()`; includes commit guard: if post-commit accuracy falls below `target_adjuster.floor`, restores pre-commit state; `validate()` returns cached metric |
| `clamp_tuner.py` | `ClampTuner` | Introduces activation clamping progressively; validates `activation_scales`, logs diagnostics, probes saturation; caches final `trainer.test()` metric |
| `activation_shift_tuner.py` | `ActivationShiftTuner` | Extends `TunerBase` (not smooth adaptation); applies shift once, recovers with LR-search + step-training using `min_improvement=accuracy_se()` and `eval_n_batches`; caches final `trainer.test()` metric |
| `activation_quantization_tuner.py` | `ActivationQuantizationTuner` | Quantizes activations to Tq levels; extends `AdaptationRateTuner` (`quantization_rate`) |
| `normalization_aware_perceptron_quantization_tuner.py` | `NormalizationAwarePerceptronQuantizationTuner` | Quantizes weights with normalization awareness; extends `PerceptronTransformTuner` |
| `../orchestration/kd_blend_adaptation_tuner.py` | `KDBlendAdaptationTuner`, `BlendActivation`, `_KDClassificationLoss`, `_InstalledForward`, `CascadeForwardInstall` | **Shared base** for blend-ramp-with-KD adaptation: snapshots a frozen teacher, installs a `BlendActivation` (old→target by rate) on each perceptron's `base_activation`, ramps 0→1 with `α·CE + (1−α)·T²·KL` (T=3, α=0.3). **Genuinely gradual ramp contract**: `_skip_one_shot=True` (no jump-to-1.0 probe) and a small uniform rate-scheduler ladder (`_initial_ramp_step=0.125`, `_ramp_step_growth=1.0`, clamped to the budget's `min_step`) — the transformation advances and recovers DURING the ramp instead of relabelling recovery as stabilization (`test_kd_blend_gradual_ramp.py`). **Ramp is value-domain by default** (`_ramp_forward()` → `None`: the per-perceptron `BlendActivation` ramp — rate 0 == continuous teacher bit-exact, rate 1 == the pointwise on-chip composition); the genuine cross-layer dynamics are installed only at `_finalize()` via `_finalize_forward()`. The deployed forward is built by the shared **`_finalize_forward_for(model)`** seam (`_finalize_forward()` = `_finalize_forward_for(self.model)`); LIF and TTFS both override only `_finalize_forward_for`, and the genuine probe reuses it on a clone so probe ≡ deploy. `_finalize` records `_stabilization_refinds_lr = (fwd is not None)` so stabilization re-finds the LR on the swapped-in deployed forward (the ramp's cached LR was tuned on the proxy). Post-finalize, `_max_stabilization_rounds = 3` runs extra LR-restart stabilization passes while validation still improves. `_after_run` wraps ramp completion in `try/finally: _remove_forward()` and records `_finalize_cliff` (ramp@1 minus post-finalize — the proxy↔genuine gap) and `_phase_seconds`. `_InstalledForward(model, T)` is the picklable `model.forward` base; `CascadeForwardInstall` the install/remove mixin. Subclasses override `_make_target_activation`, `_blend_old_activation`, `_make_blend`, `_after_make_target`, `_wrap_encoding_input`, `_finalize_forward`, `_finalize`. |
| `lif_adaptation_tuner.py` | `LIFAdaptationTuner`, `LIFBlendActivation`, `_ChipAlignedNFForward` | `KDBlendAdaptationTuner` subclass: target = `LIFActivation`. **Ramp** is the value-domain blend (no instance forward). **Finalize:** customizes the shared base `_finalize` through ordered hooks — `_before_finalize_rebuild` sets `lif_active` (so the rebuilt activations subsume decorators), `_after_finalize_rebuild` applies cycle-accurate trains when `cycle_accurate_lif_forward`; the base then installs `_ChipAlignedNFForward` (`chip_aligned_segment_forward`) as the deployed `model.forward` and sets `_stabilization_refinds_lr`. LIF no longer copies the `_finalize` body. `_ChipAlignedNFForward` subclasses `LazyExecutorForward` (aliased `_InstalledForward`). Encoding-layer perceptrons get a `ChipInputQuantizer`. |
| `ttfs_cycle_adaptation_tuner.py` | `TTFSCycleAdaptationTuner`, `_SegmentSpikeForward`, `_BlendGenuineKDLoss` | `KDBlendAdaptationTuner` subclass for `ttfs_cycle_based`: target = `TTFSActivation`; sets `adaptation_manager.ttfs_active` so the kernel subsumes the clamp/quant/shift decorators. Both schedules ramp the value-domain blend. **cascaded** installs the genuine single-spike cascade (`_SegmentSpikeForward`) at `_finalize_forward` and keeps it (the committed metric, recovery, and every downstream step run the deployed single-spike dynamics; base `_finalize` sets `_stabilization_refinds_lr`). **synchronized** installs no forward (the class forward through the ramped blend *is* the deployed analytical staircase composition) and trains the wire contract's stage-input grid snap via a `TTFSInputGridQuantizer` STE on every **segment-entry** perceptron (`segment_entry_perceptrons`). **Rung-2 KD** (`ttfs_finetune_kd_against_rung2`, default off, synchronized-only): the KD teacher becomes `_Rung2TeacherFlow` (frozen snapshot IR-mapped, evaluated through the identity-mapped contract flow). The proxy↔genuine cascade gap (cascaded only) is the dominant cascaded-deployment accuracy residual — see `docs/fine_tuning_research_directions.md`. **Genuine annealed ramp** (`ttfs_genuine_annealed_ramp`, default off, cascaded only; `ttfs_ramp_alpha_min=0.5`/`ttfs_ramp_alpha_max=2.0`): the `_genuine_annealed_ramp` flag (set in `_configure`) trains through the genuine single-spike cascade for the WHOLE ramp instead of blending the value domain. `_make_axis` → `TTFSGenuineAxis` (anneals the spike surrogate alpha alongside the rate); `_make_blend` returns the bare `TTFSActivation` target as `base_activation` (no value-blend ReLU side); `_after_install_blend` runs `_finalize_rebuild` first so the segment policy finds genuine TTFS nodes; `_ramp_forward()` installs the same `_finalize_forward_for(self.model)` cascade forward for the whole ramp. Since alpha is backward-only (forward is exact `pre>0` Heaviside), rate=1 is bit-identical to the deployed cascade, so the finalize cliff is ~0 by construction. **Genuine teacher→cascade blend ramp** (`ttfs_genuine_blend_ramp`, default off, cascaded only; mutually exclusive with the annealed ramp — blend wins, annealed forced off): the `_genuine_blend_ramp` flag shares the bare-`TTFSActivation` target setup (`_genuine_bare_target_ramp`), then `_calibrate_to_teacher_distribution` runs `spiking.distribution_matching.match_activation_distributions(model, teacher, cal_x, T, quantile/bias_iters/eta from ttfs_distmatch_*)` on the deployed cascade (scale-aware [0,1] boundaries live-mutate the perceptron scale Parameters the bare nodes already reference, so no second rebuild — stats on `_distmatch_stats`/reported). `_make_axis` → `GenuineBlendAxis`; `_ramp_forward()` installs `BlendedGenuineForward(model, teacher, T, rate=0)` as the WHOLE-ramp forward (rate 0 = frozen teacher exactly, rate 1 = genuine cascade exactly). `_make_kd_loss` → `_BlendGenuineKDLoss` (base KD+CE on the blend output + `ttfs_genuine_blend_ce_alpha`·CE on the pure-genuine logits read via the tuner-owned `BlendedGenuineForward.genuine_logits`, resolved through a provider closure — not `model.__dict__` — so the term is decoupled from the install mechanism; the weight is a registered config key read once in `_configure`, and `_remove_forward` clears the owned reference so the term is skipped once the pure cascade is deployed). `_finalize_forward` deploys the PURE `_SegmentSpikeForward` (teacher dropped → cliff 0 by construction). The value-domain proxy ramp stays the default; both genuine ramps' accuracy non-regression vs the proxy baseline is an empirical gate pending a full run. **Fast genuine-blend ramp** (`ttfs_genuine_blend_fast`, default off, requires `ttfs_genuine_blend_ramp`; `ttfs_blend_fast_steps_per_rate=120`, `ttfs_blend_fast_rates=[0.5,0.75,0.9,0.97,1.0]`): FOLDED into the one orchestrator (no bespoke `run()` engine). `_configure` sets `_fixed_ladder_policy`/`_fixed_ladder_rates` (normalized to a trailing 1.0 so the ramp always finishes through the fast attempt, never the heavy `_continue_to_full_rate` controller), so the shared `_run_with_scheduler` builds a `fixed_ladder` RateScheduler policy that walks the rate list driving the overridden `_driver_attempt` → `_fast_rate_attempt` (the tuner's `run()` first resets the per-run fast scratch so a re-run rebuilds the optimizer + spanning cosine): ONE shared optimizer + spanning warmup(5%)/cosine LR over `len(rates)·steps_per_rate` steps (built once in `_ensure_fast_optimizer`), `steps_per_rate` training steps at each rate with loss `CE((1-R)·teacher + R·genuine) + ttfs_genuine_blend_ce_alpha·CE(genuine)` (blend = `model(x)` through the owned `BlendedGenuineForward`; genuine = its `genuine_logits`), then a post-rate eval recorded as ONE `commit` per rate in the DecisionTrace (`_record_fast_cycle`). `_stabilization_budget` returns 0 (the cliff is ~0 by construction). The shared `_finalize_run`/`_after_run` removes the blend, deploys the pure `_SegmentSpikeForward`, and measures the finalize cliff. So the fast path INHERITS the trace + finalize observability through the same seam every tuner uses (`_fast_blend_path=True`, `_fast_optimizer_steps` = the exact step count, timing in `_phase_seconds["fast_blend"]`). NO per-cycle rollback / recovery-to-target / LR-find / stabilization. Reproduces `generated/_genuine_ab/full_ramp.py` (genuine 0.41 → 0.9355). |
| `noise_tuner.py` | `NoiseTuner` | Introduces training noise; extends `AdaptationRateTuner` (`noise_rate`); no pipeline step wired by default |
| `pruning_tuner.py` | `PruningTuner` | Gradually zeros least-significant rows/columns; recomputes importance at each cycle; overrides `_before_cycle`, `_recovery_training_hooks`, `_after_run`, `_update_and_evaluate`; uses base-class `_find_lr` (anchored LR search); `_force_to_full_rate` drives pruning from committed rate to 1.0 in gradual increments with `min_improvement=accuracy_se()/2`; uses base-class `_adaptation` with LR search.  **Boundary-IR caching**: `_boundary_exemption_layers` (in `pruning_tuner_masks.py`) memoises the boundary-policy result on the tuner instance.  The IR build behind it is O(model) — ~27 s for ViT-B/16 — and `_get_masks` fires twice per cycle (`_apply_masks` + `register_recovery_hooks`), so the per-tuner cache shaves ~10 minutes off a 10-cycle ViT pruning step.  Topology is invariant during pruning; call `_invalidate_boundary_cache(tuner)` if it ever isn't. |

## Tuner Hierarchy

```
TunerBase
├── SmoothAdaptationTuner
│   ├── AdaptationRateTuner (`_apply_rate`)
│   │   ├── ActivationQuantizationTuner
│   │   └── NoiseTuner
│   ├── ActivationAdaptationTuner
│   ├── ClampTuner
│   ├── PruningTuner (overrides _before_cycle, _recovery_training_hooks, _after_run)
│   └── PerceptronTransformTuner (PerceptronTransformTrainer)
│       └── NormalizationAwarePerceptronQuantizationTuner
│   └── KDBlendAdaptationTuner (teacher+KD blend ramp; in ../orchestration/)
│       ├── LIFAdaptationTuner (target=LIFActivation; cycle-accurate forward)
│       └── TTFSCycleAdaptationTuner (target=TTFSActivation; schedule-aware ttfs_cycle_based)
└── ActivationShiftTuner (one-shot, not smooth adaptation)
```

## Dependencies

- **Internal**: `tuning` (adaptation framework), `model_training` (trainers), `data_handling` (`DataLoaderFactory`), `models` (layers, `models.spiking.training.blended_genuine_forward`), `spiking` (`distribution_matching.match_activation_distributions` — genuine blend ramp), `mapping.ir` (`NeuralCore`), `transformations`.
- **External**: `torch`, `numpy`, `copy`.

## Dependents

- `pipelining.pipeline_steps` imports specific tuners for each tuning step.

## Exported API (\_\_init\_\_.py)

`TunerBase`, `SmoothAdaptationTuner`, `ClampTuner`, `ActivationAdaptationTuner`,
`ActivationQuantizationTuner`, `ActivationShiftTuner`,
`NormalizationAwarePerceptronQuantizationTuner`, `LIFAdaptationTuner`,
`TTFSCycleAdaptationTuner`, `NoiseTuner`, `PerceptronTransformTuner`, `PruningTuner`.

## Per-cycle frozen mask (PerceptronTransformTuner)

`PerceptronTransformTuner._mixed_transform(rate)` now returns a closure
that captures a private `mask_cache` dictionary. The first time the
closure is invoked for a given perceptron slot the Bernoulli mask at
probability `rate` is drawn for each parameter (keyed by
`(id(perceptron), param_name)`) and stored in the cache. Subsequent
invocations of the same closure reuse the cached mask.

Why this matters:

- `PerceptronTransformTrainer._perceptron_slots` keeps a persistent
  `temp_p` object per perceptron and refreshes its parameters from
  `aux_model` before every training step. Both training (per step) and
  validation (inside `_update_and_evaluate`) dispatch through the
  *same* `trainer.perceptron_transformation` callable, i.e. the *same*
  `_mixed_transform(rate)` closure. With the cache, all of those calls
  see the *same* stochastic realisation of the prev/new mix.
- Without the cache (legacy behaviour) the random mask was redrawn on
  every call. Combined with the now rate-aware
  `NormalizationAwarePerceptronQuantization`, this meant training
  chased a moving-target loss surface and validation measured a
  different realisation than training had just optimised -- cycles
  rolled back even at tiny rates and the committed rate could not
  progress past ~0.
- `_update_and_evaluate(rate)` creates a fresh closure (and therefore a
  fresh cache) per probe, so the mask is still stochastic across
  probes -- preserving the intended regularisation flavour while
  eliminating the within-cycle instability.

Endpoint behaviour is unchanged: `rate == 0.0` always produces an
all-False mask (identity), `rate == 1.0` always produces an all-True
mask (fully-transformed output).
