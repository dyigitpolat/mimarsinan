# tuning/tuners/ -- Concrete Tuner Implementations

Specialized tuners that use `SmartSmoothAdaptation` to progressively apply
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
| `../orchestration/kd_blend_adaptation_tuner.py` | `KDBlendAdaptationTuner`, `BlendActivation`, `_KDClassificationLoss` | **Shared base** for blend-ramp-with-KD adaptation: snapshots a frozen teacher, installs a `BlendActivation` (old→target by rate) on each perceptron's `base_activation`, ramps 0→1 with `α·CE + (1−α)·T²·KL` (T=3, α=0.3). Subclasses override `_make_target_activation`, `_blend_old_activation`, `_make_blend`, `_after_make_target`, `_wrap_encoding_input`, `_after_install_blend`, `_finalize`; `_append_encoding_input_module` is the shared encoding-input wire-op wrap (Identity→module, else Sequential). |
| `lif_adaptation_tuner.py` | `LIFAdaptationTuner`, `LIFBlendActivation`, `_CycleAccurateForward` | `KDBlendAdaptationTuner` subclass: target = `LIFActivation`. **Cycle-accurate:** when `cycle_accurate_lif_forward`, installs picklable `_CycleAccurateForward` on `model.forward` calling `run_cycle_accurate` with `simulation_steps` as `T`. **Symmetric patch/unpatch:** `_install_cycle_accurate_forward` asserts no double-patch; `_after_run` always removes the instance `model.forward` in a `try/finally`, regardless of the cycle-accurate flag, so downstream pipeline stages see the pristine class forward. Encoding-layer perceptrons get a `ChipInputQuantizer`. `LIFBlendActivation` subclasses `BlendActivation` (keeps `.lif_activation`). |
| `ttfs_cycle_adaptation_tuner.py` | `TTFSCycleAdaptationTuner`, `_SegmentSpikeForward` | `KDBlendAdaptationTuner` subclass for `ttfs_cycle_based`: target = `TTFSActivation`; sets `adaptation_manager.ttfs_active` so the kernel subsumes the clamp/quant/shift decorators. **Schedule-aware** (`ttfs_cycle_schedule`): **cascaded** trains pure spike (rate pinned 1.0) through the picklable `_SegmentSpikeForward` (`TTFSSegmentForward` cascade walk) and `_finalize` re-installs it so the committed metric and every downstream step run the deployed single-spike dynamics; **synchronized** installs no instance forward (the class forward through the ramped blend *is* the deployed analytical staircase composition), ramps the blend naturally, and trains through the wire contract's stage-input grid snap via a `TTFSInputGridQuantizer` STE on every **segment-entry** perceptron (`segment_entry_perceptrons` — the first on-chip core per segment; NOT `is_encoding_layer`, which is inert under offload and the wrong seam under subsume). With the entry snaps, synchronized NF↔contract is per-neuron bit-exact. **Rung-2 KD** (`ttfs_finetune_kd_against_rung2`, default off, synchronized-only): the KD teacher becomes `_Rung2TeacherFlow` — the frozen pre-step snapshot IR-mapped and evaluated through the identity-mapped contract flow (÷T, per-output node scales) — distilling toward the mapping-level wire semantics at the cost of one IR map + a contract-flow eval per KD batch. |
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

- **Internal**: `tuning` (adaptation framework), `model_training` (trainers), `data_handling` (`DataLoaderFactory`), `models` (layers), `mapping.ir` (`NeuralCore`), `transformations`.
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
