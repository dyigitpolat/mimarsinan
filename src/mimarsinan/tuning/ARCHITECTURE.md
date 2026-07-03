# tuning/ — Training-aware progressive adaptation of models toward hardware constraints

After the ANN is built and trained, the deployment pipeline uses this module to
apply hardware-motivated transformations (clamping, activation/weight
quantization, noise, ANN→SNN blend, LIF/TTFS conversion, pruning) gradually
while recovering accuracy at each step. The central abstractions are the
`SmoothAdaptationTuner` control loop and its services (`orchestration/`), the
rate-driven `AdaptationAxis` objects that know how to apply a transformation at
a fractional rate (`axes/`), and the concrete per-transformation tuners
(`tuners/`). Cross-cutting helpers at this level — target adjustment, LR
search, rate application, decision tracing — are shared by all tuner families.

## Key files
| File | Purpose |
|---|---|
| `adaptation_rate_tuner.py` | `AdaptationRateTuner`: `SmoothAdaptationTuner` driving one `AdaptationManager` rate field through a seeded `ManagerRateAxis`; base for the clamp/act-quant/noise family. |
| `adaptation_target_adjuster.py` | `AdaptationTargetAdjuster`: proportional target decay/growth with validation-set-sized decay and a `1 - degradation_tolerance` floor. |
| `forward_install.py` | `LazyExecutorForward` (picklable lazy cross-layer NF `model.forward` override) and `CascadeForwardInstall` (single-owner install/remove mixin). |
| `learning_rate_explorer.py` | `LRRangeFinder`: exponential LR sweep selecting the largest non-destructive LR, with optional coarse loss-slope pre-scoring; `find_lr_range_for_trainer` derives probe parameters from the `TuningBudget`. |
| `perceptron_rate.py` | SSOT for applying a rate model-wide: `rebuild_activations`, `apply_manager_rate`, `set_blend_rate`, `set_surrogate_alpha`. |
| `shift_calculation.py` | `calculate_activation_shift`: activation-space shift for quantization alignment. |
| `teacher.py` | `snapshot_frozen_teacher` / `freeze_module`: eval-mode, gradient-frozen deepcopy of a model for KD recovery (deepcopy on CPU). |
| `trace.py` | `DecisionRecord` / `DecisionTrace`: JSON-round-trippable per-cycle decision trace; iterates as legacy `_cycle_log` dicts and backs the golden-trace tests. |
| `axes/` | `AdaptationAxis` contract plus concrete axes (manager-rate family, blend/LIF/TTFS, perceptron-transform/NAPQ, pruning, activation shift); each delegates math to `transformations` and rate application to `perceptron_rate`. |
| `orchestration/` | The smooth-adaptation machinery: `TunerBase`/`SmoothAdaptationTuner` (cycle/run mixins), `AdaptationManager` + factory, `TuningBudget`, `ConversionPolicy` SSOT, temporal allocation, rate scheduler / recovery engine / acceptance sensor / checkpoint guard, optimization driver + fast ladder (with the flag-gated measurement-only `mbh_ledger`), blend-ramp strategy, KD-blend tuner, and the uniform `RateTunerSeam`. |
| `tuners/` | Concrete tuners: clamp, activation adaptation/quantization/shift, normalization-aware weight quantization, noise, LIF, TTFS cycle, perceptron transform, and `pruning/`. |

## Dependencies
- `models` — activation types (`TTFSActivation`, `LeakyGradReLU`, `make_activation`), decorator layers (`RandomMaskAdjustmentStrategy`, `RateBuffer`, `NoisyDropout`, `SavedTensorDecorator`), and the blended genuine spiking forward the KD-blend tuner installs.
- `model_training` — `BasicTrainer`, the perceptron-transform trainer, and `build_recipe`/`build_optimizer` for recovery training.
- `transformations` — normalization-aware perceptron quantization, `PerceptronTransformer`, pruning mask application and activation-stat collection.
- `data_handling` — `DataLoaderFactory`/`DataProvider` for validation-set sizes that derive budgets and decay factors.
- `mapping` — pruning boundary policy for the pruning tuner.
- `common` — `best_effort` error containment; `env` for the `MIMARSINAN_MBH_LEDGER` flag.

## Dependents
- `pipelining` — pipeline steps construct the tuners, the adaptation-manager factory, and `tuning_budget_from_pipeline`; `deployment_plan` consumes `temporal_allocation`, `ConversionPolicy`, the optimization driver, and the calibration pipeline.
- `config_schema` — `temporal_allocation` for validation; `ConversionPolicy` for deployment derivation.
- `chip_simulation` — calibration-pipeline constants in the deployment contract.
- `model_training` — LR-explorer helpers in `basic_trainer_steps`.
- `models` — `LazyExecutorForward` for the blended genuine forward.
- `spiking` — `LIFBlendActivation` from the LIF adaptation tuner.
- `mapping` — `calculate_activation_shift` for bias compensation.
- `gui` — `S_ALLOCATION_MODES` in the wizard schema.

## Exported API
- `AdaptationManager` — per-perceptron decorator-rate host (re-exported from `orchestration/`).
- `LRRangeFinder` — the LR range search.
- `calculate_activation_shift` — quantization shift amount.
