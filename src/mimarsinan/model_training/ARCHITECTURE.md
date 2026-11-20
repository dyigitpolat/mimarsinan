# model_training/ — Trainers, training recipes, losses, and pretrained-weight loading

Provides the training machinery the deployment pipeline uses to pretrain models and
to recover accuracy after each adaptation/tuning stage. The central abstraction is
`BasicTrainer` — a facade owning the model, device, data loaders, and loss, whose
epoch/step/eval/subsample APIs are implemented in the `basic_trainer_*` companion
modules. Transform trainers subclass it to optimize an auxiliary full-precision model
while the forward pass runs through a transformed copy (quantization-aware training),
and `TrainingRecipe` is a declarative optimizer/scheduler spec that both the epoch
paths and the opt-in recipe-driven step recovery consume.

## Key files
| File | Purpose |
|---|---|
| `basic_trainer.py` | `BasicTrainer`: loaders/iterators lifecycle, optimizer+scheduler construction (legacy Adam path or recipe-driven; every optimizer and the gradient clip read one set, `_params_to_optimize`, the parameters the trainer's update writes to the deployed model, U-TGAUGE T1-C), `validation_context` metric tagging, pickling; delegates epoch/step/eval/subsample APIs to the companion modules. |
| `basic_trainer_epochs.py` | Epoch-based training: `train_validation_epochs` (fixed epochs, validate each) and `train_until_target_accuracy` (early exit + final `test()`). |
| `basic_trainer_steps.py` | Step-based training: `train_n_steps`, `train_one_step` (post-update loss probing), and `train_steps_until_target` — convergence checks with patience, best-state rollback, optional plateau LR-reduction ladder, reusable external optimizer, `return_steps`/`cosine_decay` modes; `final_validation=False` skips the trailing eval for callers that re-measure with their own basis (A4 consolidation). |
| `basic_trainer_eval.py` | Evaluation helpers: the `metric_grade_eval` fp32 measurement seam (disables any ambient autocast — reported/gate metrics never inherit fp16/bf16 kernels; training keeps autocast for throughput), full-test `test()`, single-batch `validate()` (over `validate_measured`, which also returns the sample count a baseline tolerance needs), `validate_n_batches` over a GPU-resident validation cache (seeded reservoir subsample; pooled bit-identically across trainers via the shared `DataLoaderFactory`), and `validate_correctness_on_indices` for paired per-example McNemar-style comparisons — all measured metric-grade. |
| `basic_trainer_subsample.py` | `test_on_subsample`: deterministic test-set subsample evaluation using the same indices as the chip simulators (shared seed + `max_samples`), with optional VRAM probe logging. |
| `training_recipe.py` | `TrainingRecipe` frozen dataclass plus `build_recipe` (from config, strictly opt-in), `build_param_groups` (weight-decay exclusion + layer-wise LR decay; `only`: the trainer's trained set, U-TGAUGE T1-C), `build_optimizer` (adam/adamw/sgd), and `build_scheduler` (warmup + cosine/constant). [U-TGAUGE2] `clear_step_gradients`: before a step's backward every gradient it may write starts at zero (the optimizer's and the differentiated model's), and a step whose optimizer holds none of that model's parameters refuses by name (`UnwrittenStepError`), so a step never trains nothing while its gradients accumulate. [U-TGAUGE3] The step rule, every step loop: the refusal is decided by what the update writes (`writes`: the parameters of the differentiated model whose gradients the update consumes, a plain step's optimizer's by default), never by the optimizer's membership, and `BasicTrainer._optimize` clears through it with the trainer's `_update_writes` (a transform trainer's deployed parameters its transfer moves), so a gradient another step loop left on the deployed model never enters a trainer's step. |
| `training_utilities.py` | `AccuracyTracker` (forward-hook accuracy), `BasicClassificationLoss` (label-smoothed CE), `CustomClassificationLoss` (CE + activation-overflow penalty via `SavedTensorDecorator`). |
| `weight_transform_trainer.py` | `WeightTransformTrainer`: trains an aux model, applies a weight transformation into the main model each backward pass, and transfers gradients back (QAT for weight transforms). [U-TGAUGE T1-C] Its optimizer and clip read the pairs its update writes (`_written_pairs`: every parameter named a weight or a bias). [U-TGAUGE2] Its gradient transfer moves each gradient to the trained copy and clears the deployed one, so no backward adds into a gradient left from an earlier step. |
| `perceptron_transform_trainer.py` | `PerceptronTransformTrainer`: same aux-model scheme but applies a per-`Perceptron` module transformation through preallocated temp slots, with NaN-sanitized gradient transfer. [U-TGAUGE T1-C] It trains exactly what its update writes: each perceptron's own parameters; a host op's or host normalization's parameters are held out of the optimizer and the clip. [U-PFOLD] Each update resets the transform's copy to the master's per-position state and carries the projected rows and bias offsets of a fold per position to the deployed perceptron (`position_fold.copy_position_state`). |
| `weight_loading.py` | Pretrained-weight strategies (`TorchvisionWeightStrategy`, `CheckpointWeightStrategy`, `URLWeightStrategy`), `resolve_weight_strategy` (raises typed `UnsupportedPreloadError` early for builders without a pretrained source), and the non-raising `torchvision_source_supported` predicate. |

## Declared gradient clip norm (Wave 2 closure)

From the closure unit's step 1 (C4). `training_recipe.py`: `clip_norm_of(norm)` is the one reading of a declared
`grad_clip_norm`: None (no clipping) for an absent or non-positive norm (`TrainingRecipe` defaults the field to 0).
`BasicTrainer._optimize` and the fast ladder's rung loop (`tuning/orchestration/fast_ladder.py`, `_fast_train_rung`)
both read it, so a norm of 0 no longer clips every rung gradient to zero.

## Dependencies
- `data_handling` — `shutdown_data_loader` for clean loader teardown in `BasicTrainer.close()`.
- `chip_simulation` — `compute_test_subsample_indices` so `test_on_subsample` picks the exact indices the simulator runners use.
- `common` — `vram_probe_enabled` env flag gating VRAM logging during subsample evaluation.
- `tuning` — `clone_state_for_trainer` / `restore_state_for_trainer` for best-state snapshot and rollback in `train_steps_until_target`.
- `models` — `SavedTensorDecorator` (activation capture in `CustomClassificationLoss`), `Perceptron` (module slots in `PerceptronTransformTrainer`), and `freeze_batchnorm_running_stats` (the frozen-BN-stats contract for tuner-owned trainers).

## Dependents
- `pipelining` — trainer construction and reuse across pipeline steps (`trainer_pipeline_step`, `trainer_factory`, `simulation_factory`, mapping steps) and pretrained preloading (`weight_preloading_step` uses `build_recipe` + `resolve_weight_strategy`).
- `tuning` — tuner base and orchestration build trainers and recipes (`tuner_base`, `rate_tuner_seam`, `fast_ladder`); `perceptron_transform_tuner` uses `PerceptronTransformTrainer`; `mbh_ledger` measures through the shared `metric_grade_eval` seam.
- `data_handling` — lazy import of `BasicClassificationLoss` as the default loss in `data_provider`.

## Exported API
`__init__.py` re-exports:
- `BasicTrainer` — the standard trainer facade.
- `AccuracyTracker`, `BasicClassificationLoss` — accuracy hook and default loss.
- `WeightTransformTrainer`, `PerceptronTransformTrainer` — aux-model QAT trainers.
- `WeightLoadingStrategy`, `TorchvisionWeightStrategy`, `CheckpointWeightStrategy`, `URLWeightStrategy` — pretrained-weight strategies.
- `UnsupportedPreloadError`, `resolve_weight_strategy`, `torchvision_source_supported` — preload resolution and its typed failure/predicate.

(`CustomClassificationLoss`, the `training_recipe` builders, and the `basic_trainer_*` companions are imported by path, not re-exported.)
