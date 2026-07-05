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
| `basic_trainer.py` | `BasicTrainer`: loaders/iterators lifecycle, optimizer+scheduler construction (legacy Adam path or recipe-driven), `validation_context` metric tagging, pickling; delegates epoch/step/eval/subsample APIs to the companion modules. |
| `basic_trainer_epochs.py` | Epoch-based training: `train_validation_epochs` (fixed epochs, validate each) and `train_until_target_accuracy` (early exit + final `test()`). |
| `basic_trainer_steps.py` | Step-based training: `train_n_steps`, `train_one_step` (post-update loss probing), and `train_steps_until_target` — convergence checks with patience, best-state rollback, optional plateau LR-reduction ladder, reusable external optimizer, `return_steps`/`cosine_decay` modes; `final_validation=False` skips the trailing eval for callers that re-measure with their own basis (A4 consolidation). |
| `basic_trainer_eval.py` | Evaluation helpers: full-test `test()`, single-batch `validate()`, `validate_n_batches` over a GPU-resident validation cache (seeded reservoir subsample; pooled bit-identically across trainers via the shared `DataLoaderFactory`), and `validate_correctness_on_indices` for paired per-example McNemar-style comparisons. |
| `basic_trainer_subsample.py` | `test_on_subsample`: deterministic test-set subsample evaluation using the same indices as the chip simulators (shared seed + `max_samples`), with optional VRAM probe logging. |
| `training_recipe.py` | `TrainingRecipe` frozen dataclass plus `build_recipe` (from config, strictly opt-in), `build_param_groups` (weight-decay exclusion + layer-wise LR decay), `build_optimizer` (adam/adamw/sgd), and `build_scheduler` (warmup + cosine/constant). |
| `training_utilities.py` | `AccuracyTracker` (forward-hook accuracy), `BasicClassificationLoss` (label-smoothed CE), `CustomClassificationLoss` (CE + activation-overflow penalty via `SavedTensorDecorator`). |
| `weight_transform_trainer.py` | `WeightTransformTrainer`: trains an aux model, applies a weight transformation into the main model each backward pass, and transfers gradients back (QAT for weight transforms). |
| `perceptron_transform_trainer.py` | `PerceptronTransformTrainer`: same aux-model scheme but applies a per-`Perceptron` module transformation through preallocated temp slots, with NaN-sanitized gradient transfer. |
| `weight_loading.py` | Pretrained-weight strategies (`TorchvisionWeightStrategy`, `CheckpointWeightStrategy`, `URLWeightStrategy`), `resolve_weight_strategy` (raises typed `UnsupportedPreloadError` early for builders without a pretrained source), and the non-raising `torchvision_source_supported` predicate. |

## Dependencies
- `data_handling` — `shutdown_data_loader` for clean loader teardown in `BasicTrainer.close()`.
- `chip_simulation` — `compute_test_subsample_indices` so `test_on_subsample` picks the exact indices the simulator runners use.
- `common` — `vram_probe_enabled` env flag gating VRAM logging during subsample evaluation.
- `tuning` — `clone_state_for_trainer` / `restore_state_for_trainer` for best-state snapshot and rollback in `train_steps_until_target`.
- `models` — `SavedTensorDecorator` (activation capture in `CustomClassificationLoss`) and `Perceptron` (module slots in `PerceptronTransformTrainer`).

## Dependents
- `pipelining` — trainer construction and reuse across pipeline steps (`trainer_pipeline_step`, `trainer_factory`, `simulation_factory`, mapping steps) and pretrained preloading (`weight_preloading_step` uses `build_recipe` + `resolve_weight_strategy`).
- `tuning` — tuner base and orchestration build trainers and recipes (`tuner_base`, `rate_tuner_seam`, `fast_ladder`); `perceptron_transform_tuner` uses `PerceptronTransformTrainer`.
- `data_handling` — lazy import of `BasicClassificationLoss` as the default loss in `data_provider`.

## Exported API
`__init__.py` re-exports:
- `BasicTrainer` — the standard trainer facade.
- `AccuracyTracker`, `BasicClassificationLoss` — accuracy hook and default loss.
- `WeightTransformTrainer`, `PerceptronTransformTrainer` — aux-model QAT trainers.
- `WeightLoadingStrategy`, `TorchvisionWeightStrategy`, `CheckpointWeightStrategy`, `URLWeightStrategy` — pretrained-weight strategies.
- `UnsupportedPreloadError`, `resolve_weight_strategy`, `torchvision_source_supported` — preload resolution and its typed failure/predicate.

(`CustomClassificationLoss`, the `training_recipe` builders, and the `basic_trainer_*` companions are imported by path, not re-exported.)
