# model_training/ -- Training Utilities

Provides the standard training loop, specialized trainers for quantization-aware
training, and shared training utilities.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `basic_trainer.py` | `BasicTrainer` | Standard PyTorch training loop with cosine annealing, mixed precision, early stopping. **`test_on_subsample`**: deterministic test subsample (same indices as `SimulationRunner` / SCM / HCM when `seed` + `max_simulation_samples` match). Step-based APIs: `train_n_steps` (cosine `T_max` = steps), `validate_n_batches` (averaged accuracy), `train_steps_until_target` (convergence-aware: periodic checks every `check_interval` steps with `patience`-based early stopping; `min_steps` suppresses patience until that many gradient steps; `min_improvement` (default `1e-3`) sets the threshold for a check to count as progress — use `1e-4` for fine recovery; validates via `validate_n_batches`). Batch helpers: `next_training_batch`, `next_validation_batch`, `iter_validation_batches`, `evaluate_loss_on_batch`, and `train_one_step(..., return_post_update_loss=True)` for stable LR probing on fixed batches. Epoch APIs: `train_until_target_accuracy` returns a metric on the **final** model weights. `train_validation_epochs` runs fixed train epochs with validation after each epoch and **does not** call `test()` (tolerance calibration fallback). `validate()` is one minibatch; `test()` is the full test split. **Eval forwards** (`test` / `validate_n_batches`) run under `torch.amp.autocast("cuda")` on cuda devices; argmax-accuracy is robust to FP16. **`iter_validation_batches`** serves from a GPU-resident cache built lazily on first use (cleared on `set_validation_batch_size`). **`validate_correctness_on_indices(batch_indices)`** returns per-example correctness (bool list) over fixed validation-cache batches — the paired-eval primitive for the McNemar rollback gate (tuning P2b); reads only the validation cache (never `test()`, preserving test-set isolation). Step-based optimizer (`_get_optimizer_and_scheduler_steps`) uses fused Adam on cuda. `train_n_steps` / `train_steps_until_target` accept an optional `optimizer=` (default `None` → build fresh + `del`, bit-exact; supplied → reused and never deleted so Adam moments persist across recovery calls, tuning P6); `build_step_optimizer(lr)` builds an ownable step optimizer and `_scheduler_and_scaler_for_optimizer` builds the matching schedule/scaler around it. **Validation tagging**: `validation_context(kind)` is a reentrant context manager that suffixes validation metric names emitted inside the block with `" (kind)"` (e.g. `"Validation accuracy (probe)"`). Tuners wrap exploratory validations (LR range search, rate-proposal evaluation) in `validation_context("probe")` so the GUI Accuracy panel can render committed tuning progress and exploratory probes as distinct traces on the same chart; untagged validations keep the original name. |
| `training_utilities.py` | `AccuracyTracker`, `BasicClassificationLoss`, `CustomClassificationLoss` | Loss functions and accuracy tracking |
| `weight_transform_trainer.py` | `WeightTransformTrainer` | Trainer that applies weight transforms each epoch (for quantization-aware training) |
| `perceptron_transform_trainer.py` | `PerceptronTransformTrainer` | Trainer that applies per-perceptron transforms each epoch |
| `weight_loading.py` | `WeightLoadingStrategy`, `TorchvisionWeightStrategy`, `CheckpointWeightStrategy`, `URLWeightStrategy`, `resolve_weight_strategy` | Strategy pattern for loading pretrained weights from various sources |

## Dependencies

- **Internal**: `transformations.perceptron_transformer` (`PerceptronTransformer`), `models.layers` (`SavedTensorDecorator`), `models.perceptron_mixer.perceptron` (`Perceptron`).
- **External**: `torch`, reporting via reporter callback.

## Dependents

Most-imported training module. Used by:
- `tuning.tuners` (all tuners use trainers for adaptation recovery)
- `search` evaluators (small-step, TE-NAS)
- `pipelining.pipeline_steps` (pretraining, analysis, quantization, fusion steps)
- `data_handling` (lazy import for `BasicClassificationLoss`)

## Exported API (\_\_init\_\_.py)

`BasicTrainer`, `AccuracyTracker`, `BasicClassificationLoss`,
`WeightTransformTrainer`, `PerceptronTransformTrainer`.
