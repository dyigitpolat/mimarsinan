# model_training/ -- Training Utilities

Provides the standard training loop, specialized trainers for quantization-aware
training, and shared training utilities.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `basic_trainer.py` | `BasicTrainer` | Standard PyTorch training loop with cosine annealing, mixed precision, early stopping. Step-based APIs: `train_n_steps` (cosine `T_max` = steps), `validate_n_batches` (averaged accuracy over N batches), `train_steps_until_target` (convergence-aware: periodic checks every `check_interval` steps with `patience`-based early stopping; `min_steps` suppresses patience until that many gradient steps; `min_improvement` (default `1e-3`) sets the threshold for a check to count as progress — use `1e-4` for fine recovery; validates via `validate_n_batches`). Batch helpers: `next_training_batch`, `next_validation_batch`, `iter_validation_batches`, `evaluate_loss_on_batch`, and `train_one_step(..., return_post_update_loss=True)` for stable LR probing on fixed batches. Epoch APIs: `train_until_target_accuracy` returns a metric on the **final** model weights. `train_validation_epochs` runs fixed train epochs with validation after each epoch and **does not** call `test()` (tolerance calibration fallback). **Phase-D2 two-tier validation**: `validate_fast()` (small subset, ~16 batches — for LR probes and per-cycle progress checks) and `validate_full()` (full validation loader — for rollback and safety-net decisions); `set_fast_validation_batches` / `set_full_validation_batches` let tuners align the two tiers with their `TuningBudget.progress_eval_batches` / `eval_n_batches`. Legacy `validate()` (single minibatch) is kept for non-tuner callers; `test()` is still the full test split. |
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
