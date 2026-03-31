# model_training/ -- Training Utilities

Provides the standard training loop, specialized trainers for quantization-aware
training, and shared training utilities.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `basic_trainer.py` | `BasicTrainer` | Standard PyTorch training loop with cosine annealing, mixed precision, early stopping. `train_until_target_accuracy` returns a metric measured on the **final** model weights (after any extra epochs triggered by early stopping). `train_validation_epochs` runs a fixed number of train epochs with validation after each epoch and **does not** call `test()` (used for smooth-tolerance calibration). `validate()` evaluates one minibatch (fast probe); `test()` evaluates the full test split. |
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
