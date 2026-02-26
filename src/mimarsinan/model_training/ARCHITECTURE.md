# model_training/ -- Training Utilities

Provides the standard training loop, specialized trainers for quantization-aware
training, and shared training utilities.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `basic_trainer.py` | `BasicTrainer` | Standard PyTorch training loop with cosine annealing, mixed precision, early stopping |
| `training_utilities.py` | `AccuracyTracker`, `BasicClassificationLoss`, `CustomClassificationLoss` | Loss functions and accuracy tracking |
| `weight_transform_trainer.py` | `WeightTransformTrainer` | Trainer that applies weight transforms each epoch (for quantization-aware training) |
| `perceptron_transform_trainer.py` | `PerceptronTransformTrainer` | Trainer that applies per-perceptron transforms each epoch |

## Dependencies

- **Internal**: `transformations.perceptron_transformer` (`PerceptronTransformer`), `models.layers` (`SavedTensorDecorator`), `models.perceptron_mixer.perceptron` (`Perceptron`).
- **External**: `torch`, `wandb` (via reporter callback).

## Dependents

Most-imported training module. Used by:
- `tuning.tuners` (all tuners use trainers for adaptation recovery)
- `search` evaluators (small-step, TE-NAS)
- `pipelining.pipeline_steps` (pretraining, analysis, quantization, fusion steps)
- `data_handling` (lazy import for `BasicClassificationLoss`)

## Exported API (\_\_init\_\_.py)

`BasicTrainer`, `AccuracyTracker`, `BasicClassificationLoss`,
`WeightTransformTrainer`, `PerceptronTransformTrainer`.
