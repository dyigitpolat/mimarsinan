# data_handling/ -- Dataset Management

Provides abstractions for loading and serving datasets to the training,
tuning, and evaluation subsystems.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `data_provider.py` | `DataProvider`, `ClassificationMode`, `RegressionMode` | Abstract base for dataset providers; prediction mode protocols |
| `data_provider_factory.py` | `DataProviderFactory`, `BasicDataProviderFactory` | Factory with class-level registry (`@register` decorator) |
| `data_loader_factory.py` | `DataLoaderFactory` | Creates PyTorch `DataLoader`s for train/val/test splits |
| `data_providers/` | Concrete providers | MNIST, CIFAR-10, CIFAR-100, ECG implementations |

## Dependencies

- **Internal**: `model_training.training_utilities.BasicClassificationLoss` (lazy import in `ClassificationMode`).
- **External**: `torch`, `torchvision`.

## Dependents

Most-imported module in the codebase. Used by:
- `model_training` (trainer needs data loaders)
- `tuning` (tuners need data loaders for evaluation)
- `search` (evaluators need data loaders and providers)
- `pipelining` (pipeline steps and deployment pipeline)
- `chip_simulation` (simulation runner)

## Exported API (\_\_init\_\_.py)

`DataProvider`, `ClassificationMode`, `RegressionMode`, `DataProviderFactory`,
`BasicDataProviderFactory`, `DataLoaderFactory`.
