# data_handling/ -- Dataset Management

Provides abstractions for loading and serving datasets to the training,
tuning, and evaluation subsystems.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `data_provider.py` | `DataProvider`, `ClassificationMode`, `RegressionMode` | Abstract base for dataset providers; prediction mode protocols |
| `data_provider_factory.py` | `DataProviderFactory`, `BasicDataProviderFactory` | Factory with class-level registry (`@register` decorator) |
| `data_loader_factory.py` | `DataLoaderFactory`, `shutdown_data_loader` | Creates PyTorch `DataLoader`s for train/val/test splits; helper to shut down multi-worker loaders |
| `data_providers/` | Concrete providers | MNIST, CIFAR-10, CIFAR-100, ECG, ImageNet; each may set `DISPLAY_LABEL` for GUI dropdowns |

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
`BasicDataProviderFactory`, `DataLoaderFactory`, `shutdown_data_loader`.

## Multi-worker DataLoader shutdown

Any code that creates a `DataLoader` with `num_workers > 0` must call
`shutdown_data_loader(loader)` when done (e.g. in a `finally` block or at end of
use). This ensures worker processes and IPC queues are torn down in a controlled
order and avoids `OSError: Bad file descriptor` and related errors on process exit.
After shutting down workers, we unregister PyTorch's atexit worker cleanup
callbacks (registered when `persistent_workers` and `pin_memory` are True) so
process exit is fast and Ctrl+C does not produce "Exception ignored in atexit
callback" noise.
