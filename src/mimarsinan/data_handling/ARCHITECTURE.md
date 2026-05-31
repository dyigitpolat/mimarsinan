# data_handling/ -- Dataset Management

Provides abstractions for loading and serving datasets to the training,
tuning, and evaluation subsystems.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `data_provider.py` | `DataProvider`, `ClassificationMode`, `RegressionMode` | Abstract base. Subclasses override three per-split dict methods: `raw_datasets()` (datasets shared by both paths), `torch_transforms()` (raw torchvision transform lists; base class wraps with preprocessing in `_assemble_split` → `_wrap_with_preprocessing`), and `ffcv_transforms()` (FFCV op chains `[(op_name, kwargs), ...]`; non-empty opts the provider into FFCV — `enable_ffcv()` is derived). Beton write parameters like `max_resolution` are derived from `_preprocessing_spec.resize_to` (the canonical model-input contract) — no separate provider override. `get_validation_batch_size()` matches training batch size capped by validation set size. |
| `data_provider_factory.py` | `DataProviderFactory`, `BasicDataProviderFactory` | Factory with class-level registry (`@register` decorator) |
| `data_loader_factory.py` | `DataLoaderFactory`, `shutdown_data_loader` | Creates PyTorch `DataLoader`s for train/val/test splits; validation loaders use `shuffle=False`. Routes through `ffcv/` iff the provider opts in via `enable_ffcv()`. FFCV is a hard requirement when opted in — any FFCV import/build/load failure propagates; no silent fallback to the torch path. Helper to shut down multi-worker loaders. |
| `dataset_views.py` | `ApplyTransform` | Wrapper that composes a raw dataset with a per-call transform. The base class uses this in `_assemble_split` to keep the raw dataset transform-free (the form FFCV needs for beton writing). |
| `test_sample_loader.py` | `load_test_sample_by_index`, `load_test_samples_by_index` | Deterministic test-set sample fetch by global index (Loihi/SANA-FE parity steps). |
| `data_providers/` | Concrete providers | MNIST, MNIST-32, CIFAR-10, CIFAR-100, ECG, ImageNet; each may set `DISPLAY_LABEL` for GUI dropdowns. CIFAR-10 / CIFAR-100 opt into FFCV by overriding `ffcv_transforms()` with explicit per-split op chains. MNIST / MNIST-32 / ECG / ImageNet inherit the empty default (MNIST: no grayscale-collapse op available in stock FFCV; ECG: 1D signal data; ImageNet: train/val use different crop policies that don't share a single `resize_to`). |
| `ffcv/` | FFCV-backed fast data loading | See `ffcv/ARCHITECTURE.md` |

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
