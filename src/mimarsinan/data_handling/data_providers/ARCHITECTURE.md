# data_handling/data_providers/ -- Concrete Dataset Providers

Contains concrete `DataProvider` implementations that self-register with
`BasicDataProviderFactory` via the `@register` decorator.

## Key Components

| File | Provider Name | Dataset |
|------|--------------|---------|
| `mnist_data_provider.py` | `"MNIST_DataProvider"` | MNIST 28x28 |
| `mnist32_data_provider.py` | `"MNIST32_DataProvider"` | MNIST resized to 32x32 |
| `fashion_mnist_data_provider.py` | `"FashionMNIST_DataProvider"` | Fashion-MNIST 1x28x28, 10 classes |
| `kmnist_data_provider.py` | `"KMNIST_DataProvider"` | KMNIST 1x28x28, 10 classes |
| `svhn_data_provider.py` | `"SVHN_DataProvider"` | SVHN 3x32x32, 10 classes (`split=` train/test) |
| `cifar10_data_provider.py` | `"cifar10"` | CIFAR-10 |
| `cifar100_data_provider.py` | `"cifar100"` | CIFAR-100 |
| `ecg_data_provider.py` | `"ecg"` | ECG time-series classification |
| `imagenet_data_provider.py` | `"ImageNet_DataProvider"` | ImageNet ILSVRC2012: symlinks `<datasets_path>/imagenet` → `IMAGENET_ROOT` from `.env` when set; validation = tail of `split="train"` (default 99/1 like CIFAR-10); test = official `split="val"`. Also exposes `fast_fallback_dataloaders(batch_size, num_workers)` — the non-FFCV torchvision fast-recipe path (many workers + `pin_memory` + `persistent_workers` + `prefetch_factor`; train shuffles & drops-last, val/test sequential) consumed by `training.imagenet_fast_train`. This is NOT a pipeline step and does not alter the framework's torch-DataLoader path. |

## Dependencies

- **Internal**: `data_handling.data_provider.DataProvider`, `data_handling.data_provider_factory.BasicDataProviderFactory`.
- **Internal**: inline ``dotenv`` load of repo ``.env`` (read ``IMAGENET_ROOT``, create symlink under ``datasets_path``).
- **External**: `torchvision`, `torch`.

## Dependents

Imported by `data_handling/__init__.py` (via `data_providers/__init__.py`) to
trigger registration side effects.

## Exported API (\_\_init\_\_.py)

Module-level imports for side-effect registration only (no symbol re-exports).
