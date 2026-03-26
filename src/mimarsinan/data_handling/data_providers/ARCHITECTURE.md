# data_handling/data_providers/ -- Concrete Dataset Providers

Contains concrete `DataProvider` implementations that self-register with
`BasicDataProviderFactory` via the `@register` decorator.

## Key Components

| File | Provider Name | Dataset |
|------|--------------|---------|
| `mnist_data_provider.py` | `"mnist"` | MNIST 28x28 |
| `mnist32_data_provider.py` | `"mnist32"` | MNIST resized to 32x32 |
| `cifar10_data_provider.py` | `"cifar10"` | CIFAR-10 |
| `cifar100_data_provider.py` | `"cifar100"` | CIFAR-100 |
| `ecg_data_provider.py` | `"ecg"` | ECG time-series classification |
| `imagenet_data_provider.py` | `"ImageNet_DataProvider"` | ImageNet ILSVRC2012: symlinks `<datasets_path>/imagenet` → `IMAGENET_ROOT` from `.env` when set |

## Dependencies

- **Internal**: `data_handling.data_provider.DataProvider`, `data_handling.data_provider_factory.BasicDataProviderFactory`.
- **Internal**: inline ``dotenv`` load of repo ``.env`` (read ``IMAGENET_ROOT``, create symlink under ``datasets_path``).
- **External**: `torchvision`, `torch`.

## Dependents

Imported by `data_handling/__init__.py` (via `data_providers/__init__.py`) to
trigger registration side effects.

## Exported API (\_\_init\_\_.py)

Module-level imports for side-effect registration only (no symbol re-exports).
