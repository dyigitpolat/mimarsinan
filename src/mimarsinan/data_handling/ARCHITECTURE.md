# data_handling/ — Dataset providers, preprocessing, and DataLoader construction

Supplies every training, tuning, search, and chip-simulation stage with datasets
through one abstraction: a `DataProvider` declares per-split raw datasets and
transform chains, and a `DataLoaderFactory` turns a provider into PyTorch
`DataLoader`s (or FFCV loaders when the provider opts in via non-empty
`ffcv_transforms()`). Providers are created by name through
`BasicDataProviderFactory`'s class-level registry, and a config-driven
`PreprocessingSpec` (resize + normalize) wraps each provider's native
transforms so model input shape is a pipeline concern, not provider code.

## Key files
| File | Purpose |
|---|---|
| `data_provider.py` | `DataProvider` base (per-split `raw_datasets()` / `torch_transforms()` / `ffcv_transforms()` hooks, seeded splits, batch-size policy, and the `workload_profile()` registration hook — all-None `DataWorkloadProfile` unless a provider declares facts) plus `ClassificationMode` / `RegressionMode` prediction modes that supply losses. |
| `data_provider_factory.py` | `DataProviderFactory` interface and `BasicDataProviderFactory`: `@register` name registry, cached `create()` for split stability, and GUI metadata (`list_registered`, `get_metadata`). |
| `data_loader_factory.py` | `DataLoaderFactory` builds train/val/test loaders (forkserver workers, FFCV routing iff the provider opts in — no silent fallback) and `shutdown_data_loader` for controlled multi-worker teardown. `for_pipeline` returns one SHARED factory per pipeline that pools loaders and device-side eval caches across every trainer of the run (W3 wall: trainer construction stops re-spawning workers); shared pools are pickle-safe (dropped) and factory-owned (`owns_loaders`), so `BasicTrainer.close()` leaves them alive. |
| `preprocessing.py` | `PreprocessingSpec` / `resolve_preprocessing`: config-driven resize + normalization (with named presets) composed around each provider's raw transform list. |
| `dataset_views.py` | `ApplyTransform` view composing a raw dataset with a per-call transform, and the `SizedDataset` protocol; keeps raw datasets transform-free for beton writing. |
| `sample_loader.py` | `load_test_sample_by_index` / `load_test_samples_by_index`: deterministic test-set samples by global index (hardware parity steps). |
| `data_providers/` | Registered concrete providers: MNIST, MNIST-32, Fashion-MNIST, KMNIST, SVHN, CIFAR-10, CIFAR-100, ECG, ImageNet; CIFAR-10/100 opt into FFCV. |
| `ffcv/` | FFCV-backed fast loading: beton cache/writer, pipeline specs inferred from providers, and an `IndexedLoader` exposing per-batch indices. |

## Dependencies
- `common` — `env` for environment knobs (`DISABLE_FFCV_VAR`/`ffcv_disabled` kill-switch, `resource_debug_enabled`, `ffcv_cache_dir`, `imagenet_root`); `best_effort` for never-raising loader shutdown and resource snapshots.
- `model_training` — `training_utilities.BasicClassificationLoss`, lazily imported inside `ClassificationMode.create_loss()` to avoid a hard import cycle.

## Dependents
- `pipelining` — pipeline steps and the deployment pipeline construct providers and loaders.
- `tuning` — tuners evaluate candidates through data loaders.
- `search` — evaluators need providers/loaders for scoring architectures.
- `chip_simulation` — simulation runners fetch deterministic test samples and loaders.
- `model_training` — trainers consume loaders and `shutdown_data_loader`.
- `gui` — provider registry metadata for dataset dropdowns.

## Exported API
- `DataProvider`, `ClassificationMode`, `RegressionMode` — provider base class and prediction modes.
- `DataProviderFactory`, `BasicDataProviderFactory` — factory interface and registry-backed implementation.
- `DataLoaderFactory`, `shutdown_data_loader` — loader construction and multi-worker teardown.
