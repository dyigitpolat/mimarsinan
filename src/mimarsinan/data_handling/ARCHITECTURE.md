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
| `data_provider_factory.py` | `DataProviderFactory` interface (its `provider_name`, the registered provider id a factory serves, `None` when undeclared; the session binds a handed-in factory's name to the declared `data_provider_name` and the deployment record's workload reads it) and `BasicDataProviderFactory`: `@register` name registry, cached `create()` for split stability, and GUI metadata (`list_registered`, `get_metadata`). |
| `data_loader_factory.py` | `DataLoaderFactory` builds train/val/test loaders (forkserver workers, FFCV routing iff the provider opts in — no silent fallback) and `shutdown_data_loader` for controlled multi-worker teardown. `for_pipeline` returns one SHARED factory per pipeline that pools loaders and device-side eval caches across every trainer of the run (W3 wall: trainer construction stops re-spawning workers); shared pools are pickle-safe (dropped) and factory-owned (`owns_loaders`), so `BasicTrainer.close()` leaves them alive. |
| `batch_integrity.py` | Batch-integrity SSOT for every number this program reports or decides on. A float input batch carrying non-finite values is a corrupted READ of an immutable source, never data (FFCV hands out views into device buffers a producer thread refills across unrelated GPU work; a batch has been observed coming back entirely NaN and bit-correct microseconds later). `own_verified_batch` is the one call a measurement seam makes (own THEN verify, in that order, because a verdict about a live producer view covers bytes nobody later reads); `own_batch` is its snapshot half for readers that repair instead of raising, `read_verified_batch` repairs and re-reads a single batch, `restart_and_advance` is the repair those retries escalate (fresh iterator, then a positional advance), `verified_pass` restarts a full loader pass whole and raises rather than measure a silently reduced set, `verify_owned_batches` checks a materialized eval cache. |
| `preprocessing.py` | `PreprocessingSpec` / `resolve_preprocessing`: config-driven resize + normalization composed around each provider's raw transform list. Named presets are PROVIDER-REGISTERED (`register_normalization_preset`; the framework ships no dataset constants). |
| `dataset_views.py` | `ApplyTransform` view composing a raw dataset with a per-call transform, and the `SizedDataset` protocol; keeps raw datasets transform-free for beton writing. |
| `sealed_paths.py` | Sealed paths: `seal_paths` installs one process-wide audit hook (on the first non-empty seal; seals only tighten) that refuses every `open` of a sealed file, directory or anchored pattern by name (`SealedPathRefusal`), the path taken as named and as resolved through its links, whichever library opens it (torchvision, numpy, torch, tarfile). `PipelineSession` seals a run's declared `sealed_paths` before any provider exists, so a held-out split stays closed until its declared read. A process the run spawns rather than forks starts unsealed. |
| `sample_loader.py` | `load_test_sample_by_index` / `load_test_samples_by_index`: deterministic test-set samples by global index (hardware parity steps). These samples ARE the deployment claim — a chip simulator runs them and the result is compared against a torch reference — so the read is a `batch_integrity.verified_pass` and the samples are owned snapshots, not views a producer may still refill. |
| `data_providers/` | Registered concrete providers: MNIST, MNIST-32, Fashion-MNIST, KMNIST, SVHN, CIFAR-10, CIFAR-100, ECG, ImageNet; CIFAR-10/100 opt into FFCV. CIFAR/ImageNet register their canonical normalization presets at import. |
| `ffcv/` | FFCV-backed fast loading: beton cache/writer, pipeline specs inferred from providers, and an `IndexedLoader` exposing per-batch indices. |

## Dependencies
- `common` — `env` for environment knobs (`DISABLE_FFCV_VAR`/`ffcv_disabled` kill-switch, `resource_debug_enabled`, `ffcv_cache_dir`, `imagenet_root`); `best_effort` for never-raising loader shutdown and resource snapshots; `refusal.ToolchainRefusal` for the sealed-path refusals.
- `model_training` — `training_utilities.BasicClassificationLoss`, lazily imported inside `ClassificationMode.create_loss()` to avoid a hard import cycle.

## Dependents
- `pipelining` — pipeline steps and the deployment pipeline construct providers and loaders.
- `tuning` — tuners evaluate candidates through data loaders.
- `search` — evaluators need providers/loaders for scoring architectures, and `batch_integrity` for the fitness pass.
- `chip_simulation` — simulation runners fetch deterministic test samples and loaders, and verify the read with `batch_integrity`.
- `model_training` — trainers consume loaders, `shutdown_data_loader`, and `batch_integrity` on every reported/decided accuracy read.
- `gui` — provider registry metadata for dataset dropdowns.

## Exported API
- `DataProvider`, `ClassificationMode`, `RegressionMode` — provider base class and prediction modes.
- `DataProviderFactory`, `BasicDataProviderFactory` — factory interface and registry-backed implementation.
- `DataLoaderFactory`, `shutdown_data_loader` — loader construction and multi-worker teardown.

## Declared sample split (U-HV)

`sample_loader.load_pairs_by_index(split=...)` is the split-parameterized read beside the test-set loaders of the
Key files table: `DEVELOPMENT_SPLIT` (the validation carve) or `"test"` by name; any other split is refused by name
(`UnknownSampleSplit`) before a provider is built, and `sample_read` records a read (declaring key, split, indices).
The ODIN steps read through it with the split `odin_sample_split` declares
(`pipelining/pipeline_steps/verification/odin/sample_split.py`).
