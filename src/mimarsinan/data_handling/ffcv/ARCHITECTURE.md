# data_handling/ffcv/ -- FFCV-backed fast data loading

Drop-in alternative to PyTorch's `DataLoader` for the train / val / test
loops. Trades a one-time cost (writing the dataset to a memory-mapped
`.beton` file) for lower per-batch overhead.

Reachable via `MIMARSINAN_PERF_FFCV=1` *and* the provider opting in via
`enable_ffcv()` (derived from a non-empty `ffcv_transforms()` override).
Without either, `DataLoaderFactory` falls back to the torch DataLoader path.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline_spec.py` | `FieldSpec`, `SplitSpec`, `PipelineSpec`, `normalize_split_name` | Declarative spec for one (provider, dataset). Hashable / JSON-serializable so the beton cache can be keyed by spec. |
| `spec_builder.py` | `infer_spec`, `raw_dataset_for` | Builds a `PipelineSpec` from a provider's surface (`raw_datasets()`, `ffcv_transforms()`, `_preprocessing_spec`, `get_input_shape()`, `get_prediction_mode()`). `raw_dataset_for` returns the FFCV-ready raw dataset, lifted to 3-channel RGB if the model input is 1-channel (FFCV's `RGBImageField` requires it; the GPU postprocess collapses back to 1 via `to_grayscale=True`). |
| `cache.py` | `cache_root`, `beton_dir_for`, `beton_path_for` | Per-spec on-disk paths under `~/.cache/mimarsinan/ffcv/<provider_id>/<spec_hash>/<split>.beton`. Override root with `MIMARSINAN_FFCV_CACHE_DIR`. |
| `writer.py` | `ensure_beton` | Writes the beton if absent. Idempotent + parallel-safe via a sibling `.lock` flock; concurrent processes wait then re-check. Writes go to a `.tmp` sibling then atomic-rename so readers never see partial files. |
| `adapters.py` | `GPUResize`, `GPUNormalize`, `GPUResizeNormalize`, `TorchLoaderShim` | GPU postprocess ops + the torch-style shim around `ffcv.Loader`. The shim clones `x` at the FFCV boundary (FFCV yields views into a rotating buffer pool) and, when given a `label_lookup`, replaces FFCV's per-batch label tensor with an indexed lookup keyed by `batch_indices` pulled from `IndexedLoader`. |
| `label_passthrough.py` | `IndexedLoader`, `preload_labels` | `IndexedLoader` is an `ffcv.Loader` subclass whose iterator surfaces per-batch indices via a thread-safe `batch_index_queue`. `preload_labels` returns all labels as a `torch.LongTensor`, walking `Subset` and `_AsRGB` wrappers and using fast `.targets` / `.labels` metadata when available. Together these bypass FFCV's stock `IntDecoder` (which is unreliable on ViT-class workloads under fused-multi-tensor-apply optimizer state). |
| `loader_factory.py` | `available`, `build_loader`, `FFCVLoaderFactory`, `FFCVNotAvailable` | Entry point used by `DataLoaderFactory`. Materializes ops, wires the `IndexedLoader`, preloads labels on-device, and returns a `TorchLoaderShim` that yields `(x, y)` already on the GPU. |

## Dependencies

- **Internal**: `data_handling.data_provider` (the provider surface).
- **External**: `ffcv` (optional), `torch`, `torchvision`.

## Dependents

- `data_handling.data_loader_factory` (single integration point: `_try_ffcv`).

## Exported API (\_\_init\_\_.py)

See `__init__.py` for the full re-export list. The two surfaces the rest
of the codebase touches are `FFCVLoaderFactory` (constructed by
`DataLoaderFactory`) and `available()` (gate check).

## Provider opt-in contract

Providers declare two per-split dicts that the FFCV layer reads:

- `raw_datasets()` — the underlying transform-free torchvision datasets
  (used as both the beton source and the label-preload source).
- `ffcv_transforms()` — `{"train": [...], "val": [...], "test": [...]}`
  where each entry is `[(ffcv_op_class_name, kwargs_dict), ...]`. FFCV's
  CPU op catalogue is a subset of torchvision's; provider author chooses
  semantically-aligned ops (the `ffcv.transforms` module is the menu).
  Returning the empty default keeps the provider on the torch path.
