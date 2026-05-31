# data_handling/ffcv/ -- FFCV-backed fast data loading

Drop-in alternative to PyTorch's `DataLoader` for image-classification
training. Trades a one-time cost (writing the dataset to a memory-mapped
`.beton` file) for lower per-batch overhead.

A provider opts in by overriding `ffcv_transforms()` (non-empty result) on
its `DataProvider` subclass. FFCV is then a **hard requirement** at
runtime — no global toggle, no silent fallback. If FFCV isn't importable
the ImportError surfaces normally; any FFCV-side error from the spec /
writer / loader propagates unchanged.

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `pipeline_spec.py` | `FieldSpec`, `SplitSpec`, `PipelineSpec`, `normalize_split_name` | Declarative, hashable, JSON-serializable spec for one (provider, dataset). The hash drives the on-disk cache key. |
| `spec_builder.py` | `infer_spec`, `raw_dataset_for`, `_AsRGB` | Builds a `PipelineSpec` from a provider's surface (`raw_datasets()`, `ffcv_transforms()`) plus the model-input contract from `_preprocessing_spec.resize_to` (synthesized into `RGBImageField.max_resolution` so the beton stores at the size the model wants). The provider's per-split image op chain lands verbatim on the image pipeline — no tail synthesis, no GPU postprocess magic. Only the label tail (`ToTensor`/`ToDevice`/`Squeeze`) is added because the loader bypasses labels anyway. `_AsRGB` is an explicit helper providers can use to lift grayscale source datasets to 3-channel for FFCV's `RGBImageField`. |
| `cache.py` | `cache_root`, `beton_dir_for`, `beton_path_for` | Per-spec on-disk paths under `~/.cache/mimarsinan/ffcv/<provider_id>/<spec_hash>/<split>.beton`. Override with `MIMARSINAN_FFCV_CACHE_DIR`. |
| `writer.py` | `ensure_beton` | Writes the beton if absent. Parallel-safe via a sibling `.lock` flock; concurrent writers wait then re-check. Writes go to a `.tmp` sibling then atomic-rename so readers never see partial files. |
| `loader.py` | `IndexedLoader`, `preload_labels` | `ffcv.Loader` subclass. Iterator yields `(x.clone(), label_lookup[batch_indices])` directly — no shim, no torch-DataLoader-compat layer. The clone owns FFCV's rotating-buffer contract (downstream consumers can hold yielded tensors across iteration boundaries). The label-lookup path bypasses FFCV's stock `IntDecoder`, which is unreliable on ViT-class workloads under fused-multi-tensor-apply optimizer state. `preload_labels(dataset)` builds the lookup; walks `Subset` / `_AsRGB` wrappers and uses fast `.targets` / `.labels` metadata when available. |
| `loader_factory.py` | `build_loader`, `FFCVLoaderFactory` | Materializes ops, runs the writer, preloads labels on-device, and returns an `IndexedLoader`. No try/except, no `available()` gate. |

## Dependencies

- **Internal**: `data_handling.data_provider` (the provider surface).
- **External**: `ffcv` (hard requirement when reached), `torch`, `numpy`.

## Dependents

- `data_handling.data_loader_factory._ffcv_loader` — single integration
  point, gated on `provider.enable_ffcv()`.

## Provider opt-in contract

Providers that want FFCV override these three methods on `DataProvider`:

```python
def raw_datasets(self) -> dict:
    """Per-split raw datasets shared by both paths (beton source)."""

def torch_transforms(self) -> dict:
    """Per-split raw transform lists for the torch path; base class
    wraps with the configured preprocessing."""

def ffcv_transforms(self) -> dict:
    """Per-split FFCV op chains [(op_name, kwargs), ...]. Includes the
    standard tail (ToTensor / ToDevice / ToTorchImage / NormalizeImage)
    that FFCV's image pipeline requires — there is no auto-synthesis.
    Non-empty content opts the provider into FFCV."""
```

Beton-write parameters that depend on the model (`RGBImageField.max_resolution`)
are derived from `_preprocessing_spec.resize_to` — the same model-input
contract the torch path uses — so a provider doesn't double-declare them.

Currently shipping with FFCV opt-in: **CIFAR-10**, **CIFAR-100**.
MNIST / MNIST-32 / ECG / ImageNet inherit the empty default (MNIST/MNIST-32:
FFCV's `RGBImageField` requires 3 channels with no collapse op; ECG: 1D
signal data; ImageNet: train uses `RandomResizedCrop` and val uses
`Resize(256) + CenterCrop(224)` — these can't share a single `resize_to`
under the current spec).
