# training/ -- standalone fast training recipes

Self-contained, super-convergence-style training vehicles that live OUTSIDE the
deployment pipeline (`pipelining/`). They are research/benchmark drivers, not
pipeline steps: nothing here is registered with `BasicDataProviderFactory` or the
step graph, and importing this package pulls in no FFCV (guarded runtime probe).

## Key Components

| File | Symbols | Purpose |
|------|---------|---------|
| `imagenet_fast_train.py` | `FastImageNetRecipe`, `one_cycle_lr_schedule`, `progressive_resize_schedule`, `label_smoothing_cross_entropy`, `build_resnet50_channels_last`, `train_step`, `build_imagenet_dataloaders` | Fast ResNet-50 ImageNet-from-scratch recipe (FFCV / fast.ai super-convergence): one-cycle LR (linear warmup -> cosine decay), AMP (CUDA-only), channels-last, label smoothing, progressive resizing (small -> eval-matched), SGD+momentum+nesterov, large batch. `build_imagenet_dataloaders` PREFERS FFCV (`_ffcv_available()` guarded import + repo `FFCVLoaderFactory`) and falls back to the provider's optimized torchvision loaders. `build_resnet50_channels_last` reuses `models.pretrained_bridge.load_pretrained_resnet50(pretrained=False)` (random init) so the trunk matches the mapping-measured deployable. |

## Dependencies

- **Internal**: `models.pretrained_bridge` (ResNet-50 trunk), `data_handling.ffcv.loader_factory` (FFCV path, lazy import), the ImageNet provider's `fast_fallback_dataloaders` (torchvision path).
- **External**: `torch` (required); `ffcv` (OPTIONAL — probed via guarded import, never hard-required at module top); `torchvision` (via the pretrained bridge / provider).

## Dependents

- None inside the framework. Driven by a SUPERVISED post-build training run
  (the FFCV install + the actual ImageNet run are out-of-band steps).

## Recipe / dataloader contract

`build_imagenet_dataloaders(provider, batch_size, num_workers, prefer_ffcv, device)`
returns `{"train", "val", "test"}` loaders. Selection:
`prefer_ffcv and _ffcv_available()` -> FFCV (`FFCVLoaderFactory`), else the
provider's `fast_fallback_dataloaders(...)` (many workers, `pin_memory`,
`persistent_workers`, `prefetch_factor`). The FFCV probe is the ONLY place FFCV
is referenced; this module imports cleanly with no FFCV installed.
