"""FFCV-backed fast data loading."""

from __future__ import annotations

from mimarsinan.data_handling.ffcv.adapters import (
    GPUNormalize,
    GPUResize,
    GPUResizeNormalize,
    TorchLoaderShim,
)
from mimarsinan.data_handling.ffcv.cache import beton_path_for
from mimarsinan.data_handling.ffcv.label_passthrough import (
    IndexedLoader,
    preload_labels,
)
from mimarsinan.data_handling.ffcv.loader_factory import (
    FFCVLoaderFactory,
    FFCVNotAvailable,
    available,
    build_loader,
)
from mimarsinan.data_handling.ffcv.pipeline_spec import (
    FieldSpec,
    PipelineSpec,
    SplitSpec,
    normalize_split_name,
)
from mimarsinan.data_handling.ffcv.spec_builder import (
    infer_spec,
    raw_dataset_for,
)
from mimarsinan.data_handling.ffcv.writer import ensure_beton

__all__ = [
    "FieldSpec",
    "PipelineSpec",
    "SplitSpec",
    "TorchLoaderShim",
    "GPUResize",
    "GPUNormalize",
    "GPUResizeNormalize",
    "IndexedLoader",
    "FFCVLoaderFactory",
    "FFCVNotAvailable",
    "available",
    "beton_path_for",
    "build_loader",
    "ensure_beton",
    "infer_spec",
    "normalize_split_name",
    "preload_labels",
    "raw_dataset_for",
]
