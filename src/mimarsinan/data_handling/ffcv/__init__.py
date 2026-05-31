"""FFCV-backed fast data loading.

Reachable iff a provider opts in by overriding ``ffcv_transforms()`` with
non-empty content. FFCV is then a hard requirement at runtime — there is
no global toggle and no silent fallback. If FFCV isn't importable the
ImportError surfaces normally.
"""

from __future__ import annotations

from mimarsinan.data_handling.ffcv.cache import beton_path_for
from mimarsinan.data_handling.ffcv.loader import IndexedLoader, preload_labels
from mimarsinan.data_handling.ffcv.loader_factory import (
    FFCVLoaderFactory,
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
    "IndexedLoader",
    "FFCVLoaderFactory",
    "beton_path_for",
    "build_loader",
    "ensure_beton",
    "infer_spec",
    "normalize_split_name",
    "preload_labels",
    "raw_dataset_for",
]
