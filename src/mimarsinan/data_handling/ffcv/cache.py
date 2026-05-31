"""Beton file cache management — per-spec on-disk paths."""

from __future__ import annotations

import os
from pathlib import Path

from mimarsinan.data_handling.ffcv.pipeline_spec import PipelineSpec, normalize_split_name


def cache_root() -> Path:
    """Return the cache root; override with ``MIMARSINAN_FFCV_CACHE_DIR``."""
    override = os.environ.get("MIMARSINAN_FFCV_CACHE_DIR")
    if override:
        root = Path(override).expanduser()
    else:
        root = Path.home() / ".cache" / "mimarsinan" / "ffcv"
    root.mkdir(parents=True, exist_ok=True)
    return root


def beton_dir_for(spec: PipelineSpec) -> Path:
    d = cache_root() / spec.id / spec.stable_hash()
    d.mkdir(parents=True, exist_ok=True)
    return d


def beton_path_for(spec: PipelineSpec, split: str) -> Path:
    return beton_dir_for(spec) / f"{normalize_split_name(split)}.beton"
