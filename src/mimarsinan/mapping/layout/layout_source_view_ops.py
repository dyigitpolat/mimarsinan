from __future__ import annotations

from typing import Sequence

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView

def node_ids_of(input_sources) -> set[int]:
    """Set of real producer node ids feeding ``input_sources``; skips
    sentinel IRSources with ``node_id < 0`` (off/input/always-on)."""
    if input_sources is None:
        return set()
    if isinstance(input_sources, LayoutSourceView):
        return set(input_sources._node_ids)
    arr = np.asarray(input_sources, dtype=object)
    ids: set[int] = set()
    for src in arr.ravel():
        if isinstance(src, IRSource) and src.node_id >= 0:
            ids.add(src.node_id)
    return ids


def total_size(input_sources) -> int:
    """Total cell count of ``input_sources``; no materialisation."""
    if isinstance(input_sources, LayoutSourceView):
        return input_sources.size
    return int(np.asarray(input_sources, dtype=object).size)


def stack_source_views(
    inputs: Sequence,
    axis: int = 0,
) -> "LayoutSourceView | np.ndarray":
    """View-aware ``np.stack``: returns a merged view (no materialisation) when all
    inputs are same-shape ``LayoutSourceView``s, else falls back to ``np.stack``."""
    if not inputs:
        raise ValueError("stack_source_views: empty input list")
    if not all(isinstance(x, LayoutSourceView) for x in inputs):
        return np.stack(
            [np.asarray(x, dtype=object) for x in inputs], axis=axis,
        )
    base_shape = inputs[0].shape
    if not all(v.shape == base_shape for v in inputs):
        return np.stack(
            [np.asarray(x, dtype=object) for x in inputs], axis=axis,
        )
    new_shape = list(base_shape)
    if axis < 0:
        axis += len(new_shape) + 1
    new_shape.insert(axis, len(inputs))
    new_shape_t = tuple(new_shape)
    new_node_ids: set[int] = set()
    for v in inputs:
        new_node_ids.update(v._node_ids)

    materialisers = [v._materialiser for v in inputs]

    def _materialise() -> np.ndarray:
        return np.stack([m() for m in materialisers], axis=axis)

    return LayoutSourceView(new_shape_t, new_node_ids, _materialise)


def concat_source_views(
    inputs: Sequence,
    axis: int = 0,
) -> "LayoutSourceView | np.ndarray":
    """Concatenate along ``axis``: returns a ``LayoutSourceView`` (no materialisation)
    when all inputs are compatible views, else falls back to ``np.concatenate``."""
    if not inputs:
        raise ValueError("concat_source_views: empty input list")

    all_views = all(isinstance(x, LayoutSourceView) for x in inputs)
    if not all_views:
        materialised = [np.asarray(x, dtype=object) for x in inputs]
        return np.concatenate(materialised, axis=axis)

    base_shape = inputs[0].shape
    ndim = len(base_shape)
    for v in inputs[1:]:
        if len(v.shape) != ndim:
            materialised = [np.asarray(x, dtype=object) for x in inputs]
            return np.concatenate(materialised, axis=axis)
        for d_axis in range(ndim):
            if d_axis == axis:
                continue
            if v.shape[d_axis] != base_shape[d_axis]:
                materialised = [np.asarray(x, dtype=object) for x in inputs]
                return np.concatenate(materialised, axis=axis)

    new_shape = list(base_shape)
    new_shape[axis] = sum(v.shape[axis] for v in inputs)
    new_shape_t = tuple(new_shape)
    new_node_ids: set[int] = set()
    for v in inputs:
        new_node_ids.update(v._node_ids)

    materialisers = [v._materialiser for v in inputs]

    def _materialise() -> np.ndarray:
        return np.concatenate([m() for m in materialisers], axis=axis)

    return LayoutSourceView(new_shape_t, new_node_ids, _materialise)
