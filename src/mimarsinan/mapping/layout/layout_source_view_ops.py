from __future__ import annotations

import numpy as np

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView

def node_ids_of(input_sources) -> set[int]:
    """Set of *real* producer node ids feeding ``input_sources``.

    Skips IRSource sentinels with ``node_id < 0`` (off/input/always-on).
    Fast path for ``LayoutSourceView`` (no iteration).
    """
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
    """View-aware ``np.stack``.

    When every input is a ``LayoutSourceView`` of identical shape, returns a
    merged view with a new axis inserted at ``axis`` (no materialisation).
    Otherwise falls back to materialised ``np.stack``.
    """
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
    """Concatenate views or arrays along ``axis``.

    Returns a ``LayoutSourceView`` when every input is a ``LayoutSourceView``
    of compatible shape (no materialisation).  When any input is already a
    materialised numpy array, falls back to ``np.concatenate`` on materialised
    arrays.
    """
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
