"""Lightweight composable shape descriptor for the shape-only mapping path.

``LayoutSourceView`` duck-types as a numpy object array of ``IRSource`` for
all numpy-style operations the mapper graph uses (``.shape``, ``.flatten()``,
``.reshape()``, ``.transpose()``, ``__getitem__``, ``__array__``,
``concat_source_views``), but stores only ``(shape, node_ids, materialiser)``
and never allocates per-cell ``IRSource`` instances until something forces
materialisation via ``np.asarray`` or iteration.

The shape-only ``LayoutIRMapping`` path consumes only ``.shape`` and the set
of producer node ids, so it never materialises.  The full ``IRMapping``
subclass converts ``input_sources`` to real ``IRSource`` numpy arrays at the
IR-graph boundary (via ``np.asarray``) and replaces its own returned values
with the materialised arrays, so view leakage stops there and downstream
mappers in the full path see only real arrays.

Materialisation of a from-producer view yields
``IRSource(producer_node_id, original_flat_index)`` for every cell, matching
the pre-refactor ``np.array([IRSource(...)...], dtype=object)`` shape.
Reshape, flatten, transpose, indexing, and concatenation compose lambdas so
the eventual materialised array obeys the corresponding numpy semantics.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from mimarsinan.mapping.ir import IRSource


class LayoutSourceView:
    """Duck-typed numpy object-array of ``IRSource`` that defers allocation."""

    __slots__ = ("_shape", "_size", "_node_ids", "_materialiser")

    def __init__(
        self,
        shape: tuple[int, ...],
        node_ids: Iterable[int],
        materialiser: Callable[[], np.ndarray],
    ) -> None:
        self._shape = tuple(int(d) for d in shape)
        size = 1
        for d in self._shape:
            size *= d
        self._size = size
        self._node_ids = frozenset(int(n) for n in node_ids)
        self._materialiser = materialiser

    @classmethod
    def from_producer(
        cls,
        *,
        producer_node_id: int,
        shape: tuple[int, ...] | Sequence[int],
    ) -> "LayoutSourceView":
        """Canonical view for one softcore / compute-op's outputs.

        Materialises to ``IRSource(producer_node_id, i)`` for flat index ``i``
        reshaped to ``shape``.
        """
        shape_t = tuple(int(d) for d in shape)
        size = 1
        for d in shape_t:
            size *= d
        node_id = int(producer_node_id)

        def _materialise() -> np.ndarray:
            arr = np.empty(size, dtype=object)
            for i in range(size):
                arr[i] = IRSource(node_id, i)
            return arr.reshape(shape_t)

        node_ids = (node_id,) if node_id >= 0 else ()
        return cls(shape_t, node_ids, _materialise)

    # numpy duck-type properties

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return self._size

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(object)

    @property
    def T(self) -> "LayoutSourceView":
        return self.transpose()

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of 0-d LayoutSourceView")
        return self._shape[0]

    # Shape-changing view ops (composes materialiser; no eager materialisation)

    def reshape(self, *shape) -> "LayoutSourceView":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new_shape = self._resolve_shape(shape)
        parent = self._materialiser
        return LayoutSourceView(
            new_shape, self._node_ids,
            lambda: parent().reshape(new_shape),
        )

    def flatten(self) -> "LayoutSourceView":
        parent = self._materialiser
        size = self._size
        return LayoutSourceView(
            (size,), self._node_ids,
            lambda: parent().reshape(size),
        )

    def ravel(self) -> "LayoutSourceView":
        return self.flatten()

    def copy(self) -> "LayoutSourceView":
        return LayoutSourceView(self._shape, self._node_ids, self._materialiser)

    def transpose(self, axes: Sequence[int] | None = None) -> "LayoutSourceView":
        if axes is None:
            axes_t: tuple[int, ...] = tuple(range(self.ndim))[::-1]
        else:
            axes_t = tuple(int(a) for a in axes)
        new_shape = tuple(self._shape[a] for a in axes_t)
        parent = self._materialiser
        return LayoutSourceView(
            new_shape, self._node_ids,
            lambda: parent().transpose(axes_t),
        )

    def __getitem__(self, idx) -> "LayoutSourceView | IRSource":
        if isinstance(idx, int) and self.ndim == 1:
            return self._materialise_cell((idx,))
        if isinstance(idx, tuple) and all(isinstance(i, int) for i in idx) \
                and len(idx) == self.ndim:
            return self._materialise_cell(idx)

        new_shape = self._resolve_getitem_shape(idx)
        parent = self._materialiser
        return LayoutSourceView(
            new_shape, self._node_ids,
            lambda: parent()[idx],
        )

    # Materialisation

    def __array__(self, dtype=None) -> np.ndarray:
        arr = self._materialiser()
        if dtype is not None and arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr

    def __iter__(self):
        for cell in self._materialiser().ravel():
            yield cell

    def _materialise_cell(self, idx: tuple[int, ...]) -> IRSource:
        arr = self._materialiser()
        return arr[idx]

    # Internal shape helpers

    def _resolve_shape(self, shape: Iterable[int]) -> tuple[int, ...]:
        shape_t = tuple(int(d) for d in shape)
        unknown = [i for i, d in enumerate(shape_t) if d == -1]
        if len(unknown) > 1:
            raise ValueError(
                "LayoutSourceView.reshape: can only specify one unknown dim"
            )
        if not unknown:
            known = 1
            for d in shape_t:
                known *= d
            if known != self._size:
                raise ValueError(
                    f"LayoutSourceView.reshape: cannot reshape size "
                    f"{self._size} into {shape_t}"
                )
            return shape_t
        idx = unknown[0]
        known = 1
        for d in shape_t:
            if d != -1:
                known *= d
        if known == 0 or self._size % known != 0:
            raise ValueError(
                f"LayoutSourceView.reshape: cannot reshape size "
                f"{self._size} into {shape_t}"
            )
        return shape_t[:idx] + (self._size // known,) + shape_t[idx + 1:]

    def _resolve_getitem_shape(self, idx) -> tuple[int, ...]:
        """Compute the shape resulting from ``self[idx]`` without materialising.

        Supports int / slice / None tuples (the subset the mapper graph uses).
        For anything else we'd have to materialise to discover the shape.
        """
        if not isinstance(idx, tuple):
            idx = (idx,)
        result: list[int] = []
        dim_iter = iter(self._shape)
        for item in idx:
            if item is None:
                result.append(1)
                continue
            try:
                size = next(dim_iter)
            except StopIteration as exc:
                raise IndexError(
                    f"LayoutSourceView: too many indices for shape "
                    f"{self._shape}"
                ) from exc
            if isinstance(item, int):
                continue
            if isinstance(item, slice):
                start, stop, step = item.indices(size)
                if step > 0:
                    length = max(0, (stop - start + step - 1) // step)
                else:
                    length = max(0, (start - stop - step - 1) // (-step))
                result.append(length)
                continue
            raise TypeError(
                f"LayoutSourceView: unsupported index element {item!r}"
            )
        for remaining in dim_iter:
            result.append(remaining)
        return tuple(result)


# Module-level helpers


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
