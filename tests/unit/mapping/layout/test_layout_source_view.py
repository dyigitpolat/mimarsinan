"""Tests for ``LayoutSourceView`` -- the lightweight currency of the shape-only
mapping path. Pins the duck-type surface that the mapper graph relies on."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_source_view import (
    LayoutSourceView,
    concat_source_views,
    node_ids_of,
    total_size,
)


def _producer_view(node_id: int, shape: tuple[int, ...]) -> LayoutSourceView:
    return LayoutSourceView.from_producer(producer_node_id=node_id, shape=shape)


def test_view_shape_size_ndim_dtype() -> None:
    v = _producer_view(7, (197, 768))
    assert v.shape == (197, 768)
    assert v.size == 197 * 768
    assert v.ndim == 2
    assert v.dtype == np.dtype(object)


def test_view_len_returns_first_dim() -> None:
    v = _producer_view(7, (197, 768))
    assert len(v) == 197


def test_view_flatten_returns_view_with_same_node_ids() -> None:
    v = _producer_view(7, (197, 768))
    flat = v.flatten()
    assert isinstance(flat, LayoutSourceView)
    assert flat.shape == (197 * 768,)
    assert flat.size == v.size
    assert node_ids_of(flat) == {7}


def test_view_reshape_returns_view() -> None:
    v = _producer_view(7, (197, 768))
    reshaped = v.reshape((197 * 768,))
    assert isinstance(reshaped, LayoutSourceView)
    assert reshaped.shape == (197 * 768,)

    reshaped2 = v.reshape((-1, 192))
    assert reshaped2.shape == (197 * 768 // 192, 192)


def test_view_getitem_slice_returns_view() -> None:
    v = _producer_view(7, (197, 768))
    sliced = v[:, 0]
    assert isinstance(sliced, LayoutSourceView)
    assert sliced.shape == (197,)
    assert node_ids_of(sliced) == {7}


def test_view_array_materialises_to_irsource_grid() -> None:
    v = _producer_view(7, (3, 4))
    materialised = np.asarray(v)
    assert isinstance(materialised, np.ndarray)
    assert materialised.dtype == object
    assert materialised.shape == (3, 4)
    flat = materialised.ravel()
    for idx, src in enumerate(flat):
        assert isinstance(src, IRSource)
        assert src.node_id == 7
        assert src.index == idx


def test_concat_two_views_merges_node_ids() -> None:
    v1 = _producer_view(7, (4,))
    v2 = _producer_view(9, (6,))
    cat = concat_source_views([v1, v2])
    assert isinstance(cat, LayoutSourceView)
    assert cat.shape == (10,)
    assert node_ids_of(cat) == {7, 9}


def test_concat_view_materialises_to_correct_irsources() -> None:
    v1 = _producer_view(7, (3,))
    v2 = _producer_view(9, (2,))
    cat = concat_source_views([v1, v2])
    flat = list(np.asarray(cat).ravel())
    assert flat[0] == IRSource(7, 0)
    assert flat[2] == IRSource(7, 2)
    assert flat[3] == IRSource(9, 0)
    assert flat[4] == IRSource(9, 1)


def test_node_ids_of_accepts_plain_array() -> None:
    arr = np.array(
        [IRSource(3, 0), IRSource(3, 1), IRSource(5, 0)], dtype=object,
    )
    assert node_ids_of(arr) == {3, 5}


def test_total_size_accepts_view_or_array() -> None:
    v = _producer_view(7, (10, 20))
    assert total_size(v) == 200

    arr = np.empty((6,), dtype=object)
    assert total_size(arr) == 6


def test_view_iteration_materialises() -> None:
    v = _producer_view(7, (3,))
    seq = list(v)
    assert seq == [IRSource(7, 0), IRSource(7, 1), IRSource(7, 2)]


def test_view_indexing_with_int_returns_irsource() -> None:
    v = _producer_view(7, (3, 4))
    src = v[1, 2]
    assert isinstance(src, IRSource)
    assert src.node_id == 7
    assert src.index == 1 * 4 + 2


def test_views_compare_shapes() -> None:
    a = _producer_view(7, (3, 4))
    b = _producer_view(9, (3, 4))
    assert a.shape == b.shape


def test_view_getitem_with_the_conv_patch_gather_stays_a_view() -> None:
    """The convolution mapper gathers its patches with broadcast INTEGER-ARRAY
    indices; a view that refuses that form makes every conv layout unmappable
    (it surfaced as a swallowed ``TypeError`` reported as layout infeasible)."""
    v = _producer_view(7, (2, 4, 4))
    h_idx = (np.arange(3) * 1)[:, None, None, None, None] + \
        np.arange(2)[None, None, None, :, None]
    w_idx = (np.arange(3) * 1)[None, :, None, None, None] + \
        np.arange(2)[None, None, None, None, :]
    c_idx = np.arange(2)[None, None, :, None, None]
    c_b, h_b, w_b = np.broadcast_arrays(c_idx, h_idx, w_idx)

    patches = v[c_b, h_b, w_b]

    assert isinstance(patches, LayoutSourceView)
    assert patches.shape == (3, 3, 2, 2, 2)
    assert node_ids_of(patches) == {7}
    assert np.array_equal(
        np.asarray(patches, dtype=object),
        np.asarray(v, dtype=object)[c_b, h_b, w_b],
    )


def test_view_getitem_advanced_shape_matches_numpy_exactly() -> None:
    """Separated advanced indices move the broadcast dims to the FRONT; the
    view must report the shape numpy itself would, never a guessed one."""
    v = _producer_view(3, (4, 5, 6))
    rows = np.array([[0, 1], [2, 3]])
    cols = np.array([[1, 2], [3, 4]])
    for idx in (
        (rows, slice(None), cols),
        (rows, cols),
        (slice(None), rows, 2),
        ([0, 2], slice(1, 3), slice(None)),
    ):
        got = v[idx]
        expected = np.asarray(v, dtype=object)[idx]
        assert got.shape == expected.shape, idx
        assert np.array_equal(np.asarray(got, dtype=object), expected), idx


def test_view_getitem_still_refuses_a_form_numpy_cannot_index() -> None:
    v = _producer_view(3, (4, 5))
    with pytest.raises(IndexError):
        v[object()]


def test_node_ids_excludes_off_and_negative_sentinels() -> None:
    """``IRSource`` with ``node_id < 0`` are sentinels (off/input/always-on)
    and must not be counted as upstream producers."""
    arr = np.array(
        [IRSource(3, 0), IRSource(-1, 0), IRSource(-2, 5), IRSource(7, 1)],
        dtype=object,
    )
    assert node_ids_of(arr) == {3, 7}
