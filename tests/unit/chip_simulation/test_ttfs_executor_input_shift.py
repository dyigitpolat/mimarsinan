"""Numpy segment-input shift application for the analytical TTFS contract path."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    apply_input_shifts_numpy,
)


def _src(node_id, offset, size):
    return SimpleNamespace(node_id=node_id, offset=offset, size=size)


def test_apply_input_shifts_numpy_adds_per_slice():
    seg_in = np.zeros((2, 5), dtype=np.float64)
    seg_in[:, 3] = -0.4
    input_map = [_src(7, 0, 3), _src(9, 3, 2)]
    shifts = {9: np.array([0.4, 0.1])}
    out = apply_input_shifts_numpy(input_map, seg_in, shifts)
    np.testing.assert_allclose(out[:, :3], 0.0)
    np.testing.assert_allclose(out[:, 3], 0.0)
    np.testing.assert_allclose(out[:, 4], 0.1)


def test_apply_input_shifts_numpy_identity_when_empty():
    seg_in = np.random.default_rng(0).normal(size=(3, 4))
    input_map = [_src(1, 0, 4)]
    assert apply_input_shifts_numpy(input_map, seg_in, {}) is seg_in
    assert apply_input_shifts_numpy(input_map, seg_in, None) is seg_in


def test_apply_input_shifts_numpy_does_not_mutate_input():
    seg_in = np.zeros((1, 2), dtype=np.float64)
    input_map = [_src(5, 0, 2)]
    out = apply_input_shifts_numpy(input_map, seg_in, {5: np.array([1.0, 2.0])})
    np.testing.assert_allclose(seg_in, 0.0)
    np.testing.assert_allclose(out, [[1.0, 2.0]])
