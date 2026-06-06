"""TTFS-grid input quantization: the encode→decode round-trip SSOT helper."""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
    ttfs_input_grid_quantize,
    ttfs_spike_time,
)


@pytest.mark.parametrize("s", [4, 16, 60])
def test_roundtrip_matches_encode(s):
    x = np.linspace(-0.25, 1.25, 301)
    k = ttfs_spike_time(x, s)
    expected = np.where(k < s, (s - k) / float(s), 0.0)
    np.testing.assert_array_equal(ttfs_input_grid_quantize(x, s), expected)


@pytest.mark.parametrize("s", [4, 16])
def test_on_grid_identity(s):
    x = np.arange(s + 1) / float(s)
    np.testing.assert_array_equal(ttfs_input_grid_quantize(x, s), x)


def test_idempotent():
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.1, 1.1, size=(2, 50))
    q = ttfs_input_grid_quantize(x, 16)
    np.testing.assert_array_equal(ttfs_input_grid_quantize(q, 16), q)


def test_off_grid_rounds_to_nearest_level():
    # S=4: x=0.6 -> k=round(1.6)=2 -> 0.5; x=0.7 -> k=round(1.2)=1 -> 0.75.
    np.testing.assert_array_equal(
        ttfs_input_grid_quantize(np.array([0.6, 0.7]), 4),
        np.array([0.5, 0.75]),
    )


def test_out_of_range_clamps_like_encoder():
    q = ttfs_input_grid_quantize(np.array([-0.5, 0.0, 1.0, 2.0]), 8)
    np.testing.assert_array_equal(q, np.array([0.0, 0.0, 1.0, 1.0]))
