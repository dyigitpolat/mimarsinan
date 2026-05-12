"""Golden-output regression for the lifted ``_uniform_rate_encode`` helper.

The same function is consumed by ``LavaLoihiRunner`` and (next) by the
SANA-FE runner.  Any drift in its output would silently change the
HCM-vs-Lava parity surface — so we pin a handful of small reference
inputs here.
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.chip_simulation._spike_encoding import uniform_rate_encode


def test_uniform_rate_encode_returns_shape_N_D_T():
    out = uniform_rate_encode(np.zeros((3, 5), dtype=np.float32), T=7)
    assert out.shape == (3, 5, 7)
    assert out.dtype == np.float32


def test_uniform_rate_encode_zero_rates_produce_no_spikes():
    out = uniform_rate_encode(np.zeros((2, 4), dtype=np.float32), T=10)
    assert out.sum() == 0.0


def test_uniform_rate_encode_full_rates_produce_full_train():
    """rate == 1.0 ⇒ a spike at every cycle."""
    out = uniform_rate_encode(np.ones((1, 3), dtype=np.float32), T=8)
    assert np.array_equal(out, np.ones((1, 3, 8), dtype=np.float32))


def test_uniform_rate_encode_half_rate_produces_T_over_2_spikes():
    """rate == 0.5, T == 8 ⇒ exactly 4 spikes per (sample, dim)."""
    out = uniform_rate_encode(np.full((1, 3), 0.5, dtype=np.float32), T=8)
    per_dim = out.sum(axis=2)
    np.testing.assert_array_equal(per_dim, np.full((1, 3), 4.0))


def test_uniform_rate_encode_clips_negative_rates_to_zero():
    out = uniform_rate_encode(np.full((1, 2), -0.3, dtype=np.float32), T=5)
    assert out.sum() == 0.0


def test_uniform_rate_encode_clips_super_one_rates_to_one():
    """Anything > 1.0 saturates at "fire every cycle"."""
    out = uniform_rate_encode(np.full((1, 2), 1.7, dtype=np.float32), T=4)
    assert np.array_equal(out, np.ones((1, 2, 4), dtype=np.float32))


def test_uniform_rate_encode_byte_identical_to_lava_runner_inline_copy():
    """The Lava runner imports the same helper — verify it still re-exports.

    This guards the helper-lift: ``LavaLoihiRunner`` must keep calling the
    canonical implementation, not a stale local one.
    """
    from mimarsinan.chip_simulation import lava_loihi_runner

    rates = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(2, 6)
    ours = uniform_rate_encode(rates, T=16)
    theirs = lava_loihi_runner._uniform_rate_encode(rates, T=16)
    np.testing.assert_array_equal(ours, theirs)
