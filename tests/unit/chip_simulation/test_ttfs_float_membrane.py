"""The TTFS executor's float membrane, a fixed function of a neuron's own nonzero terms (closure targeted pass, R1).

A neuron off the integer lattice (a raw input of 8-bit codes k/255 in float64, a float chip) takes the float
membrane of ``chip_simulation/ttfs/float_membrane.py``. Every input and weight is split into 16-bit windows of one
global power-of-two grid; each window pair's product sum is exact in float64 in any order, so a packing that moves,
duplicates or zero-pads a neuron's terms (the per-axon split's W+/W- rows, a co-resident soft core at an axon offset,
a fused core, the bias as a hardware lane or an always-on row) gives that neuron the same membrane bit for bit. The
summation order of the hard core's own axon layout does not: the necessity witness below finds BLAS moving a bit.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from mimarsinan.chip_simulation.ttfs.float_membrane import (
    MAX_TERMS,
    FloatMembraneRangeError,
    float_membrane,
)


def _draw(rng, batch=4, axons=60, neurons=12, *, integral=True):
    """Raw-input-like rows (8-bit codes k/255 in float64, continuous values) and sparse mixed-sign weights."""
    x = np.concatenate([rng.integers(0, 256, (batch // 2, axons)) / 255.0,
                        rng.random((batch - batch // 2, axons))])
    w = rng.integers(-7, 8, (neurons, axons)).astype(np.float64)
    if not integral:
        w = w * rng.random((neurons, axons))
    w[rng.random((neurons, axons)) < 0.3] = 0.0
    b = rng.integers(-20, 21, neurons).astype(np.float64)
    return x, w, b


def _bits(a):
    return np.asarray(a, dtype=np.float64).tobytes()


def _split_rows(x, w):
    """The per-axon split's layout: every input on two rows, its positive weights on the first, its negative on the
    second."""
    x2 = np.repeat(x, 2, axis=1)
    w2 = np.zeros((w.shape[0], 2 * w.shape[1]))
    w2[:, 0::2], w2[:, 1::2] = np.maximum(w, 0.0), np.minimum(w, 0.0)
    return x2, w2


class TestAPackingMovesNoBit:
    @pytest.mark.parametrize("integral", [True, False])
    @pytest.mark.parametrize("seed", range(6))
    def test_every_layout_of_a_neurons_terms_gives_its_membrane(self, seed, integral):
        rng = np.random.default_rng(seed)
        x, w, b = _draw(rng, integral=integral)
        want = _bits(float_membrane(x, w, b))
        perm = rng.permutation(x.shape[1])
        assert _bits(float_membrane(x[:, perm], w[:, perm], b)) == want, "axons in another order"
        x2, w2 = _split_rows(x, w)
        assert _bits(float_membrane(x2, w2, b)) == want, "W+/W- rows"
        order = np.r_[0:2 * x.shape[1]:2, 1:2 * x.shape[1]:2]
        assert _bits(float_membrane(x2[:, order], w2[:, order], b)) == want, "every W+ row before every W- row"
        # A co-resident soft core: its inputs (another magnitude range) on other axons, its neurons beside these.
        other = np.concatenate([rng.random((x.shape[0], 30)) * 1e6, rng.random((x.shape[0], 9)) * 1e-30], axis=1)
        xs = np.concatenate([other[:, :20], x, other[:, 20:]], axis=1)
        ws = np.zeros((w.shape[0] + 5, xs.shape[1]))
        ws[:w.shape[0], 20:20 + x.shape[1]] = w
        ws[w.shape[0]:] = rng.integers(-7, 8, (5, xs.shape[1]))
        bs = np.concatenate([b, np.zeros(5)])
        assert _bits(float_membrane(xs, ws, bs)[:, :w.shape[0]]) == want, "co-resident at an axon offset"
        # The bias as an always-on row of the core instead of a hardware bias lane.
        xo = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        wo = np.concatenate([w, b[:, None]], axis=1)
        assert _bits(float_membrane(xo, wo, None)) == want, "the bias as an always-on row"

    def test_the_layouts_summation_order_moves_a_bit(self):
        """The necessity witness: BLAS sums in the layout's order, and these layouts move some neuron's membrane."""
        moved = 0
        for seed in range(6):
            x, w, b = _draw(np.random.default_rng(seed))
            x2, w2 = _split_rows(x, w)
            moved += int(_bits(x @ w.T + b) != _bits(x2 @ w2.T + b))
        assert moved > 0


class TestTheMembraneValue:
    def test_an_exactly_representable_sum_is_returned_exactly(self):
        """float32 codes on integer weights sum exactly in any order: the membrane is that sum, and BLAS's."""
        rng = np.random.default_rng(11)
        k = rng.integers(0, 256, (3, 90))
        x = (k.astype(np.float32) / np.float32(255.0)).astype(np.float64)
        w = rng.integers(-7, 8, (20, 90)).astype(np.float64)
        b = rng.integers(-9, 10, 20).astype(np.float64)
        got = float_membrane(x, w, b)
        for i in range(3):
            for n in range(20):
                exact = sum((Fraction(float(x[i, a])) * int(w[n, a]) for a in range(90)), Fraction(int(b[n])))
                assert got[i, n] == float(exact)
        np.testing.assert_array_equal(got, x @ w.T + b)

    @pytest.mark.parametrize("integral", [True, False])
    def test_it_is_within_float_rounding_of_the_exact_membrane(self, integral):
        rng = np.random.default_rng(12)
        x, w, b = _draw(rng, integral=integral)
        got = float_membrane(x, w, b)
        for i in range(x.shape[0]):
            for n in range(w.shape[0]):
                terms = [Fraction(float(x[i, a])) * Fraction(float(w[n, a])) for a in range(x.shape[1])]
                exact = sum(terms, Fraction(float(b[n])))
                scale = sum(abs(t) for t in terms) + abs(Fraction(float(b[n])))
                assert abs(Fraction(float(got[i, n])) - exact) <= scale * Fraction(1, 2 ** 46)

    def test_no_input_gives_the_bias(self):
        w = np.array([[3.0, -2.0], [0.0, 5.0]])
        got = float_membrane(np.zeros((2, 2)), w, np.array([1.5, -0.25]))
        assert _bits(got) == _bits(np.array([[1.5, -0.25], [1.5, -0.25]]))

    def test_a_non_finite_input_propagates_as_in_the_dot_product(self):
        x = np.array([[0.5, np.nan, 0.25], [0.5, 0.75, 0.25]])
        w = np.array([[1.0, 0.0, 2.0], [3.0, 1.0, 0.0]])
        got = float_membrane(x, w, None)
        assert np.isnan(got[0]).all()
        assert _bits(got[1]) == _bits(np.array([1.0, 2.25]))


class TestARangeRefusedByName:
    def test_a_fan_in_past_the_exact_window_sum_is_refused(self):
        with pytest.raises(FloatMembraneRangeError):
            float_membrane(np.zeros((1, MAX_TERMS + 1)), np.zeros((1, MAX_TERMS + 1)), None)

    def test_a_window_product_below_the_subnormal_unit_is_refused(self):
        with pytest.raises(FloatMembraneRangeError):
            float_membrane(np.array([[1e-300, 0.5]]), np.array([[1e-300, 1.0]]), None)

    def test_a_tiny_input_on_a_zero_weight_is_admitted(self):
        got = float_membrane(np.array([[1e-300, 0.5]]), np.array([[0.0, 1e-300]]), None)
        assert got[0, 0] == 0.5 * 1e-300
