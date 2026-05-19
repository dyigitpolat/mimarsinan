"""Tests for transformations.quantization_bounds."""

import pytest

from mimarsinan.transformations.quantization_bounds import quantization_bounds


class TestQuantizationBounds:
    def test_eight_bit_symmetric(self):
        q_min, q_max = quantization_bounds(8)
        assert q_min == -128
        assert q_max == 127

    def test_bits_must_be_positive(self):
        with pytest.raises(ValueError):
            quantization_bounds(0)
