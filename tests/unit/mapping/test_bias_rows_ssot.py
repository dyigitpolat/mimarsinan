"""The bias-row splitting SSOT: the computed bound, the scale recovery, and the
exactness contract the k always-on rows owe the deployed integer grid.

A parameter-encoded bias is a core-matrix ROW bound by the ±q_max weight
register. One row forces the shared per-perceptron grid to scale to the bias
(measured: max|b|/max|w| = 7.04 collapsed a 4-bit MLP to 3 weight levels and
99.6% zeros). k rows let the grid come from max|w| alone, and k is not a
tuning knob: it is ``ceil(max|b| * s_w / q_max)``.
"""

import numpy as np
import pytest
import torch

from mimarsinan.mapping.support.bias_rows import (
    bias_row_bound,
    bias_rows_from_scales,
    param_encoded_bias_rows,
    split_bias_row_values,
)


class TestBiasRowBound:
    def test_a_bias_inside_one_register_needs_one_row(self):
        assert bias_row_bound(b_max=0.5, weight_scale=10.0, q_max=7.0) == 1

    def test_the_bound_is_the_ceiling_of_the_register_demand(self):
        # 7.04 registers of bias demand -> 8 rows (the measured platform-J hop).
        assert bias_row_bound(b_max=7.04, weight_scale=7.0, q_max=7.0) == 8

    def test_an_exact_multiple_does_not_round_up(self):
        assert bias_row_bound(b_max=3.0, weight_scale=7.0, q_max=7.0) == 3

    def test_float_noise_below_the_lattice_does_not_buy_a_row(self):
        assert bias_row_bound(b_max=3.0 + 1e-12, weight_scale=7.0, q_max=7.0) == 3

    def test_a_zero_bias_still_occupies_its_row(self):
        assert bias_row_bound(b_max=0.0, weight_scale=10.0, q_max=7.0) == 1

    @pytest.mark.parametrize("q_max,weight_scale", [(0.0, 10.0), (7.0, 0.0), (-1.0, 10.0)])
    def test_a_degenerate_grid_fails_loud(self, q_max, weight_scale):
        with pytest.raises(ValueError):
            bias_row_bound(b_max=1.0, weight_scale=weight_scale, q_max=q_max)


class TestRowsFromScales:
    def test_a_shared_grid_recovers_one_row(self):
        assert bias_rows_from_scales(torch.tensor(40.0), torch.tensor(40.0)) == 1

    def test_an_absent_bias_scale_recovers_one_row(self):
        assert bias_rows_from_scales(None, torch.tensor(40.0)) == 1

    def test_the_integer_ratio_snap_carries_the_row_count(self):
        assert bias_rows_from_scales(torch.tensor(40.0 / 8), torch.tensor(40.0)) == 8

    def test_a_non_integer_ratio_fails_loud(self):
        with pytest.raises(ValueError, match="integer row count"):
            bias_rows_from_scales(torch.tensor(13.0), torch.tensor(40.0), name="fc")


class TestPlatformView:
    """The installed grids ARE the authority — no capability flag reaches here,
    so a mapper, a shape-only layout walk and a verifier cannot disagree."""

    def test_a_param_encoded_platform_reads_k_off_the_scales(self):
        assert param_encoded_bias_rows(
            torch.tensor(5.0), torch.tensor(40.0), hardware_bias=False, name="fc"
        ) == 8

    def test_a_shared_grid_reads_the_legacy_single_row(self):
        assert param_encoded_bias_rows(
            torch.tensor(40.0), torch.tensor(40.0), hardware_bias=False, name="fc"
        ) == 1

    def test_an_on_chip_bias_lane_spends_no_row(self):
        assert param_encoded_bias_rows(
            torch.tensor(5.0), torch.tensor(40.0), hardware_bias=True, name="fc"
        ) == 1


class TestSplitExactness:
    def test_the_rows_sum_to_the_bias(self):
        b = np.array([0.75, -0.5, 0.25])
        rows = split_bias_row_values(b, 4)
        assert rows.shape == (4, 3)
        np.testing.assert_allclose(rows.sum(axis=0), b, rtol=0, atol=1e-15)

    def test_one_row_is_the_legacy_bias_row(self):
        b = np.array([0.75, -0.5, 0.25])
        np.testing.assert_array_equal(split_bias_row_values(b, 1), b.reshape(1, 3))

    def test_the_integer_weights_sum_exactly_to_the_deployed_bias(self):
        """THE exactness bar: with the two-scale install the k rows' rounded
        integer weights sum to round(b_j * s_w) with no residue."""
        q_max, k = 7.0, 8
        weight_scale = 40.0
        bias_scale = weight_scale / k
        bias_ints = np.array([7.0, -6.0, 3.0, 0.0])
        b = bias_ints / bias_scale

        rows = split_bias_row_values(b, k)
        row_ints = np.round(rows * weight_scale)

        assert np.abs(row_ints).max() <= q_max
        np.testing.assert_array_equal(row_ints.sum(axis=0), np.round(b * weight_scale))
        np.testing.assert_array_equal(row_ints.sum(axis=0), bias_ints * k)

    def test_zero_rows_fail_loud(self):
        with pytest.raises(ValueError):
            split_bias_row_values(np.array([1.0]), 0)
