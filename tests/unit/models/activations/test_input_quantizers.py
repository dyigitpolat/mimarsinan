"""Encoding-layer input quantizers: STE grid snap vs the numpy wire SSOT."""

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_input_grid_quantize
from mimarsinan.models.nn.activations.autograd import (
    ChipInputQuantizer,
    TTFSInputGridQuantizer,
)


def _sweep_values(S: int) -> torch.Tensor:
    """Dense [0,1] sweep including exact grid points and rounding ties."""
    grid = torch.arange(S + 1, dtype=torch.float64) / S
    ties = (torch.arange(S, dtype=torch.float64) + 0.5) / S
    dense = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    return torch.cat([grid, ties, dense])


class TestTTFSInputGridQuantizerMatchesWireSSOT:
    @pytest.mark.parametrize("S", [1, 2, 3, 4, 8, 16])
    def test_matches_ttfs_input_grid_quantize(self, S):
        x = _sweep_values(S)
        quantizer = TTFSInputGridQuantizer(T=S, activation_scale=1.0)
        got = quantizer(x).numpy()
        expected = ttfs_input_grid_quantize(x.numpy(), S)
        np.testing.assert_array_equal(
            got, expected,
            err_msg=f"TTFSInputGridQuantizer diverges from the numpy wire SSOT at S={S}",
        )

    @pytest.mark.parametrize("S", [4, 16])
    def test_scale_normalization_round_trip(self, S):
        scale = 2.5
        x = _sweep_values(S) * scale
        quantizer = TTFSInputGridQuantizer(T=S, activation_scale=scale)
        got = quantizer(x).numpy()
        expected = ttfs_input_grid_quantize((x / scale).numpy(), S) * scale
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_out_of_range_inputs_clamp_like_the_wire(self):
        S = 4
        x = torch.tensor([-0.5, 1.5, 2.0], dtype=torch.float64)
        quantizer = TTFSInputGridQuantizer(T=S, activation_scale=1.0)
        got = quantizer(x).numpy()
        expected = ttfs_input_grid_quantize(x.numpy(), S)
        np.testing.assert_array_equal(got, expected)

    def test_ste_gradient_passes_through(self):
        x = torch.linspace(0.05, 0.95, 7, dtype=torch.float64, requires_grad=True)
        quantizer = TTFSInputGridQuantizer(T=4, activation_scale=1.0)
        quantizer(x).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones(7), rtol=0, atol=1e-12)


class TestChipInputQuantizer:
    def test_rounds_to_chip_rate_grid(self):
        quantizer = ChipInputQuantizer(T=4, activation_scale=1.0)
        x = torch.tensor([0.0, 0.2, 0.3, 0.8, 1.0], dtype=torch.float64)
        expected = torch.round(x * 4) / 4
        torch.testing.assert_close(quantizer(x), expected, rtol=0, atol=0)

    def test_ste_gradient_passes_through(self):
        x = torch.linspace(0.05, 0.95, 7, dtype=torch.float64, requires_grad=True)
        quantizer = ChipInputQuantizer(T=4, activation_scale=1.0)
        quantizer(x).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), np.ones(7), rtol=0, atol=1e-12)
