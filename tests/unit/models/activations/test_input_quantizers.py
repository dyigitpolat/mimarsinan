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


class TestSigmaAwareChipInputQuantizer:
    """sigma-aware entry composition (boundary algebra I1): the trained op
    equals the deployed seam ``kappa*(snap(clamp((v+sigma)/kappa)) ) - sigma``;
    default sigma=None stays bitwise-identical to the historical form."""

    def test_default_none_is_bitwise_identical(self):
        S, kappa = 8, 1.7
        v = torch.linspace(-2.0, 2.5, 259, dtype=torch.float64)
        plain = ChipInputQuantizer(T=S, activation_scale=kappa)
        with_default = ChipInputQuantizer(
            T=S, activation_scale=kappa, negative_shift=None,
        )
        assert torch.equal(plain(v), with_default(v))

    def test_sigma_composition_matches_deployed_seam(self):
        S, kappa = 8, 1.7
        sigma = torch.tensor([0.4, 0.0, 0.9], dtype=torch.float64)
        v = torch.randn(16, 3, dtype=torch.float64)
        quantizer = ChipInputQuantizer(
            T=S,
            activation_scale=torch.tensor(kappa, dtype=torch.float64),
            negative_shift=sigma,
        )
        got = quantizer(v)
        wire = ((v + sigma) / kappa).clamp(0.0, 1.0)
        expected = torch.round(wire * S) / S * kappa - sigma
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=0.0)

    def test_sigma_makes_in_range_signed_values_identity_up_to_grid(self):
        S, kappa = 16, 1.7
        sigma = torch.tensor(0.4, dtype=torch.float64)
        v = torch.linspace(-0.4, kappa - 0.4, 33, dtype=torch.float64)
        quantizer = ChipInputQuantizer(
            T=S, activation_scale=kappa, negative_shift=sigma,
        )
        assert float((quantizer(v) - v).abs().max()) <= kappa / (2 * S) + 1e-12

    def test_ttfs_grid_subclass_inherits_sigma(self):
        S, kappa = 4, 2.0
        sigma = torch.tensor(0.5, dtype=torch.float64)
        v = torch.linspace(-0.5, 2.0, 41, dtype=torch.float64)
        quantizer = TTFSInputGridQuantizer(
            T=S,
            activation_scale=torch.tensor(kappa, dtype=torch.float64),
            negative_shift=sigma,
        )
        got = quantizer(v)
        wire = ((v + sigma) / kappa).clamp(0.0, 1.0)
        snapped = torch.from_numpy(ttfs_input_grid_quantize(wire.numpy(), S))
        expected = snapped * kappa - sigma
        torch.testing.assert_close(got, expected, atol=1e-12, rtol=0.0)
