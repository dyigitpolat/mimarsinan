"""TTFSCountStaircaseFunction — the θ-in-loop TTFS ceil staircase (the TTFS
analog of LIFCountStaircaseFunction for the generic exact-QAT).

Forward is bit-exact to the deployed ``ttfs_quantized_staircase`` kernel (the
exact-QAT identity); backward is the SAME clamp-gated STE + in-band LSQ
θ-gradient the LIF function uses (shared ``_gated_lsq_backward``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.autograd import (
    LIF_EXACT_QAT_THETA_FLOOR,
    TTFSCountStaircaseFunction,
)
from mimarsinan.models.nn.decorators.clamp_quantize import TTFSCountStaircaseDecorator
from mimarsinan.models.spiking.wire_semantics import ttfs_quantized_staircase

# Grid ties (k/S), the dead zone (<1/S), interiors, saturation (>=1), negatives.
_SWEEP = [-0.5, 0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.96875, 1.0, 1.25, 2.0]


def _ref(z, theta, S, half=False):
    th = torch.as_tensor(theta, dtype=torch.float64)
    one = torch.ones((), dtype=torch.float64)
    return th * ttfs_quantized_staircase(z / th, one, S, comparator_half_step=half)


class TestForwardBitExact:
    @pytest.mark.parametrize("S", [4, 8, 16, 32])
    @pytest.mark.parametrize("theta", [0.5, 1.0, 2.0])
    def test_matches_deployed_kernel(self, S, theta):
        z = torch.tensor(_SWEEP, dtype=torch.float64)
        th = torch.tensor(theta, dtype=torch.float64)
        out = TTFSCountStaircaseFunction.apply(z, th, S, False)
        assert torch.equal(out, _ref(z, theta, S))

    def test_comparator_half_step_variant(self):
        z = torch.tensor(_SWEEP, dtype=torch.float64)
        th = torch.tensor(1.0, dtype=torch.float64)
        out = TTFSCountStaircaseFunction.apply(z, th, 8, True)
        assert torch.equal(out, _ref(z, 1.0, 8, half=True))

    def test_theta_positivity_floor(self):
        z = torch.tensor([0.5], dtype=torch.float64)
        th = torch.tensor(1e-9, dtype=torch.float64)  # below the floor
        out = TTFSCountStaircaseFunction.apply(z, th, 8, False)
        assert torch.equal(out, _ref(z, LIF_EXACT_QAT_THETA_FLOOR, 8))


class TestGradient:
    def test_ste_gated_to_the_in_band_region(self):
        # grad_z passes (STE) only where 0 < r < 1; dead (r<=0) and saturated
        # (r>=1) are clamped to zero gradient.
        z = torch.tensor([-0.2, 0.0, 0.3, 0.99, 1.0, 1.5],
                         dtype=torch.float64, requires_grad=True)
        th = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        TTFSCountStaircaseFunction.apply(z, th, 8, False).sum().backward()
        r = z.detach()
        inband = ((r > 0) & (r < 1)).double()
        assert torch.allclose(z.grad, inband)

    def test_theta_lsq_gradient(self):
        z = torch.tensor([0.3, 0.7, 1.4], dtype=torch.float64, requires_grad=True)
        th = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        TTFSCountStaircaseFunction.apply(z, th, 8, False).sum().backward()
        r = z.detach()
        q = ttfs_quantized_staircase(r, torch.ones((), dtype=torch.float64), 8)
        inband = ((r > 0) & (r < 1)).double()
        assert torch.allclose(th.grad, (q - r * inband).sum())

    def test_per_channel_theta_reduces_to_channel_shape(self):
        z = torch.tensor([[0.3, 0.7], [0.5, 0.9]], dtype=torch.float64, requires_grad=True)
        th = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
        TTFSCountStaircaseFunction.apply(z, th, 8, False).sum().backward()
        assert th.grad.shape == (2,)


class TestDecorator:
    def test_applies_the_function_and_trains_theta(self):
        scale = nn.Parameter(torch.tensor(1.0))
        dec = TTFSCountStaircaseDecorator(8, scale)
        x = torch.tensor([0.3, 0.7], requires_grad=True)
        y = dec.output_transform(x)
        assert torch.allclose(y.double(), _ref(x.detach().double(), 1.0, 8))
        y.sum().backward()
        assert scale.grad is not None  # θ trained in-loop (not applied around the STE)

    def test_rejects_mismatched_vector_theta(self):
        scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        dec = TTFSCountStaircaseDecorator(8, scale)
        x = torch.zeros(2, 2)  # last dim 2 != theta numel 3
        with pytest.raises(ValueError, match="theta"):
            dec.output_transform(x)
