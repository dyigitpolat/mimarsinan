"""TTFSCycleActivation: differentiable single-spike TTFS == analytical == ReLU-floor-quant.

By the ReLU↔TTFS equivalence (Stanojevic et al. 2024), the single-spike cycle
activation's value equals ``ttfs_quantized_activation(x, θ, S)·θ``. The module's
forward must reproduce that exactly (so the cycle-based mode matches the analytical
quantized TTFS numerically), with a straight-through clamped-ReLU gradient.
"""

import torch

from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation


class TestMatchesAnalyticalKernel:
    def test_forward_equals_ttfs_quantized_scaled(self):
        S = 16
        scale = torch.tensor(1.3, dtype=torch.float64)
        act = TTFSCycleActivation(T=S, activation_scale=scale)
        x = torch.linspace(-0.5, 1.7, 223, dtype=torch.float64).reshape(-1, 1)
        out = act(x)
        ref = ttfs_quantized_activation(x, scale, S) * scale
        torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_preserves_relu_floor_quant_equivalence(self):
        S = 32
        scale = torch.tensor(2.0, dtype=torch.float64)
        act = TTFSCycleActivation(T=S, activation_scale=scale)
        x = torch.linspace(-1.0, 2.5, 257, dtype=torch.float64).reshape(-1, 1)
        out = act(x)
        r = (torch.relu(x) / scale).clamp(0.0, 1.0)
        relu_floor_quant = torch.floor(S * r) / S * scale
        torch.testing.assert_close(out, relu_floor_quant, rtol=0, atol=0)

    def test_negatives_zero_and_saturation(self):
        S = 8
        act = TTFSCycleActivation(T=S, activation_scale=1.0)
        x = torch.tensor([[-2.0], [0.0], [1.0], [5.0]], dtype=torch.float64)
        out = act(x)
        assert out[0].item() == 0.0
        assert out[1].item() == 0.0
        assert out[2].item() == 1.0
        assert out[3].item() == 1.0


class TestDifferentiable:
    def test_straight_through_gradient_in_active_range(self):
        act = TTFSCycleActivation(T=16, activation_scale=1.0)
        x = torch.tensor([[0.5]], dtype=torch.float64, requires_grad=True)
        act(x).sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().sum().item() > 0.0

    def test_gradient_zero_outside_clamp(self):
        act = TTFSCycleActivation(T=16, activation_scale=1.0)
        x = torch.tensor([[-1.0], [3.0]], dtype=torch.float64, requires_grad=True)
        act(x).sum().backward()
        # relu kills the negative; clamp saturates the >scale input -> no gradient.
        assert x.grad.abs().sum().item() == 0.0


def test_activation_type_is_ttfs():
    assert TTFSCycleActivation(T=4, activation_scale=1.0).activation_type == "TTFS"
