"""ReLU↔TTFS equivalence hypothesis (foundation of train-ReLU / deploy-TTFS).

The analytical ``ttfs_quantized_activation`` must equal a floor-quantised clamped
ReLU: ``ttfs_quantized_activation(V, θ, S) == floor(S·clamp(relu(V)/θ, 0, 1)) / S``.
This is what lets a ReLU/clamp/quantise-trained model deploy as time-to-first-spike.
"""

import torch

from mimarsinan.models.nn.activations.autograd import StaircaseFunction
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation


def _floor_quant_relu(V, theta, S):
    r = (torch.relu(V) / theta).clamp(0.0, 1.0)
    return torch.floor(S * r) / S


class TestReluTtfsEquivalence:
    def test_matches_floor_quant_relu_on_grid(self):
        S = 16
        theta = torch.tensor(1.3, dtype=torch.float64)
        V = torch.linspace(-0.5, 1.6, 211, dtype=torch.float64).reshape(-1, 1)
        kernel = ttfs_quantized_activation(V, theta, S)
        ref = _floor_quant_relu(V, theta, S)
        torch.testing.assert_close(kernel, ref, rtol=0, atol=0)

    def test_matches_staircase_floor_quantizer(self):
        # ttfs_quantized == StaircaseFunction(clamp(relu(V)/θ, 0, 1), S): ties TTFS
        # to the exact floor quantiser the ReLU path would use (no half-LSB shift).
        S = 32
        theta = torch.tensor(2.0, dtype=torch.float64)
        V = torch.linspace(-1.0, 2.5, 257, dtype=torch.float64).reshape(-1, 1)
        kernel = ttfs_quantized_activation(V, theta, S)
        r = (torch.relu(V) / theta).clamp(0.0, 1.0)
        staircase = StaircaseFunction.apply(r, torch.tensor(float(S), dtype=torch.float64))
        torch.testing.assert_close(kernel, staircase, rtol=0, atol=0)

    def test_negative_and_zero_map_to_zero(self):
        S = 8
        theta = torch.tensor(1.0, dtype=torch.float64)
        V = torch.tensor([[-3.0], [-1e-9], [0.0]], dtype=torch.float64)
        out = ttfs_quantized_activation(V, theta, S)
        assert torch.all(out == 0.0)

    def test_saturates_at_one_for_V_ge_theta(self):
        S = 8
        theta = torch.tensor(1.0, dtype=torch.float64)
        V = torch.tensor([[1.0], [1.5], [10.0]], dtype=torch.float64)
        out = ttfs_quantized_activation(V, theta, S)
        assert torch.all(out == 1.0)
