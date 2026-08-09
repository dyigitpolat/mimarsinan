"""[composition identity, n7 D3] the synchronized walk is the exact-QAT
training composition: staircase hops are theorem-equal to LIF hops (§16),
and the walk's boundary physics must be trainable (STE) — unmodeled it read
−6.6pp on the offloaded mixer with the QAT blind to all of it."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.autograd import LIFCountStaircaseFunction
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.torch_mapping.converter import convert_torch_model

T = 8


class _TwoSegLayerNorm(nn.Module):
    """input -> [Linear+ReLU] -> LayerNorm (host ComputeOp) -> [Linear+ReLU]."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 6)
        self.act1 = nn.ReLU()
        self.ln = nn.LayerNorm(6)
        self.fc2 = nn.Linear(6, 4)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.ln(x)
        return self.act2(self.fc2(x))


class _StaircaseActivation(nn.Module):
    """The AQ-stage QAT stand-in: the deployed LIF count staircase with STE."""

    def __init__(self, theta: float):
        super().__init__()
        self.theta = torch.tensor(float(theta))

    def forward(self, x):
        return LIFCountStaircaseFunction.apply(
            x, self.theta.to(x.device, x.dtype), T, True,
        )


def _flow(activation_for):
    torch.manual_seed(0)
    flow = convert_torch_model(
        _TwoSegLayerNorm().eval(), input_shape=(8,), num_classes=4,
    )
    for p in flow.get_perceptrons():
        p.set_activation_scale(1.0)
        p.activation = activation_for(p)
    return flow


def _lif_flow():
    def make(p):
        return LIFActivation(
            T=T, activation_scale=p.activation_scale, thresholding_mode="<",
        )
    return _flow(make)


def _stair_flow():
    return _flow(lambda p: _StaircaseActivation(1.0))


class TestStaircaseHopWalk:
    def test_staircase_hops_match_lif_hops_under_the_sync_walk(self):
        """The finalize no-op identity at walk level: swapping theorem-equal
        hop kernels must not change the synchronized composition."""
        torch.manual_seed(3)
        x = torch.rand(5, 8)
        with torch.no_grad():
            out_lif = chip_aligned_segment_forward(
                _lif_flow(), x, T, synchronized=True)
            out_stair = chip_aligned_segment_forward(
                _stair_flow(), x, T, synchronized=True)
        torch.testing.assert_close(out_stair, out_lif, atol=1e-6, rtol=0.0)

    def test_walk_gradient_reaches_layers_behind_host_boundaries(self):
        """The QAT must DESCEND the walk: pre-fix, the boundary round and the
        output comb were gradient-dead, so layers behind a host op trained
        blind to the deployed composition."""
        flow = _stair_flow()
        torch.manual_seed(4)
        x = torch.rand(6, 8)
        out = chip_aligned_segment_forward(flow, x, T, synchronized=True)
        out.sum().backward()
        first = flow.get_perceptrons()[0].layer.weight
        assert first.grad is not None and float(first.grad.abs().max()) > 0.0

    def test_walk_output_stays_on_the_wire_grid(self):
        """The output path is the uniform train's mean by identity:
        round(rate*T)/T * scale — STE must not move the forward."""
        with torch.no_grad():
            out = chip_aligned_segment_forward(
                _stair_flow(), torch.rand(4, 8), T, synchronized=True)
        grid = out * T  # scale == 1.0
        torch.testing.assert_close(grid, torch.round(grid), atol=1e-5, rtol=0.0)
