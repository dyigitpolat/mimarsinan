"""Negative-shift calibration generalized over spiking modes via forward_fn."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.neg_shift_bias import (
    apply_negative_value_shifts,
    calibration_forward_for_mode,
)


class _TwoSegLayerNorm(nn.Module):
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


def _ttfs_flow(T):
    torch.manual_seed(0)
    flow = convert_torch_model(_TwoSegLayerNorm().eval(), (8,), 4, device="cpu")
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=T,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False),
        ))
    return flow.double()


def _lif_flow(T):
    torch.manual_seed(0)
    flow = convert_torch_model(_TwoSegLayerNorm().eval(), (8,), 4)
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    return flow


def _layernorm_op(flow):
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    ops = [n for n in repr_._exec_order if isinstance(n, ComputeOpMapper)
           and "LayerNorm" in type(getattr(n, "module", None)).__name__]
    assert len(ops) == 1
    return ops[0]


def test_calibration_forward_for_mode_lif_is_chip_aligned():
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    assert calibration_forward_for_mode("lif") is chip_aligned_segment_forward


@pytest.mark.parametrize("mode", ["rate", "bogus"])
def test_calibration_forward_for_mode_unsupported_raises(mode):
    with pytest.raises(NotImplementedError, match="negative_value_shift"):
        calibration_forward_for_mode(mode)


def test_apply_shifts_with_ttfs_cycle_based_forward():
    T = 8
    flow = _ttfs_flow(T)
    fwd = calibration_forward_for_mode("ttfs_cycle_based")
    shifts = apply_negative_value_shifts(
        flow, torch.rand(16, 8, dtype=torch.float64), T, forward_fn=fwd,
    )
    assert shifts, "LayerNorm boundary must derive a shift"
    ln = _layernorm_op(flow)
    assert ln in shifts
    assert getattr(ln, "_negative_shift", None) is not None
    consumer = flow.get_perceptrons()[1]
    assert getattr(consumer, "_neg_shift_baked", False)


@pytest.mark.parametrize("mode", ["ttfs", "ttfs_quantized"])
def test_apply_shifts_with_analytical_ttfs_forward(mode):
    T = 8
    flow = _ttfs_flow(T)
    fwd = calibration_forward_for_mode(mode)
    shifts = apply_negative_value_shifts(
        flow, torch.rand(16, 8, dtype=torch.float64), T, forward_fn=fwd,
    )
    assert shifts
    ln = _layernorm_op(flow)
    assert ln in shifts
    assert getattr(flow.get_perceptrons()[1], "_neg_shift_baked", False)


def test_apply_shifts_default_forward_is_lif():
    T = 8
    flow = _lif_flow(T)
    shifts = apply_negative_value_shifts(flow, torch.rand(16, 8), T)
    assert shifts
    assert _layernorm_op(flow) in shifts


def test_shift_value_matches_recorded_min_ttfs():
    """The derived shift equals max(0, -min) of the TTFS NF boundary values."""
    T = 8
    flow = _ttfs_flow(T)
    fwd = calibration_forward_for_mode("ttfs_cycle_based")
    x = torch.rand(16, 8, dtype=torch.float64)
    recorder: dict = {}
    with torch.no_grad():
        fwd(flow, x, T, compute_min_recorder=recorder)
    ln = _layernorm_op(flow)
    expected = torch.clamp(-recorder[ln], min=0.0)

    flow2 = _ttfs_flow(T)
    shifts = apply_negative_value_shifts(flow2, x, T, forward_fn=fwd)
    ln2 = _layernorm_op(flow2)
    torch.testing.assert_close(
        torch.as_tensor(shifts[ln2]), expected, atol=0.0, rtol=0.0,
    )
