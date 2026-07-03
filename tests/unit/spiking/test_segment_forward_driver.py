"""Characterization locks: unified SegmentForwardDriver == the parallel walks it replaces."""

from __future__ import annotations

import pickle

import pytest
import torch
import torch.nn as nn

from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore

from mimarsinan.spiking.segment_forward import (
    SegmentForwardDriver,
    LifSegmentPolicy,
    TtfsSegmentPolicy,
)


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


class _TinyMLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(d_h, d_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        return self.act2(self.fc2(self.act1(self.fc1(x))))


def _lif_flow(model, input_shape, num_classes, T):
    torch.manual_seed(0)
    flow = convert_torch_model(model.eval(), input_shape=input_shape, num_classes=num_classes)
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    return flow


def _ttfs_flow(model, input_shape, num_classes, S, *, second_is_encoding=False):
    torch.manual_seed(0)
    flow = convert_torch_model(model.eval(), input_shape, num_classes, device="cpu")
    if second_is_encoding:
        flow.get_perceptrons()[1].is_encoding_layer = True
    for p in flow.get_perceptrons():
        p.set_activation(TTFSActivation(
            T=S,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode="<=",
            encoding=getattr(p, "is_encoding_layer", False),
        ))
    return flow.double()


def test_lif_driver_equals_chip_aligned_forward():
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    T = 8
    flow = _lif_flow(_TwoSegLayerNorm(), (8,), 4, T)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy())
    x = torch.rand(4, 8)
    with torch.no_grad():
        expected = chip_aligned_segment_forward(flow, x, T)
        actual = driver(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_lif_driver_compute_min_recorder_matches_chip_aligned():
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    T = 8
    flow = _lif_flow(_TwoSegLayerNorm(), (8,), 4, T)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy())
    x = torch.rand(4, 8)
    rec_expected: dict = {}
    rec_actual: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(flow, x, T, compute_min_recorder=rec_expected)
        driver(x, compute_min_recorder=rec_actual)
    assert set(rec_actual) == set(rec_expected) and rec_expected
    for node in rec_expected:
        torch.testing.assert_close(rec_actual[node], rec_expected[node], atol=0.0, rtol=0.0)


def test_lif_driver_applies_negative_shift_like_chip_aligned():
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
    from mimarsinan.mapping.support.bias_compensation import apply_negative_value_shifts

    T = 8
    flow = _lif_flow(_TwoSegLayerNorm(), (8,), 4, T)
    shifts = apply_negative_value_shifts(flow, torch.rand(16, 8), T)
    assert shifts, "calibration must derive a shift for the LayerNorm boundary"
    driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy())
    x = torch.rand(4, 8)
    with torch.no_grad():
        expected = chip_aligned_segment_forward(flow, x, T)
        actual = driver(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def _mmixcore_lif_flow(T):
    torch.manual_seed(0)
    m = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    ).eval()
    flow = convert_torch_model(m, input_shape=(1, 28, 28), num_classes=10)
    flow.eval()
    repr_ = flow.get_mapper_repr()
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    mark_encoding_layers(repr_)
    return flow


def test_lif_driver_equals_chip_aligned_forward_mmixcore():
    from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward

    T = 4
    flow = _mmixcore_lif_flow(T)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy())
    x = torch.rand(1, 1, 28, 28)
    with torch.no_grad():
        expected = chip_aligned_segment_forward(flow, x, T)
        actual = driver(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_lif_partition_single_segment_mmixcore():
    """mmixcore is one neural segment (its compute ops sit only at the graph ends)."""
    T = 4
    flow = _mmixcore_lif_flow(T)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy())
    assert len(driver.segments) == 1


def test_ttfs_driver_equals_ttfs_segment_forward():
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

    S = 8
    flow = _ttfs_flow(_TinyMLP(6, 5, 4), (6,), 4, S)
    legacy = TTFSSegmentForward(flow.get_mapper_repr(), S)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), S, TtfsSegmentPolicy())
    x = torch.randn(3, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = legacy(x)
        actual = driver(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_ttfs_driver_equals_ttfs_segment_forward_two_segments():
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward

    S = 8
    flow = _ttfs_flow(_TinyMLP(6, 5, 4), (6,), 4, S, second_is_encoding=True)
    legacy = TTFSSegmentForward(flow.get_mapper_repr(), S)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), S, TtfsSegmentPolicy())
    assert len(driver.segments) == 2
    x = torch.randn(3, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = legacy(x)
        actual = driver(x)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_ttfs_driver_is_differentiable():
    S = 8
    flow = _ttfs_flow(_TinyMLP(6, 5, 4), (6,), 4, S)
    driver = SegmentForwardDriver(flow.get_mapper_repr(), S, TtfsSegmentPolicy())
    x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    out = driver(x)
    out.sum().backward()
    for p in flow.get_perceptrons():
        g = p.layer.weight.grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_tuner_forward_installs_picklable():
    from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import _SegmentSpikeForward
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import _ChipAlignedNFForward

    S = 8
    ttfs_flow = _ttfs_flow(_TinyMLP(6, 5, 4), (6,), 4, S)
    fwd = _SegmentSpikeForward(ttfs_flow, S)
    x = torch.randn(3, 6, dtype=torch.float64)
    with torch.no_grad():
        before = fwd(x)
    restored = pickle.loads(pickle.dumps(fwd))
    with torch.no_grad():
        after = restored(x)
    torch.testing.assert_close(after, before, atol=0.0, rtol=0.0)

    T = 4
    lif_flow = _lif_flow(_TwoSegLayerNorm(), (8,), 4, T)
    nf = _ChipAlignedNFForward(lif_flow, T)
    xl = torch.rand(2, 8)
    with torch.no_grad():
        out_before = nf(xl)
    nf_restored = pickle.loads(pickle.dumps(nf))
    with torch.no_grad():
        out_after = nf_restored(xl)
    torch.testing.assert_close(out_after, out_before, atol=0.0, rtol=0.0)
