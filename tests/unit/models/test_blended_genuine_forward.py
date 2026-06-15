"""Teacher->genuine output blend forward (opt-in TTFS ramp).

``BlendedGenuineForward`` is a picklable ``model.forward`` override computing
``(1 - rate) * teacher(x) + rate * genuine(x)``, where ``genuine`` is a lazily
built differentiable ``TTFSSegmentForward`` over the model and ``teacher`` is a
frozen snapshot. These tests fix the design: rate endpoints reproduce the two
pure forwards bit-exactly, the midpoint is the arithmetic mean, gradients flow
into the model (never the teacher), and the lazy executor is dropped on pickle.
"""

from __future__ import annotations

import copy
import pickle

import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.blended_genuine_forward import (
    BlendedGenuineForward,
)
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.tuning.teacher import freeze_module


class _TinyMLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(d_h, d_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        return self.act2(self.fc2(self.act1(self.fc1(x))))


def _convert(model, input_shape, num_classes):
    from mimarsinan.torch_mapping.converter import convert_torch_model

    return convert_torch_model(model, input_shape, num_classes, device="cpu")


def _install_ttfs(flow, S):
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


def _deployed_flow(S=8, d_in=6, d_h=5, d_out=4, seed=0):
    torch.manual_seed(seed)
    return _install_ttfs(_convert(_TinyMLP(d_in, d_h, d_out), (d_in,), d_out), S)


def _teacher_of(flow):
    return freeze_module(copy.deepcopy(flow))


def _blend(flow, S=8):
    return BlendedGenuineForward(flow, _teacher_of(flow), S)


def test_rate_zero_is_teacher_exact():
    S = 8
    flow = _deployed_flow(S)
    teacher = _teacher_of(flow)
    blend = BlendedGenuineForward(flow, teacher, S)
    blend.rate = 0.0
    x = torch.randn(3, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = teacher(x)
        got = blend(x)
    assert torch.equal(got, expected)


def test_rate_one_is_freshly_built_genuine_exact():
    S = 8
    flow = _deployed_flow(S)
    blend = BlendedGenuineForward(flow, _teacher_of(flow), S)
    blend.rate = 1.0
    x = torch.randn(3, 6, dtype=torch.float64)
    with torch.no_grad():
        expected = TTFSSegmentForward(flow.get_mapper_repr(), S)(x)
        got = blend(x)
    assert torch.equal(got, expected)


def test_rate_half_is_midpoint():
    S = 8
    flow = _deployed_flow(S)
    teacher = _teacher_of(flow)
    blend = BlendedGenuineForward(flow, teacher, S)
    x = torch.randn(3, 6, dtype=torch.float64)
    with torch.no_grad():
        t = teacher(x)
        g = TTFSSegmentForward(flow.get_mapper_repr(), S)(x)
        blend.rate = 0.5
        got = blend(x)
    assert torch.equal(got, 0.5 * t + 0.5 * g)


def test_rate_is_read_live_each_call():
    """Mutating ``.rate`` between calls re-reads it (no captured snapshot)."""
    S = 8
    flow = _deployed_flow(S)
    teacher = _teacher_of(flow)
    blend = BlendedGenuineForward(flow, teacher, S)
    x = torch.randn(2, 6, dtype=torch.float64)
    with torch.no_grad():
        t = teacher(x)
        g = TTFSSegmentForward(flow.get_mapper_repr(), S)(x)
        blend.rate = 0.25
        a = blend(x)
        blend.rate = 0.75
        b = blend(x)
    assert torch.equal(a, 0.75 * t + 0.25 * g)
    assert torch.equal(b, 0.25 * t + 0.75 * g)


def test_grad_flows_to_model_not_teacher():
    S = 8
    flow = _deployed_flow(S)
    teacher = _teacher_of(flow)
    blend = BlendedGenuineForward(flow, teacher, S)
    blend.rate = 0.5
    x = torch.randn(3, 6, dtype=torch.float64)
    out = blend(x)
    out.sum().backward()
    for p in flow.get_perceptrons():
        g = p.layer.weight.grad
        assert g is not None and torch.isfinite(g).all()
    assert flow.get_perceptrons()[0].layer.weight.grad.abs().sum() > 0
    for tp in teacher.parameters():
        assert tp.grad is None


def test_picklable_drops_lazy_executor():
    S = 8
    flow = _deployed_flow(S)
    blend = BlendedGenuineForward(flow, _teacher_of(flow), S)
    blend.rate = 1.0
    x = torch.randn(2, 6, dtype=torch.float64)
    with torch.no_grad():
        _ = blend(x)  # build the lazy genuine executor
    assert blend._executor is not None
    restored = pickle.loads(pickle.dumps(blend))
    assert restored._executor is None
    # round-trip still runs and matches a fresh genuine forward
    with torch.no_grad():
        expected = TTFSSegmentForward(restored.model.get_mapper_repr(), S)(x)
        got = restored(x)
    assert torch.equal(got, expected)
