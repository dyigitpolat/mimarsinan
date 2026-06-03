"""Segment-aware TTFS spike-train forward (pre-mapping, differentiable).

``TTFSSegmentForward`` walks a converted model's exec graph and runs each neural
segment as a genuine single-spike sim (encoding layer value->spike, interior
cascade, decode at the boundary), with value-domain compute ops between segments
-- the trainable analog of the deployed ``SpikingHybridCoreFlow``. These tests
fix that design: correct segment partition, encode/decode round-trip on a real
converted graph, value-domain compute-op handling, and differentiability through
the whole forward.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.training.ttfs_segment_forward import (
    TTFSSegmentForward,
    partition_perceptron_segments,
)


def _convert(model, input_shape, num_classes):
    from mimarsinan.torch_mapping.converter import convert_torch_model
    return convert_torch_model(model, input_shape, num_classes, device="cpu")


def _install_ttfs(flow, S):
    """Set each chip perceptron's activation to a TTFSActivation matching its role."""
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


class _TinyMLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(d_h, d_out)
        self.act2 = nn.ReLU()

    def forward(self, x):
        return self.act2(self.fc2(self.act1(self.fc1(x))))


def test_partition_groups_cascaded_perceptrons_one_segment():
    flow = _convert(_TinyMLP(6, 5, 4), (6,), 4)
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    seg_of = partition_perceptron_segments(repr_._exec_order, repr_._deps)
    # Two stacked FCs (fc1 source = input, fc2 source = fc1) -> one cascade segment.
    assert len(seg_of) == 2
    assert len(set(seg_of.values())) == 1
    # Exactly one encoding layer (the entry).
    enc = [n for n in seg_of if getattr(n.perceptron, "is_encoding_layer", False)]
    assert len(enc) == 1


def test_single_perceptron_matches_analytical_ttfs_kernel():
    """One encoding perceptron, decoded at the boundary, must equal the canonical
    analytical TTFS kernel (floor-quantised clamped ReLU * scale)."""
    from mimarsinan.models.nn.activations import TTFSCycleActivation

    S = 16
    torch.manual_seed(0)
    flow = _install_ttfs(_convert(nn.Sequential(nn.Linear(5, 3), nn.ReLU()), (5,), 3), S)
    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(2.0))
        p.activation.activation_scale = p.activation_scale
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    x = torch.randn(4, 5, dtype=torch.float64)
    with torch.no_grad():
        out = drv(x)
        p = flow.get_perceptrons()[0]
        kernel = TTFSCycleActivation(T=S, activation_scale=p.activation_scale, thresholding_mode="<=")
        expected = kernel(p.layer(x.double()))
    np.testing.assert_allclose(out.numpy(), expected.numpy(), atol=1e-9)


def test_cascade_depth_increments_per_perceptron_hop():
    """Cascade latency = perceptron-hops from the entry (transparent ops add none)."""
    S = 8
    flow = _install_ttfs(_convert(_TinyMLP(6, 5, 4), (6,), 4), S)
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    (seg_nodes,) = drv._segments.values()           # single cascade segment
    depth = drv._segment_depths(seg_nodes)
    perceptron_depths = sorted(d for n, d in depth.items() if n.__class__.__name__.startswith("Perceptron"))
    assert perceptron_depths == [0, 1]               # fc1 entry @0, fc2 @1 (one hop)


def test_segment_forward_is_differentiable():
    S = 8
    flow = _install_ttfs(_convert(_TinyMLP(6, 5, 4), (6,), 4), S)
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    out = drv(x)
    assert out.shape == (3, 4)
    out.sum().backward()
    for p in flow.get_perceptrons():
        g = p.layer.weight.grad
        assert g is not None and torch.isfinite(g).all()
    # gradient reaches the earliest (encoding) layer's weights
    assert flow.get_perceptrons()[0].layer.weight.grad.abs().sum() > 0


def test_encoding_cut_creates_sequential_segments():
    """An is_encoding_layer perceptron consumes a *decoded value* -> the partition
    cuts at its input, producing two sequential segments (upstream decodes, the
    encoding layer re-encodes). Validates the value<->spike boundary + ordering."""
    S = 8
    flow = _convert(_TinyMLP(6, 5, 4), (6,), 4)
    # Force the 2nd perceptron to be a segment entry, as a real value boundary would.
    flow.get_perceptrons()[1].is_encoding_layer = True
    flow = _install_ttfs(flow, S)
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    seg_of = partition_perceptron_segments(repr_._exec_order, repr_._deps)
    assert len(set(seg_of.values())) == 2
    drv = TTFSSegmentForward(repr_, S)
    x = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    out = drv(x)
    assert out.shape == (3, 4)
    out.sum().backward()
    for p in flow.get_perceptrons():
        assert p.layer.weight.grad is not None
        assert p.layer.weight.grad.abs().sum() > 0
