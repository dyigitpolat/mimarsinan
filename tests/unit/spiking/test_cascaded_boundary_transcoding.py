"""Cascaded NF↔SCM boundary-transcoding parity: the NF segment walk must be
bit-exact to the identity-mapped hybrid executor across host-op segment cuts.

The wire contract (W1 incident class, t0_16/t0_17/t0_18): a boundary spike time
encodes the value NORMALIZED by the consumer's ``input_activation_scale`` (== the
source's propagated boundary out-scale); the consumer's weight fold multiplies
the same scale back in. Encoding the raw value-domain tensor instead (the
pre-fix ``TtfsSegmentPolicy`` behavior) mistimes every boundary spike and
saturates values above the scale — bit-exact only when the boundary scale is 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.latency.ir import IRLatency
from mimarsinan.mapping.pruning.ir_pruning_core import prune_ir_graph
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
from mimarsinan.models.spiking.hybrid.identity_flow import build_identity_spiking_flow
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.spiking.segment_boundary import normalize_ttfs_boundary_value
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.transformations.quantization_bounds import quantization_bounds

WEIGHT_BITS = 8


class _MulChainMLP(nn.Module):
    """Mixer-style: Linear+ReLU stages cut by scale-commuting host mul ops."""

    def __init__(self, dims):
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.ReLU())
            for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = x * 0.7  # host ComputeOp cut (scale-commuting)
        return x


class _AffineChainMLP(nn.Module):
    """Linear+ReLU stages cut by a NON-scale-commuting host affine op.

    ``x*a + b`` does not commute with the boundary scale, so the executor must
    decode to the value domain around the host op (as the deployed nevresim /
    SANA-FE runners do) for NF↔SCM parity — the t0_18 residual after the
    boundary-encode fix (torch↔deployed-sim 0.9570) localized here.
    """

    def __init__(self, dims):
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Linear(a, b), nn.ReLU())
            for a, b in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = x * 0.7 + 0.3  # host ComputeOp cut (affine, non-commuting)
        return x


class _ConvPoolNet(nn.Module):
    """Conv-style (the t0_18 shape): enc conv -> pool -> conv -> pool -> conv -> head."""

    def __init__(self, ch=(4, 6, 6), num_classes=3):
        super().__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(1, ch[0], 3, padding=1), nn.ReLU())
        self.pool0 = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(ch[1], ch[2], 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(ch[2] * 2 * 2, num_classes), nn.ReLU())

    def forward(self, x):
        x = self.pool0(self.conv0(x))
        x = self.pool1(self.conv1(x))
        x = self.conv2(x)
        return self.head(x)


def _calibrated_scales(flow, perceptrons, input_shape, *, mult=0.5):
    maxima = {}
    handles = []
    for i, p in enumerate(perceptrons):
        def hook(_m, _inp, out, idx=i):
            maxima[idx] = max(
                maxima.get(idx, 0.0),
                float(out.detach().abs().max()),
            )
        handles.append(p.activation.register_forward_hook(hook))
    torch.manual_seed(11)
    with torch.no_grad():
        flow(torch.rand(16, *input_shape))
    for h in handles:
        h.remove()
    return [max(maxima[i] * mult, 1e-3) for i in range(len(perceptrons))]


def _build_pair(model, input_shape, num_classes, S, thresholding_mode, *, prune=False):
    """Converted cascaded NF flow + identity-mapped hybrid executor on its IR."""
    flow = convert_torch_model(model, input_shape, num_classes, device="cpu").eval()
    repr_ = flow.get_mapper_repr()
    mark_encoding_layers(repr_)
    perceptrons = list(flow.get_perceptrons())

    scales = _calibrated_scales(flow, perceptrons, input_shape)
    calibrate_scale_aware_boundaries(flow, scales, input_data_scale=1.0)

    # The divergence regime requires a non-unit boundary scale at some segment cut.
    assert any(
        abs(float(torch.as_tensor(p.input_activation_scale).float().mean()) - 1.0) > 0.05
        for p in perceptrons
        if not getattr(p, "is_encoding_layer", False)
    ), "fixture must exercise a boundary scale != 1"

    for p in perceptrons:
        p.set_activation(TTFSActivation(
            T=S,
            activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale,
            bias=p.layer.bias,
            thresholding_mode=thresholding_mode,
            encoding=getattr(p, "is_encoding_layer", False),
        ))

    repr_.assign_perceptron_indices()
    compute_per_source_scales(repr_)

    _, q_max = quantization_bounds(WEIGHT_BITS)
    ir_graph = IRMapping(
        q_max=q_max, firing_mode="TTFS", max_axons=2048, max_neurons=2048,
    ).map(repr_)
    quantize_ir_graph(ir_graph, WEIGHT_BITS, weight_quantization=False)
    if prune:
        prune_ir_graph(
            ir_graph, simulation_steps=S, spiking_mode="ttfs_cycle_based",
        )
    IRLatency(ir_graph).calculate()

    executor = build_identity_spiking_flow(
        input_shape,
        ir_graph,
        S,
        None,
        "TTFS",
        "TTFS",
        thresholding_mode,
        spiking_mode="ttfs_cycle_based",
        ttfs_cycle_schedule="cascaded",
    ).eval()
    return flow.double().eval(), executor, ir_graph


def _assert_nf_scm_bit_exact(flow, executor, S, x):
    """NF final segment decode must equal the executor's spike counts bit-exactly."""
    nf = TTFSSegmentForward(flow.get_mapper_repr(), S)
    with torch.no_grad():
        out_nf = nf(x.double())
        out_scm = executor(x.double())
    theta_last = torch.as_tensor(
        list(flow.get_perceptrons())[-1].activation_scale, dtype=torch.float64,
    ).reshape(-1)
    nf_counts = (out_nf / theta_last.clamp(min=1e-12)) * S
    torch.testing.assert_close(
        nf_counts, out_scm.to(nf_counts.dtype), atol=1e-6, rtol=0.0,
    )


def _samples(input_shape, n=16):
    torch.manual_seed(3)
    return torch.rand(n, *input_shape, dtype=torch.float64)


@pytest.mark.parametrize("S", [4, 32])
@pytest.mark.parametrize("thresholding_mode", ["<=", "<"])
def test_mixer_style_host_op_boundaries_bit_exact(S, thresholding_mode):
    torch.manual_seed(0)
    model = _MulChainMLP([10, 12, 12, 4])
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)
                nn.init.uniform_(m.bias, -0.05, 0.05)
    flow, executor, _ = _build_pair(model, (10,), 4, S, thresholding_mode)
    _assert_nf_scm_bit_exact(flow, executor, S, _samples((10,)))


@pytest.mark.parametrize("S", [4, 32])
def test_affine_host_op_boundaries_bit_exact(S):
    torch.manual_seed(4)
    model = _AffineChainMLP([10, 12, 12, 4])
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.5, 0.5)
                nn.init.uniform_(m.bias, -0.05, 0.05)
    flow, executor, _ = _build_pair(model, (10,), 4, S, "<=")
    _assert_nf_scm_bit_exact(flow, executor, S, _samples((10,)))


def test_resolve_stage_compute_scales_wrapper_owns_domain():
    """A ScaleNormalizingWrapper op transcodes rate<->absolute itself; the outer
    executor scales must stay (1, 1) or the domain is applied twice."""
    from types import SimpleNamespace

    from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
        resolve_stage_compute_scales,
    )
    from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper

    mapping = SimpleNamespace(
        node_activation_scales={7: 3.0},
        node_input_activation_scales={7: 2.0},
    )
    wrapped = ScaleNormalizingWrapper(
        nn.Identity(), [torch.tensor([2.0])], torch.tensor([3.0]),
    )
    op_wrapped = SimpleNamespace(params={"module": wrapped})
    assert resolve_stage_compute_scales(
        mapping, 7, apply_ttfs=True, op=op_wrapped,
    ) == (1.0, 1.0)

    op_plain = SimpleNamespace(params={"module": nn.Identity()})
    assert resolve_stage_compute_scales(
        mapping, 7, apply_ttfs=True, op=op_plain,
    ) == (2.0, 3.0)
    assert resolve_stage_compute_scales(
        mapping, 7, apply_ttfs=False, op=op_plain,
    ) == (1.0, 1.0)


def _inflated_conv_net(seed):
    """ConvPool net with inflated weights so post-ReLU activations span the grid."""
    torch.manual_seed(seed)
    model = _ConvPoolNet()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                m.weight.mul_(3.0)
                if m.bias is not None:
                    m.bias.copy_(torch.randn_like(m.bias) * 0.1)
    return model


@pytest.mark.parametrize("S", [4, 32])
def test_conv_style_pool_boundaries_bit_exact(S):
    model = _inflated_conv_net(1)
    flow, executor, _ = _build_pair(model, (1, 8, 8), 3, S, "<=")
    _assert_nf_scm_bit_exact(flow, executor, S, _samples((1, 8, 8)))


@pytest.mark.parametrize("S", [4])
def test_conv_style_pruned_graph_bit_exact(S):
    """Pruned deployment: structurally-zeroed channels compact away without
    dropping LIVE rows through the pool ops (the t0_18 0.2031 second mechanism)."""
    model = _inflated_conv_net(2)
    with torch.no_grad():
        # Structurally prune half of the first on-chip conv's output channels.
        conv1 = model.conv1[0]
        conv1.weight[::2] = 0.0
        conv1.bias[::2] = 0.0
    flow, executor, ir_graph = _build_pair(model, (1, 8, 8), 3, S, "<=", prune=True)
    _assert_nf_scm_bit_exact(flow, executor, S, _samples((1, 8, 8)))


def test_normalize_ttfs_boundary_value_contract():
    """Wire-domain transcode: divide by the boundary scale, clamp to [0, 1]."""
    v = torch.tensor([[-0.5, 0.0, 1.0, 2.0, 5.0]], dtype=torch.float64)
    out = normalize_ttfs_boundary_value(v, 2.0)
    torch.testing.assert_close(
        out, torch.tensor([[0.0, 0.0, 0.5, 1.0, 1.0]], dtype=torch.float64),
    )
    # Scalar tensor scale behaves like the float scale.
    out_t = normalize_ttfs_boundary_value(v, torch.tensor(2.0))
    torch.testing.assert_close(out_t, out)
    # Unit scale is the identity on the already-normalized domain.
    torch.testing.assert_close(
        normalize_ttfs_boundary_value(torch.tensor([[0.25]]), 1.0),
        torch.tensor([[0.25]]),
    )


def test_boundary_scale_above_one_previously_saturated():
    """Regression pin: with a boundary scale of 2, a value of 1.5 must encode the
    normalized 0.75 (spike time round(S/4)), not saturate at 1.0 (spike time 0)."""
    from mimarsinan.models.spiking.wire_semantics import ttfs_spike_time

    S = 4
    v = torch.tensor([[1.5]], dtype=torch.float64)
    normalized = normalize_ttfs_boundary_value(v, 2.0)
    assert float(ttfs_spike_time(normalized, S)) == 1.0
    saturated = v.clamp(0.0, 1.0)
    assert float(ttfs_spike_time(saturated, S)) == 0.0
