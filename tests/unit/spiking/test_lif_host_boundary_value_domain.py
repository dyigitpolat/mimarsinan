"""W1c wire contract: LIF host-op boundaries re-encode value-domain trains (uniform(rate) * producer out-scale)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.spiking.scale_aware_boundaries import read_boundary_out_scales
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.spiking.spike_trains import uniform_spike_train
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


class _HostRelay(nn.Module):
    """Order- and scale-preserving host op (the MaxPool seam without geometry)."""

    def forward(self, x):
        return x * 1.0


def _lif_perceptron(out_ch, in_features, theta, T, *, encoding=False):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.is_encoding_layer = encoding
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    lif.use_cycle_accurate_trains = True
    p.base_activation = lif
    p.activation = lif
    return p


def _tiny_host_boundary_model(T: int, theta_enc: float, theta_hidden: float = 0.5):
    """input(8) -> encoding Perceptron(6, theta_enc) -> host op -> Perceptron(3, theta_hidden).

    Mirrors the deep_cnn conv->MaxPool->conv seam: the host op splits the graph
    into two neural segments and its boundary re-encode must carry theta_enc.
    """
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, theta_enc, T, encoding=True)
    m1 = PerceptronMapper(inp, p1)
    host = ComputeOpMapper(m1, _HostRelay(), input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, theta_hidden, T)
    # Production scale propagation folds the producer's out-scale into the
    # consumer's effective weights (per_input_scales); the wire contract
    # multiplies the same scale back at the boundary re-encode.
    p2.per_input_scales = torch.full((6,), float(theta_enc))
    m2 = PerceptronMapper(host, p2)
    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return repr_, hybrid, p1, p2, host


# ---------------------------------------------------------------------------
# 1. The pure boundary out-scale table (read-only twin of the deployed fold)
# ---------------------------------------------------------------------------

def test_read_boundary_out_scales_pass_through() -> None:
    theta_enc, theta_hidden, T = 2.185, 0.5, 8
    repr_, _, _, _, _ = _tiny_host_boundary_model(T, theta_enc, theta_hidden)
    repr_._ensure_exec_graph()
    scales = read_boundary_out_scales(repr_, input_data_scale=1.0)

    by_type = {}
    for node in repr_._exec_order:
        by_type.setdefault(type(node).__name__, []).append(scales[node])
    assert by_type["InputMapper"] == [pytest.approx(1.0)]
    # Host op inherits its producer's out-scale (scale-homogeneous pass-through).
    assert by_type["ComputeOpMapper"] == [pytest.approx(theta_enc)]
    assert sorted(by_type["PerceptronMapper"]) == [
        pytest.approx(theta_hidden), pytest.approx(theta_enc),
    ]


def test_read_boundary_out_scales_does_not_mutate() -> None:
    repr_, _, p1, p2, _ = _tiny_host_boundary_model(8, 2.185)
    before = (float(p1.input_activation_scale), float(p2.input_activation_scale))
    read_boundary_out_scales(repr_, input_data_scale=1.0)
    after = (float(p1.input_activation_scale), float(p2.input_activation_scale))
    assert before == after


# ---------------------------------------------------------------------------
# 2. The t0_03 regression pin: value-domain trains across a host boundary
# ---------------------------------------------------------------------------

def test_host_boundary_train_is_value_domain() -> None:
    """The NF walk's boundary re-encode after a host op must hand the consumer
    ``uniform(rate) * theta_producer`` (value-domain), mirroring the deployed IR
    fold that bakes the producer scale into the consumer's weights."""
    from spikingjelly.activation_based import functional

    theta_enc, theta_hidden, T = 2.185, 0.5, 8
    repr_, _, p1, p2, host = _tiny_host_boundary_model(T, theta_enc, theta_hidden)
    torch.manual_seed(7)
    x = 3.0 * torch.rand(2, 8)

    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy())
    with torch.no_grad():
        nf_out = driver(x)

    # Manual mirror: encoder wire rate -> host op on the rate -> value-domain
    # uniform train -> downstream perceptron per-cycle.
    lif1, lif2 = p1.activation, p2.activation
    lif1.set_cycle_accurate(False)
    functional.reset_net(lif1.if_node)
    with torch.no_grad():
        rate_norm = (p1(x) / lif1.activation_scale.clamp(min=1e-12)).clamp(0.0, 1.0)
        host_rate = host.module(rate_norm)
        train = uniform_spike_train(host_rate.clamp(0.0, 1.0), T) * theta_enc
        lif2.set_cycle_accurate(True)
        functional.reset_net(lif2.if_node)
        outs = [p2(train[t]) for t in range(T)]
        lif2.set_cycle_accurate(False)
        expected = torch.stack(outs, dim=0).mean(dim=0)

    torch.testing.assert_close(nf_out, expected, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("theta_enc", [2.185, 1.0])
def test_nf_driver_equals_hybrid_flow_across_host_boundary(theta_enc) -> None:
    """Cross-twin mirror through a mid-graph host op: NF driver and HCM flow
    must agree (theta != 1 is the t0_03 deep_cnn MaxPool-seam signature;
    theta == 1 guards that the re-encode stays the identity there)."""
    T = 8
    theta_hidden = 0.5
    repr_, hybrid, _, p2, _ = _tiny_host_boundary_model(T, theta_enc, theta_hidden)
    torch.manual_seed(11)
    x = 3.0 * torch.rand(2, 8)

    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy())
    flow = SpikingHybridCoreFlow(
        (8,), hybrid, simulation_length=T,
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )
    with torch.no_grad():
        nf = driver(x) / p2.activation.activation_scale.clamp(min=1e-12)
        hc = flow(x) / T
    torch.testing.assert_close(
        nf.to(torch.float32), hc.to(torch.float32), atol=1e-6, rtol=0.0,
    )
