"""[calculus §16/PR38] HCM synchronized reference: count-domain segment
execution bit-follows the NF synchronized walk (the same strict staircase)."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8


def _tiny(theta_enc: float = 1.0, theta_hidden: float = 1.0):
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = Perceptron(6, 8, normalization=nn.Identity())
    p1.is_encoding_layer = True
    p1.set_activation_scale(theta_enc)
    lif1 = LIFActivation(T=T, activation_scale=p1.activation_scale)
    lif1.use_cycle_accurate_trains = True
    p1.base_activation = lif1
    p1.activation = lif1
    p2 = Perceptron(3, 6, normalization=nn.Identity())
    p2.set_activation_scale(theta_hidden)
    p2.per_input_scales = torch.full((6,), float(theta_enc))
    lif2 = LIFActivation(T=T, activation_scale=p2.activation_scale)
    lif2.use_cycle_accurate_trains = True
    p2.base_activation = lif2
    p2.activation = lif2
    repr_ = ModelRepresentation(PerceptronMapper(PerceptronMapper(inp, p1), p2))
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return repr_, hybrid


def _flow(hybrid, synchronized: bool) -> SpikingHybridCoreFlow:
    return SpikingHybridCoreFlow(
        (8,), hybrid, T,
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
        lif_execution_synchronized=synchronized,
    )


def test_sync_flow_matches_nf_sync_walk():
    repr_, hybrid = _tiny()
    x = torch.rand(4, 8) * 0.9
    driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(synchronized=True))
    with torch.no_grad():
        nf = driver(x)
        hcm = _flow(hybrid, synchronized=True)(x)
    # The flow's contract is raw counts; the NF walk returns values
    # (counts/T * theta_out, theta_out = 1.0 here). Bit-check the counts.
    torch.testing.assert_close(
        (hcm / T).to(torch.float64), nf.to(torch.float64), atol=1e-9, rtol=0.0,
    )


def test_sync_flow_is_deterministic_and_flag_off_is_streaming():
    repr_, hybrid = _tiny()
    x = torch.rand(2, 8) * 0.9
    with torch.no_grad():
        a = _flow(hybrid, synchronized=True)(x)
        b = _flow(hybrid, synchronized=True)(x)
        s = _flow(hybrid, synchronized=False)(x)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)
    assert s.shape == a.shape
