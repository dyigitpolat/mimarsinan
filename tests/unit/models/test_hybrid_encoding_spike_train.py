"""Hybrid flow: encoding spike trains and cycle-accurate segment input."""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers


def _tiny_lif_ir():
    inp = InputMapper((8,))
    enc = Perceptron(4, 8, normalization=nn.Identity())
    enc.is_encoding_layer = True
    enc.base_activation = LIFActivation(T=4, activation_scale=torch.tensor(1.0))
    enc.activation = enc.base_activation
    enc.use_cycle_accurate_trains = True
    hid = Perceptron(2, 4, normalization=nn.Identity())
    hid.base_activation = LIFActivation(T=4, activation_scale=torch.tensor(1.0))
    hid.activation = hid.base_activation
    repr_ = ModelRepresentation(
        PerceptronMapper(PerceptronMapper(inp, enc), hid),
    )
    mark_encoding_layers(repr_)
    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=32,
        max_neurons=32,
    ).map(repr_)
    return ir, enc


def test_cycle_accurate_hybrid_forward_uses_encoding_trains() -> None:
    ir, enc = _tiny_lif_ir()
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    flow = SpikingHybridCoreFlow(
        (8,),
        hybrid,
        simulation_length=4,
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )
    x = torch.rand(2, 8)
    with torch.no_grad():
        out = flow(x)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()

    gathered = torch.rand(2, 8)
    expected = enc.forward_spiking(gathered)
    assert expected.shape[0] == 4


def test_build_segment_input_uniform_fallback_for_missing_trains() -> None:
    """Conv/host boundaries without LIF trains fall back to uniform rate encoding."""
    ir, _ = _tiny_lif_ir()
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    flow = SpikingHybridCoreFlow(
        (8,),
        hybrid,
        simulation_length=4,
        spiking_mode="lif",
        cycle_accurate_lif_forward=True,
    )
    stage = next(s for s in hybrid.stages if s.kind == "neural")
    in_size = sum(s.size for s in stage.input_map)
    rates = torch.rand(1, in_size)
    train = flow._build_segment_input_spike_train(
        stage,
        rates,
        {},
        T=4,
        batch_size=1,
        device=rates.device,
    )
    assert train.shape == (4, 1, in_size)
