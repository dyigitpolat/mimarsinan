"""Parity: neural segment softcore build produces valid hybrid mapping."""

import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _tiny_ir():
    inp = InputMapper((32,))
    p = Perceptron(16, 32, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(inp, p))
    return IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=128,
        max_neurons=128,
    ).map(repr_)


def test_hybrid_stage_counts_for_neural_segment():
    ir = _tiny_ir()
    cores = [{"max_axons": 128, "max_neurons": 128, "count": 16}]
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=cores,
    )
    assert len(hybrid.stages) >= 1
    neural_stages = [s for s in hybrid.stages if s.kind == "neural"]
    assert len(neural_stages) >= 1
    for stage in neural_stages:
        assert stage.hard_core_mapping is not None
        assert len(stage.hard_core_mapping.cores) >= 1
