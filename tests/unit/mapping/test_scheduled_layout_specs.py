"""IRGraph carries layout specs; scheduled split uses them."""

import torch.nn as nn

from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _wide_fc_repr():
    inp = InputMapper((512,))
    p = Perceptron(64, 512, normalization=nn.Identity(), base_activation_name="ReLU")
    return ModelRepresentation(PerceptronMapper(inp, p))


class TestScheduledLayoutSpecs:
    def test_ir_graph_carries_layout_softcores(self):
        repr_ = _wide_fc_repr()
        ir = IRMapping(
            q_max=127.0,
            firing_mode="Default",
            max_axons=64,
            max_neurons=64,
            allow_coalescing=True,
        ).map(repr_)
        assert len(ir.layout_softcores) >= 1
        neural = ir.get_neural_cores()
        assert all(c.layout_softcore_index is not None for c in neural)

    def test_scheduled_build_uses_layout_specs(self):
        repr_ = _wide_fc_repr()
        ir = IRMapping(
            q_max=127.0,
            firing_mode="Default",
            max_axons=64,
            max_neurons=64,
            allow_coalescing=True,
        ).map(repr_)
        cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 32}]
        hcm = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=cores_config,
            allow_scheduling=True,
            allow_coalescing=True,
        )
        assert len(hcm.stages) >= 1
        neural_stages = [s for s in hcm.stages if s.kind == "neural"]
        assert neural_stages
