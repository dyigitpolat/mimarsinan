"""Parity: legacy vs direct neural segment softcore build."""

import torch.nn as nn

from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
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


def test_legacy_and_direct_hybrid_stage_counts_match():
    ir = _tiny_ir()
    cores = [{"max_axons": 128, "max_neurons": 128, "count": 16}]
    legacy = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=cores,
        use_legacy_softcore_flush=True,
    )
    direct = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=cores,
        use_legacy_softcore_flush=False,
    )
    assert len(legacy.stages) == len(direct.stages)
    for ls, ds in zip(legacy.stages, direct.stages):
        assert ls.kind == ds.kind
        if ls.kind == "neural":
            assert ls.hard_core_mapping is not None
            assert ds.hard_core_mapping is not None
            assert len(ls.hard_core_mapping.cores) == len(ds.hard_core_mapping.cores)
