"""Identity hybrid mapping: 1:1 NeuralCore→HardCore, packing-free (rung-2 gate)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import make_tiny_ir_graph

from mimarsinan.chip_simulation.ttfs.ttfs_executor import run_ttfs_hybrid_contract
from mimarsinan.mapping.ir import NeuralCore
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)

_CORES_CONFIG = [{"max_axons": 256, "max_neurons": 256, "count": 20}]


def _make_mini_mixer_ir_graph(seed=42):
    """Conv → flatten → permute → FC → permute → mean → classifier IR graph."""
    from mimarsinan.mapping.support.compute_modules import ComputeAdapter as _CA
    from mimarsinan.mapping.mappers.structural import (
        InputMapper, PermuteMapper, EinopsRearrangeMapper,
    )
    from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    torch.manual_seed(seed)
    input_shape = (1, 4, 4)
    num_patches, patch_dim = 4, 2

    conv = Conv2DPerceptronMapper(
        InputMapper(input_shape),
        in_channels=1, out_channels=patch_dim,
        kernel_size=2, stride=2, padding=0,
        bias=True, use_batchnorm=False,
        base_activation_name="Identity",
    )
    flat = EinopsRearrangeMapper(conv, "... c h w -> ... c (h w)")
    perm1 = PermuteMapper(flat, (0, 2, 1))
    perm2 = PermuteMapper(perm1, (0, 2, 1))
    p_tok = Perceptron(num_patches, num_patches, normalization=nn.Identity(),
                       base_activation_name="ReLU")
    fc_tok = PerceptronMapper(perm2, p_tok)
    perm3 = PermuteMapper(fc_tok, (0, 2, 1))
    mean = ComputeOpMapper(perm3, _CA(torch.mean, kwargs={"dim": 1}))
    p_cls = Perceptron(3, patch_dim, normalization=nn.Identity(),
                       base_activation_name="Identity")
    classifier = PerceptronMapper(mean, p_cls)
    repr_ = ModelRepresentation(classifier)

    compute_per_source_scales(repr_)
    ir_mapping = IRMapping(q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024)
    ir_graph = ir_mapping.map(repr_)
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)
    from mimarsinan.mapping.latency.ir import IRLatency

    IRLatency(ir_graph).calculate()
    return ir_graph, input_shape


class TestIdentityStructure:
    def test_one_hardcore_per_neuralcore_no_packing(self):
        ir_graph = make_tiny_ir_graph()
        identity = build_identity_hybrid_mapping(ir_graph=ir_graph)

        neural_node_ids = {n.id for n in ir_graph.get_neural_cores()}
        placed_ids = []
        for stage in identity.stages:
            if stage.kind != "neural":
                continue
            hcm = stage.hard_core_mapping
            for core_idx, hardcore in enumerate(hcm.cores):
                placements = hcm.soft_core_placements_per_hard_core[core_idx]
                assert len(placements) == 1, "identity mapping must not pool softcores"
                placement = placements[0]
                assert placement["axon_offset"] == 0
                assert placement["neuron_offset"] == 0
                assert hardcore.axons_per_core == placement["axons"]
                assert hardcore.neurons_per_core == placement["neurons"]
                assert hardcore.available_axons == 0
                assert hardcore.available_neurons == 0
                assert not any(s.is_off_ for s in hardcore.axon_sources), (
                    "identity cores must not be padded with off sources"
                )
                placed_ids.append(placement["ir_node_id"])
        assert sorted(placed_ids) == sorted(neural_node_ids)

    def test_packed_mapping_pools_by_contrast(self):
        # Sanity that the tiny graph genuinely exercises pooling in the packed
        # path (otherwise the equivalence test below proves nothing).
        ir_graph = make_tiny_ir_graph()
        packed = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph, cores_config=_CORES_CONFIG,
        )
        packed_cores = [c for s in packed.stages if s.kind == "neural"
                        for c in s.hard_core_mapping.cores]
        assert any(
            any(s.is_off_ for s in core.axon_sources) or core.available_neurons > 0
            for core in packed_cores
        )


class TestIdentityVsPackedEquivalence:
    """Packing must be value-preserving: identity isolates IR semantics, so
    the contract runner's stage outputs must match the packed mapping exactly."""

    @pytest.mark.parametrize("spiking_mode,schedule", [
        ("ttfs_quantized", None),
        ("ttfs_cycle_based", "synchronized"),
    ])
    def test_tiny_graph_outputs_identical(self, spiking_mode, schedule):
        np.random.seed(11)
        ir_graph = make_tiny_ir_graph()
        identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
        packed = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph, cores_config=_CORES_CONFIG,
        )
        x = np.random.rand(1, 8).astype(np.float64)
        self._assert_contract_outputs_equal(
            identity, packed, x, spiking_mode=spiking_mode, schedule=schedule,
        )

    @pytest.mark.parametrize("spiking_mode,schedule", [
        ("ttfs_quantized", None),
        ("ttfs_cycle_based", "synchronized"),
    ])
    def test_mini_mixer_outputs_identical(self, spiking_mode, schedule):
        ir_graph, input_shape = _make_mini_mixer_ir_graph()
        identity = build_identity_hybrid_mapping(ir_graph=ir_graph)
        packed = build_hybrid_hard_core_mapping(
            ir_graph=ir_graph, cores_config=_CORES_CONFIG,
        )
        torch.manual_seed(3)
        x = torch.rand(1, *input_shape).reshape(1, -1).double().numpy()
        self._assert_contract_outputs_equal(
            identity, packed, x, spiking_mode=spiking_mode, schedule=schedule,
        )

    @staticmethod
    def _assert_contract_outputs_equal(identity, packed, x, *, spiking_mode, schedule):
        runs = {}
        for label, mapping in (("identity", identity), ("packed", packed)):
            runs[label] = run_ttfs_hybrid_contract(
                mapping,
                x,
                simulation_length=4,
                spiking_mode=spiking_mode,
                ttfs_cycle_schedule=schedule,
            )
        id_segments = runs["identity"].record.segments
        pk_segments = runs["packed"].record.segments
        assert id_segments.keys() == pk_segments.keys()
        for stage_index in id_segments:
            np.testing.assert_allclose(
                id_segments[stage_index].seg_output,
                pk_segments[stage_index].seg_output,
                rtol=0, atol=0,
                err_msg=(
                    f"stage {stage_index}: packing changed segment output "
                    f"({spiking_mode}/{schedule})"
                ),
            )
