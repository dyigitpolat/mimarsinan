"""Value-domain executor: the packed/identity program IS the model's affine math."""

import torch
import torch.nn as nn

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model


def _fused_flow_and_ir(model, input_shape, *, max_axons=256, max_neurons=64):
    """Mirror the pipeline: convert (mvm) -> fuse norms -> IR-map."""
    from mimarsinan.transformations.normalization_fusion import fuse_into_perceptron

    flow = convert_torch_model(
        model.eval(), input_shape, 4, device="cpu", packaging=MVM_PACKAGING
    ).eval()
    for perceptron in flow.get_perceptrons():
        fuse_into_perceptron(perceptron, device="cpu")
    repr_ = flow.get_mapper_repr()
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default",
        max_axons=max_axons, max_neurons=max_neurons,
    ).map(repr_)
    return flow, ir


def _identity_flow(model, input_shape, dtype=torch.float64):
    flow, ir = _fused_flow_and_ir(model, input_shape)
    hybrid = build_identity_hybrid_mapping(ir_graph=ir)
    return ValueHybridCoreFlow(hybrid, dtype=dtype), flow


def _packed_flow(ir, *, max_axons=64, max_neurons=64, split=False, dtype=torch.float64):
    from mimarsinan.mapping.platform.mapping_structure import (
        ChipCapabilities,
        MappingStrategy,
    )

    strategy = MappingStrategy.resolve(
        ChipCapabilities(allow_neuron_splitting=split)
    )
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[
            {"max_axons": max_axons, "max_neurons": max_neurons, "count": 999}
        ],
        strategy=strategy,
    )
    return ValueHybridCoreFlow(hybrid, dtype=dtype)


def _mlp():
    torch.manual_seed(7)
    return nn.Sequential(
        nn.Linear(8, 16), nn.BatchNorm1d(16), nn.ReLU(),
        nn.Linear(16, 12), nn.GELU(), nn.Linear(12, 4),
    ).eval()


class TestIdentityProgramMatchesModel:
    def test_mlp_with_host_activations(self):
        # vs the fused source flow: exact modulo fp64 kernel order (R-edge).
        model = _mlp()
        flow, fused = _identity_flow(model, (8,))
        x = torch.randn(6, 8)
        with torch.no_grad():
            got = flow(x)
            want = fused.double()(x.double())
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)

    def test_matches_native_model_at_fold_honesty(self):
        # vs the NATIVE model: the only residual is the fp32 norm fold the
        # Normalization Fusion step performs (shared with the spiking path).
        model = _mlp()
        flow, _ = _identity_flow(model, (8,))
        x = torch.randn(6, 8)
        with torch.no_grad():
            got = flow(x)
            want = model.double()(x.double())
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)

    def test_conv_weight_bank_model(self):
        torch.manual_seed(3)
        model = nn.Sequential(
            nn.Conv2d(1, 4, 3), nn.Flatten(), nn.Linear(4 * 6 * 6, 4)
        ).eval()
        flow, fused = _identity_flow(model, (1, 8, 8))
        x = torch.randn(2, 1, 8, 8)
        with torch.no_grad():
            got = flow(x)
            want = fused.double()(x.double())
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)

    def test_layernorm_host_op_between_packages(self):
        torch.manual_seed(11)
        model = nn.Sequential(
            nn.Linear(8, 16), nn.LayerNorm(16), nn.Linear(16, 4)
        ).eval()
        flow, fused = _identity_flow(model, (8,))
        x = torch.randn(4, 8)
        with torch.no_grad():
            got = flow(x)
            want = fused.double()(x.double())
        torch.testing.assert_close(got, want, atol=1e-9, rtol=1e-9)


class TestPackedProgramMatchesIdentity:
    def test_neuron_split_packing(self):
        # 16-wide layers onto 8-neuron cores force neuron splitting.
        model = _mlp()
        _, ir = _fused_flow_and_ir(model, (8,))
        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=ir), dtype=torch.float64
        )
        packed = _packed_flow(ir, max_axons=256, max_neurons=8, split=True)
        x = torch.randn(5, 8)
        with torch.no_grad():
            torch.testing.assert_close(
                packed(x), identity(x), atol=1e-12, rtol=1e-12
            )

    def test_pool_packing_default(self):
        model = _mlp()
        _, ir = _fused_flow_and_ir(model, (8,))
        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=ir), dtype=torch.float64
        )
        packed = _packed_flow(ir)
        x = torch.randn(5, 8)
        with torch.no_grad():
            torch.testing.assert_close(
                packed(x), identity(x), atol=1e-12, rtol=1e-12
            )


class TestObservableSeam:
    def test_stage_recorder_captures_values(self):
        model = _mlp()
        flow, _ = _identity_flow(model, (8,))
        captured = []
        flow.stage_count_recorder = lambda stage, out: captured.append(
            (stage.name, out.shape)
        )
        with torch.no_grad():
            flow(torch.randn(3, 8))
        assert captured, "neural stages must report their output values"
        assert all(shape[0] == 3 for _name, shape in captured)

    def test_flow_node_counts_reuse(self):
        # The node-granular capture helper works verbatim on value flows.
        from mimarsinan.certification.count_alignment import flow_node_counts

        model = _mlp()
        _, ir = _fused_flow_and_ir(model, (8,))
        identity = ValueHybridCoreFlow(
            build_identity_hybrid_mapping(ir_graph=ir), dtype=torch.float64
        )
        packed = _packed_flow(ir, max_axons=256, max_neurons=8, split=True)
        x = torch.randn(4, 8)
        ref = flow_node_counts(identity, x)
        got = flow_node_counts(packed, x)
        assert set(ref) == set(got)
        for nid in ref:
            torch.testing.assert_close(
                got[nid], ref[nid], atol=1e-12, rtol=1e-12
            )
