"""ViT vehicle DoD: the value program IS the leaf-ViT's math, fp64-bit-tight.

convert (mvm) -> IR -> identity/packed value program vs the NATIVE model in
fp64. Measured parity: TinyViT identity 1.6e-15, TinyViT packed 1.1e-15,
DeiT-Tiny-scale identity 4.2e-15 (5001 IR nodes) — asserted at 1e-12
following the TestPackedProgramMatchesIdentity atol conventions.
"""

import pytest
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.packing.hybrid_build_pool import (
    build_hybrid_hard_core_mapping,
    build_identity_hybrid_mapping,
)
from mimarsinan.mapping.platform.mapping_structure import (
    ChipCapabilities,
    MappingStrategy,
)
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.vit_leaf import deit_tiny_leaf, tiny_test_vit
from mimarsinan.torch_mapping.converter import convert_torch_model


def _converted_ir(model, input_shape, num_classes):
    """Probe-measured path: convert (mvm) -> mapper repr -> IR graph."""
    flow = convert_torch_model(
        model, input_shape, num_classes=num_classes, packaging=MVM_PACKAGING
    )
    return map_model_to_ir(flow.get_mapper_repr())


def _identity_value_flow(ir):
    return ValueHybridCoreFlow(
        build_identity_hybrid_mapping(ir_graph=ir), device="cpu",
        dtype=torch.float64,
    )


def _batches(input_shape, n_batches=3, batch=4):
    for seed in range(n_batches):
        generator = torch.Generator().manual_seed(seed)
        yield torch.randn(batch, *input_shape, dtype=torch.float64,
                          generator=generator)


class TestTinyViTValueParity:
    """Fast DoD gates at TinyViT scale (8px, d=16, 1 block)."""

    def test_identity_program_matches_native_model_fp64(self):
        torch.manual_seed(0)
        model = tiny_test_vit().eval()
        ir = _converted_ir(model, (3, 8, 8), num_classes=10)
        identity = _identity_value_flow(ir)
        model64 = model.double()
        for x in _batches((3, 8, 8)):
            with torch.no_grad():
                want = model64(x)
                got = identity(x)
            torch.testing.assert_close(got.reshape(want.shape), want,
                                       atol=1e-12, rtol=1e-12)

    def test_packed_program_matches_native_model_fp64(self):
        # Bin-packed HardCores (64 axons x 32 neurons, neuron splitting on)
        # carry the exact same math as the native model.
        torch.manual_seed(0)
        model = tiny_test_vit().eval()
        ir = _converted_ir(model, (3, 8, 8), num_classes=10)
        strategy = MappingStrategy.resolve(
            ChipCapabilities(allow_neuron_splitting=True)
        )
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=[{"max_axons": 64, "max_neurons": 32, "count": 999}],
            strategy=strategy,
        )
        assert any(stage.kind == "neural" for stage in hybrid.stages)
        packed = ValueHybridCoreFlow(hybrid, device="cpu", dtype=torch.float64)
        model64 = model.double()
        for x in _batches((3, 8, 8)):
            with torch.no_grad():
                want = model64(x)
                got = packed(x)
            torch.testing.assert_close(got.reshape(want.shape), want,
                                       atol=1e-12, rtol=1e-12)


@pytest.mark.slow
@pytest.mark.timeout(900)
class TestDeiTTinyScaleValueParity:
    """DeiT-Tiny geometry (d=192, 3 heads, 12 blocks, 197 tokens): the full
    5001-node IR runs end to end and stays fp64-bit-tight vs the native model."""

    def test_identity_program_matches_native_model_fp64(self):
        torch.manual_seed(0)
        model = deit_tiny_leaf(num_classes=1000).eval()
        ir = _converted_ir(model, (3, 224, 224), num_classes=1000)
        assert len(ir.nodes) > 4000  # the whole 12-block graph, not a stub
        identity = _identity_value_flow(ir)
        model64 = model.double()
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(2, 3, 224, 224, dtype=torch.float64, generator=generator)
        with torch.no_grad():
            want = model64(x)
            got = identity(x)
        torch.testing.assert_close(got.reshape(want.shape), want,
                                   atol=1e-12, rtol=1e-12)
