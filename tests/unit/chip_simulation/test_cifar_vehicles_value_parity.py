"""BC-2 vehicle DoD: the mvm value program IS each CIFAR model's math, fp64-tight.

For each vehicle (cifar_resnet20 / cifar_vgg8 / cifar_vit_leaf) at the REAL
CIFAR-10 geometry (3, 32, 32): convert (MVM_PACKAGING) -> IR -> identity value
program vs the NATIVE model in fp64 on 2 random batches, asserted at 1e-12
following the test_vit_value_parity conventions. Residual adds, maxpools,
avgpools, and post-add ReLUs ride as host ComputeOps; conv/linear (+BN
absorption) become affine cores — parity certifies the whole assembly.

The model is converted in fp64 (the converter follows the source parameter
dtype): BN absorption COMPUTES folded weights, so the fold must run at the
comparison precision — converting at fp32 leaves an irreducible ~1e-9
fold-rounding floor that is a property of fp32 storage, not of the conversion.
"""

import pytest
import torch

from mimarsinan.chip_simulation.value_run import ValueHybridCoreFlow
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.packing.hybrid_build_pool import build_identity_hybrid_mapping
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.models.cifar_models import cifar_resnet20, cifar_vgg8
from mimarsinan.models.vit_leaf import cifar_vit_leaf
from mimarsinan.torch_mapping.converter import convert_torch_model

_INPUT_SHAPE = (3, 32, 32)
_NUM_CLASSES = 10


def _identity_value_flow(model):
    """Convert (mvm) -> mapper repr -> IR -> identity value program."""
    flow = convert_torch_model(
        model, _INPUT_SHAPE, num_classes=_NUM_CLASSES, packaging=MVM_PACKAGING
    )
    ir = map_model_to_ir(flow.get_mapper_repr())
    return ValueHybridCoreFlow(
        build_identity_hybrid_mapping(ir_graph=ir), device="cpu",
        dtype=torch.float64,
    )


def _batches(n_batches=2, batch=2):
    for seed in range(n_batches):
        generator = torch.Generator().manual_seed(seed)
        yield torch.randn(batch, *_INPUT_SHAPE, dtype=torch.float64,
                          generator=generator)


def _assert_identity_parity(model):
    model64 = model.double()
    identity = _identity_value_flow(model64)
    for x in _batches():
        with torch.no_grad():
            want = model64(x)
            got = identity(x)
        torch.testing.assert_close(got.reshape(want.shape), want,
                                   atol=1e-12, rtol=1e-12)


@pytest.mark.timeout(300)
class TestCifarVehicleValueParity:
    def test_cifar_resnet20_identity_program_matches_native_fp64(self):
        torch.manual_seed(0)
        _assert_identity_parity(cifar_resnet20(num_classes=_NUM_CLASSES).eval())

    def test_cifar_vgg8_identity_program_matches_native_fp64(self):
        torch.manual_seed(0)
        _assert_identity_parity(cifar_vgg8(num_classes=_NUM_CLASSES).eval())

    def test_cifar_vit_leaf_identity_program_matches_native_fp64(self):
        torch.manual_seed(0)
        _assert_identity_parity(cifar_vit_leaf(num_classes=_NUM_CLASSES).eval())
