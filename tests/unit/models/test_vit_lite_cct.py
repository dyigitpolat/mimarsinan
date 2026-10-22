"""ViT-Lite (CCT block), the ViT gate's network, as a toolchain builder that N1 converts.

Parameter counts are the gate's measured ones (3,740,004 at 100 classes, 3,716,874 at 10). Both attention forms
convert under both contracts: under MVM the qkv, projection, linear1, linear2 and classifier Linears are perceptrons,
and QK^T and PV are Contractions with operand roles (activation, activation), read after the batch-moving
``permute(2, 0, 3, 1, 4)`` through the push-forward; the converted model and the identity value program equal the
native model.
"""

from __future__ import annotations

import contextlib
import copy
import io

import pytest
import torch

from mimarsinan.mapping.mapping_utils import ComputeOpMapper, PerceptronMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.models.builders import BUILDERS_REGISTRY
from mimarsinan.models.families.vit_lite import ViTLiteCCT
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.value_parity import relative_error
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program

SMALL = {"img_size": 8, "patch": 4, "dim": 16, "depth": 2, "heads": 4, "num_classes": 10}
CONTRACTS = {"mvm": MVM_PACKAGING, "spiking": SPIKING_PACKAGING}


def native(factory):
    torch.manual_seed(0)
    return factory().double().eval()


def _convert(model, shape, contract):
    with contextlib.redirect_stdout(io.StringIO()):
        return convert_torch_model(copy.deepcopy(model), shape, 10, packaging=CONTRACTS[contract]).eval()


def _worst(candidate, model, shape, batches=(1, 2, 3)):
    worst = 0.0
    for batch in batches:
        x = torch.randn(batch, *shape, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            want = model(x)
            worst = max(worst, relative_error(candidate(x).reshape(want.shape), want).relative_error)
    return worst


def assert_converted_exact(model, shape, contract):
    flow = _convert(model, shape, contract)
    assert _worst(flow, model, shape) <= 1e-12
    return flow


def assert_value_program_exact(model, shape, batches=(2, 3)):
    program = identity_value_program(_convert(model, shape, "mvm"))
    assert _worst(program, model, shape, batches) <= 1e-12


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("classes,count", [(100, 3_740_004), (10, 3_716_874)])
def test_vit_lite_7_4_has_the_gates_parameter_count(classes, count):
    assert _params(ViTLiteCCT(num_classes=classes)) == count


def test_the_builder_builds_vit_lite_7_4_by_default():
    builder = BUILDERS_REGISTRY["vit_lite_cct"]("cpu", (3, 32, 32), 100, {})
    assert _params(builder.build({})) == 3_740_004


@pytest.mark.parametrize("attention", ["softmax", "relu"])
@pytest.mark.parametrize("contract", sorted(CONTRACTS))
def test_both_attention_forms_convert_exactly(attention, contract):
    flow = assert_converted_exact(native(lambda: ViTLiteCCT(attention=attention, **SMALL)), (3, 8, 8), contract)
    nodes = flow.get_mapper_repr().execution_order()
    contractions = [m for m in nodes if isinstance(m, ComputeOpMapper) and type(m.module).__name__ == "Contraction"]
    assert len(contractions) == 2 * SMALL["depth"]
    assert all(m.module.operand_roles == ("activation", "activation") for m in contractions)
    if contract == "mvm":
        perceptrons = [m for m in nodes if isinstance(m, PerceptronMapper)]
        assert len(perceptrons) == 4 * SMALL["depth"] + 1  # qkv, proj, linear1, linear2 per block, the classifier


@pytest.mark.parametrize("attention", ["softmax", "relu"])
def test_both_attention_forms_value_programs_are_exact(attention):
    assert_value_program_exact(native(lambda: ViTLiteCCT(attention=attention, **SMALL)), (3, 8, 8))


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("attention", ["softmax", "relu"])
def test_vit_lite_7_4_converts_exactly_under_mvm(attention):
    assert_value_program_exact(native(lambda: ViTLiteCCT(attention=attention, num_classes=100)), (3, 32, 32),
                               batches=(2,))
