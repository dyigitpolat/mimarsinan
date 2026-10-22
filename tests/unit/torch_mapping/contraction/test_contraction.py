"""Every product of two activations is one host op whose payload is a ``Contraction``.

The payload evaluates the traced call itself (bit for bit the native op) and declares the interface the capability
table and N2 read: ``einsum_spec``, ``operand_roles`` (activation, activation), ``operand_shapes`` (per sample) and
``forward_macs()``. The IR op's ``op_type`` is ``Contraction``; a product with a parameter operand keeps today's path.
"""

from __future__ import annotations

import contextlib
import copy
import io

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.mapping.verification.onchip_majority import count_host_ops
from mimarsinan.torch_mapping.converter import convert_torch_model, trace_and_convert
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_value_program_exact,
    convert,
    native,
)

T, F = 6, 8


class Bilinear(nn.Module):
    """(x W1)(x W2)^T through ``product``: the scout's ``bmm_act_x_act`` and its other spellings."""

    def __init__(self, product):
        super().__init__()
        self.a, self.b = nn.Linear(F, 5), nn.Linear(F, 5)
        self.head = nn.Linear(T * T, 10)
        self.product = product

    def forward(self, x):
        return self.head(torch.flatten(self.product(self.a(x), self.b(x)), 1))


PRODUCTS = {
    "torch_bmm": lambda a, b: torch.bmm(a, b.transpose(1, 2)),
    "torch_matmul": lambda a, b: torch.matmul(a, b.transpose(1, 2)),
    "operator_matmul": lambda a, b: a @ b.transpose(1, 2),
    "method_matmul": lambda a, b: a.matmul(b.transpose(1, 2)),
    "method_bmm": lambda a, b: a.bmm(b.transpose(1, 2)),
    "einsum": lambda a, b: torch.einsum("bik,bjk->bij", a, b),
    "einsum_list": lambda a, b: torch.einsum("bik,bjk->bij", [a, b]),
}
SPECS = {"torch_bmm": "bmk,bkn->bmn", "method_bmm": "bmk,bkn->bmn", "einsum": "bik,bjk->bij",
         "einsum_list": "bik,bjk->bij"}


def _contractions(flow):
    return [m for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and type(m.module).__name__ == "Contraction"]


def _ir(flow):
    with contextlib.redirect_stdout(io.StringIO()):
        return IRMapping(q_max=1.0, firing_mode="Default", max_axons=None, max_neurons=None,
                         allow_coalescing=False, hardware_bias=False).map(flow.get_mapper_repr())


@pytest.mark.parametrize("key", sorted(PRODUCTS))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_product_of_two_activations_is_one_contraction(key, contract):
    flow = assert_converted_exact(native(lambda: Bilinear(PRODUCTS[key])), (T, F), contract)
    found = _contractions(flow)
    assert len(found) == 1
    payload = found[0].module
    assert payload.operand_roles == ("activation", "activation")
    assert payload.operand_shapes == ((T, 5), (5, T)) or payload.operand_shapes == ((T, 5), (T, 5))
    assert payload.einsum_spec == SPECS.get(key, "...mk,...kn->...mn")
    assert payload.forward_macs() == T * T * 5
    ops = [op for op in _ir(flow).get_compute_ops() if op.op_type == "Contraction"]
    assert len(ops) == 1 and len(ops[0].params["input_shapes"]) == 2


@pytest.mark.parametrize("key", sorted(PRODUCTS))
def test_a_contraction_value_program_is_exact(key):
    assert_value_program_exact(native(lambda: Bilinear(PRODUCTS[key])), (T, F))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_ellipsis_einsum_stays_refused_by_the_batch_domain(contract):
    """Wave 0's batch domain states no extent for an ellipsis ``einsum``, so a reshape it reaches is refused by name
    (``BatchAxisLayoutError``), as on the base: a boundary of the shape proof, not of the contraction rule."""
    from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
    model = native(lambda: Bilinear(lambda a, b: torch.einsum("...ik,...jk->...ij", a, b)))
    with pytest.raises(BatchAxisLayoutError, match="einsum"):
        convert(model, (T, F), contract)


class ParamProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(F, 4))
        self.head = nn.Linear(T * 4, 10)

    def forward(self, x):
        return self.head(torch.flatten(x @ self.w, 1))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_product_with_a_parameter_keeps_the_generic_host_op(contract):
    flow = assert_converted_exact(native(ParamProduct), (T, F), contract)
    assert not _contractions(flow)
    generic = [m for m in flow.get_mapper_repr().execution_order()
               if isinstance(m, ComputeOpMapper) and type(m.module).__name__ == "ComputeAdapter"
               and getattr(m.module, "fn", None) is not None and m.module.fn.__name__ == "matmul"]
    assert len(generic) == 1 and generic[0].module._bound_count == 1


def test_both_mac_counters_read_the_declared_count():
    model = native(lambda: Bilinear(PRODUCTS["torch_bmm"])).float()
    with contextlib.redirect_stdout(io.StringIO()):
        flow = trace_and_convert(model, (T, F), packaging=MVM_PACKAGING).flow
    est = estimate_onchip_fraction(flow, (T, F), 10, metric="macs")
    assert count_host_ops(_ir(flow)) == est.host == T * T * 5


def test_swapped_contraction_operands_are_refused_by_stage_2_naming_the_node(monkeypatch):
    """A lowering that swaps the two operands of the Contraction (square operands, so every shape still agrees) is
    caught by the identity value program's comparison, which names the product's node."""

    class Square(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Linear(F, T), nn.Linear(F, T)
            self.head = nn.Linear(T * T, 10)

        def forward(self, x):
            return self.head(torch.flatten(torch.bmm(self.a(x), self.b(x)), 1))

    original = ComputeOpMapper._map_to_ir

    def swapped(self, ir_mapping):
        if type(self.module).__name__ == "Contraction":
            self._sources_list = list(reversed(self._sources_list))
            try:
                return original(self, ir_mapping)
            finally:
                self._sources_list = list(reversed(self._sources_list))
        return original(self, ir_mapping)

    monkeypatch.setattr(ComputeOpMapper, "_map_to_ir", swapped)
    with pytest.raises(ToolchainRefusal) as excinfo:
        with contextlib.redirect_stdout(io.StringIO()):
            convert_torch_model(copy.deepcopy(native(Square)), (T, F), 10, packaging=MVM_PACKAGING,
                                value_program_stage=True)
    assert "identity value program" in str(excinfo.value) and "bmm" in str(excinfo.value)
