"""The tensor-method rule: a method with a torch function of its name converts as the host op of the method itself.

``.softmax(dim=-1)``, ``.sigmoid()``, ``.relu()``, ``.sum(1)`` convert exactly (on the base: refused, outside the
14-name allowlist); a ``chunk`` whose every user is an integer ``getitem`` becomes ``narrow``s (ShuffleNetV2's
split), one with another user, ``split`` and ``unbind`` refuse by name (``TupleMethodError``), and a method with no
torch function of its name keeps ``UnsupportedCallMethodError`` (through the analyzer's ``RepresentabilityError``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.representability_analyzer import RepresentabilityError
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_refused_by_name,
    assert_value_program_exact,
    convert,
    native,
)

C, S = 8, 4


def _halves(pieces):
    return torch.cat([pieces[0] * 2, pieces[1]], dim=1)


def _restacked(pieces):
    return torch.stack([pieces[i] for i in range(C)], dim=1)


class Method(nn.Module):
    def __init__(self, call):
        super().__init__()
        self.conv, self.act = nn.Conv2d(3, C, 3, padding=1), nn.ReLU()
        self.head = nn.Linear(C * S * S, 10)
        self.call = call

    def forward(self, x):
        return self.head(torch.flatten(self.call(self.act(self.conv(x))), 1))


METHODS = {
    "softmax": lambda y: y.softmax(dim=-1),
    "sigmoid": lambda y: y.sigmoid(),
    "relu": lambda y: y.relu(),
    "tanh": lambda y: y.tanh(),
    "clamp": lambda y: y.clamp(min=0.1, max=0.7),
    "chunk_getitems": lambda y: _halves(y.chunk(2, dim=1)),
    "chunk_uneven": lambda y: torch.cat(list(reversed([y.chunk(3, dim=1)[i] for i in range(3)])), dim=1),
    "torch_chunk": lambda y: torch.cat([y.chunk(2, 1)[1], torch.chunk(y, 2, dim=1)[0]], dim=1),
}


@pytest.mark.parametrize("key", sorted(METHODS))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_method_with_a_torch_function_converts_exactly(key, contract):
    assert_converted_exact(native(lambda: Method(METHODS[key])), (3, S, S), contract)


@pytest.mark.parametrize("key", sorted(METHODS))
def test_a_method_value_program_is_exact(key):
    assert_value_program_exact(native(lambda: Method(METHODS[key])), (3, S, S))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_chunk_read_through_getitems_becomes_narrows(contract):
    flow = assert_converted_exact(native(lambda: Method(METHODS["chunk_getitems"])), (3, S, S), contract)
    narrows = [m for m in flow.get_mapper_repr().execution_order()
               if isinstance(m, ComputeOpMapper) and getattr(m.module, "fn", None) is torch.narrow]
    assert len(narrows) == 2


REFUSED = {
    "chunk_other_user": lambda y: torch.cat(y.chunk(2, dim=1), dim=1),
    "split": lambda y: _halves(y.split(4, dim=1)),
    "unbind": lambda y: _restacked(y.unbind(1)),
}


@pytest.mark.parametrize("key", sorted(REFUSED))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_method_with_a_tuple_value_refuses_by_name(key, contract):
    from mimarsinan.torch_mapping.converter_handlers.rules.method_rules import TupleMethodError
    assert_refused_by_name(lambda: convert(native(lambda: Method(REFUSED[key])), (3, S, S), contract),
                           TupleMethodError, "returns a tuple")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_method_with_no_torch_function_keeps_its_refusal(contract):
    model = native(lambda: Method(lambda y: y.half().double()))
    assert_refused_by_name(lambda: convert(model, (3, S, S), contract), RepresentabilityError, ".half(")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_in_place_method_keeps_its_refusal(contract):
    model = native(lambda: Method(lambda y: y.clone().sigmoid_()))
    assert_refused_by_name(lambda: convert(model, (3, S, S), contract), RepresentabilityError, ".sigmoid_(")
