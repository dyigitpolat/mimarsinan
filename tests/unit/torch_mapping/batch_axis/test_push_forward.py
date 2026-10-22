"""A permute moving the batch off axis 0, read only back into a batch-leading form, converts exactly.

On the base each of these refused by name (``BatchAxisLayoutError`` through ``_require_batch_leading``). The rewrite
reads ``qkv.permute(2, 0, 3, 1, 4)[i]`` as ``qkv.select(2, i).permute(0, 2, 1, 3)`` and a second permute composed with
the first; any other consumer keeps the traced permute and refuses by name as before, and no activation permute
becomes a ``ReshapeMapper``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import ReshapeMapper
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_refused_by_name,
    assert_value_program_exact,
    convert,
    native,
)

N, D, H = 5, 8, 2


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.head = nn.Linear(N * D, 10)

    def heads(self, x):
        b, n, _ = x.shape
        return self.qkv(x).reshape(b, n, 3, H, D // H).permute(2, 0, 3, 1, 4)


class Getitem(Base):
    """timm-style: the three heads by ``getitem``, then batch-leading products."""

    def forward(self, x):
        qkv = self.heads(x)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
        return self.head(torch.flatten(o.transpose(1, 2), 1))


class Unbind(Base):
    """Newer timm: ``qkv.unbind(0)``."""

    def forward(self, x):
        q, k, v = self.heads(x).unbind(0)
        o = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
        return self.head(torch.flatten(o.transpose(1, 2), 1))


class SecondPermute(nn.Module):
    """A transpose of the batch away and straight back (a second permute restores it)."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(D, D)
        self.head = nn.Linear(N * D, 10)

    def forward(self, x):
        y = self.a(x).transpose(0, 1)
        return self.head(torch.flatten(y.permute(1, 0, 2), 1))


class Both(Base):
    """``getitem`` and a second permute of one batch-moving permute."""

    def forward(self, x):
        qkv = self.heads(x)
        q = qkv[0]
        kv = qkv.permute(1, 0, 2, 3, 4)  # (B, 3, H, N, d): the batch back on axis 0
        o = torch.softmax(q @ kv[:, 1].transpose(-2, -1), dim=-1) @ kv[:, 2]
        return self.head(torch.flatten(o.transpose(1, 2), 1))


class OtherConsumer(Base):
    """A reshape of the batch-moved value: no batch-leading form, still refused by name."""

    def forward(self, x):
        y = self.heads(x).reshape(3, -1)
        return self.head(y[0].reshape(-1, N * D))


class UnsqueezeConsumer(Base):
    """An ``unsqueeze`` of the batch-moved value (a layout alias): never routed around the batch check."""

    def forward(self, x):
        y = self.heads(x).unsqueeze(0)
        return self.head(torch.flatten(y[0, 0].transpose(1, 2), 1))


CASES = {"getitem": Getitem, "unbind": Unbind, "second_permute": SecondPermute, "both": Both}


@pytest.mark.parametrize("key", sorted(CASES))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_batch_moving_permute_read_back_converts_exactly(key, contract):
    flow = assert_converted_exact(native(CASES[key]), (N, D), contract)
    assert not [m for m in flow.get_mapper_repr().execution_order() if isinstance(m, ReshapeMapper)
                and "permute" in str(getattr(m, "name", ""))]


@pytest.mark.parametrize("key", sorted(CASES))
def test_a_batch_moving_permute_read_back_value_program_is_exact(key):
    assert_value_program_exact(native(CASES[key]), (N, D))


@pytest.mark.parametrize("factory", [OtherConsumer, UnsqueezeConsumer], ids=["reshape", "unsqueeze"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_any_other_consumer_still_refuses_by_name(factory, contract):
    assert_refused_by_name(lambda: convert(native(factory), (N, D), contract), BatchAxisLayoutError, "permute")


def test_no_reshape_mapper_is_emitted_for_an_activation_permute():
    """The ``ReshapeMapper`` branch of a permute whose dims do not start at 0 is gone: every such activation permute
    refuses by name before it (Wave 0's ``_require_batch_leading``), and a parameter's refuses by name too."""

    class ParamPermute(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(D, D)
            self.p = nn.Parameter(torch.randn(N, 1, D))

        def forward(self, x):
            return self.a(x) + self.p.permute(1, 0, 2)

    assert_refused_by_name(lambda: convert(native(ParamPermute), (N, D), "mvm"), BatchAxisLayoutError, "permute")
