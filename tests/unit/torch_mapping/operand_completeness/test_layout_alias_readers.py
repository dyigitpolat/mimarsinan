"""Every reader of a layout alias reads its native value (review round 15).

``_lower_layout`` carries ``unsqueeze``, ``squeeze`` and ``expand`` of an activation as their source's value (an alias
of another shape). Round 14 lowered such an alias as the host op of its native method before a generic host op, a
host-op module call or the output read it; the round-14 reviewer's ``r14_alias.py`` found the other readers: the
LayerNorm and BatchNorm host ops and ``torch.cat`` (major), and the static mappers, a permute or transpose, a reshape of
another element count, a perceptron and a convolution (minor), each crashing in the converted model's forward
(``RuntimeError`` at the mapper; ``w0-base`` refuses them at its probe). Now each reads the native value: exact at
batches 1 to 13 under both contracts, and the value program too. A reshape of the alias's own element count still
reads the alias (its element sequence is the alias's own).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ComputeOpMapper

from .harness import CONTRACTS, assert_converted_exact, assert_value_program_exact, convert, native

BATCHES = (1, 2, 3, 6, 7, 9, 13)


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)


class AliasLNUnsqueeze(_Fc):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(1)

    def forward(self, x):
        return self.ln(self.fc(x).unsqueeze(2)).flatten(1)


class AliasBNExpand(_Fc):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, x):
        return self.bn(self.fc(x).unsqueeze(1).expand(-1, 4, -1)).flatten(1)


class AliasCatBound(_Fc):
    def __init__(self):
        super().__init__()
        self.tok = nn.Parameter(torch.randn(1, 1, 6))

    def forward(self, x):
        a = self.fc(x).unsqueeze(1)
        return torch.cat([self.tok.expand(x.size(0), -1, -1), a], 1).transpose(1, 2).flatten(1)


class AliasPermute(_Fc):
    def forward(self, x):
        return self.fc(x).unsqueeze(2).permute(0, 2, 1).flatten(1)


class AliasExpandTranspose(_Fc):
    def forward(self, x):
        return self.fc(x).unsqueeze(1).expand(-1, 3, -1).transpose(1, 2).flatten(1)


class AliasExpandReshape(_Fc):
    def forward(self, x):
        return self.fc(x).unsqueeze(2).expand(-1, -1, 2).reshape(x.size(0), -1)


class AliasFlattenFunc(_Fc):
    def forward(self, x):
        return torch.flatten(self.fc(x).unsqueeze(2).expand(-1, -1, 2), 1)


class AliasConv(_Fc):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 2, 3)

    def forward(self, x):
        return self.conv(self.fc(x).unsqueeze(1)).flatten(1)


class AliasLinearExpand(_Fc):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(6, 5)

    def forward(self, x):
        return self.fc2(self.fc(x).unsqueeze(1).expand(-1, 3, -1)).flatten(1)


class AliasCatMixedT(_Fc):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(8, 6)

    def forward(self, x):
        return torch.cat([self.fc(x).unsqueeze(1), self.fc2(x).unsqueeze(1)], 1).transpose(1, 2).flatten(1)


class AliasFlattenSameCount(_Fc):
    """Control: a reshape of the alias's own element count reads the alias (no host op is added)."""

    def forward(self, x):
        return F.softmax(self.fc(x).unsqueeze(1).flatten(1), -1)


READERS = [AliasLNUnsqueeze, AliasBNExpand, AliasCatBound, AliasPermute, AliasExpandTranspose, AliasExpandReshape,
           AliasFlattenFunc, AliasConv, AliasLinearExpand, AliasCatMixedT, AliasFlattenSameCount]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", READERS, ids=lambda f: f.__name__)
def test_a_reader_of_a_layout_alias_reads_the_native_value(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=BATCHES, kinds=("normal", "uniform"))


@pytest.mark.parametrize("factory", READERS, ids=lambda f: f.__name__)
def test_the_value_program_of_a_layout_alias_reader_is_exact(factory):
    assert_value_program_exact(native(factory), (8,), batches=(1, 2, 3, 7))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_reshape_of_the_same_element_count_reads_the_alias(contract):
    flow = convert(native(AliasFlattenSameCount), (8,), contract)
    names = [m.name for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]
    assert "unsqueeze" not in names, names
