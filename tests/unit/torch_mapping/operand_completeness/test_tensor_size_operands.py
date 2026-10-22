"""A tensor given to a kernel as a size argument is data, never a static extent (review round 15).

The round-14 reviewer's witnesses (``review_r14/scripts/r14_dd.py``): a 0-dim integer tensor as a pad amount, an
interpolation size, an adaptive pool's output size or a one-hot class count. Torch's FakeTensor run raises on it, and
the round-14 pass took the seeded traces as the stand-in: its ``LEADING`` rule copied the trailing extent from the
batch-1 trace, and a 0-dim tensor operand counted as batch-free. A data-dependent extent the three traces agreed on
became a static literal: PadTensorLate2 wrong at every batch from 7 (MVM stage 2 passing), the ``*Huge`` forms wrong on
every input with an entry above 1e6. Now an extent read from the traces is opaque, and a tensor where a kernel reads a
number makes its rule state nothing.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH, Opaque
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.kernel_rules import leading, padded
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain.shapes import Call, Shape
from mimarsinan.torch_mapping.value_parity import relative_error

from .harness import CONTRACTS, EXACT, assert_converted_exact, convert, native, quietly

ALL_BATCHES = tuple(range(1, 65))


def _huge(x):
    """1 when some |x| exceeds 1e6 (never on the probe's inputs), else 0: a 0-dim int64 tensor."""
    return torch.add(torch.gt(torch.amax(torch.abs(x)), 1e6), 0)


def _late(x):
    """1 from batch 7 on, else 0: the batch counted as data (a sum of ones)."""
    return torch.add(torch.ge(torch.sum(torch.ones_like(x[:, 0])), 7), 0)


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)


class PadTensorLate2(_Fc):
    def forward(self, x):
        y = self.fc(x)
        return y * F.pad(y, (0, _late(x))).shape[-1]


class PadTensorHuge(_Fc):
    def forward(self, x):
        y = self.fc(x)
        return y * F.pad(y, (0, 1 + _huge(x))).shape[-1]


class PadTensorCount(_Fc):
    def forward(self, x):
        y = self.fc(x)
        return y * F.pad(y, (0, torch.sum(torch.gt(x[:, 0], 1e6)))).shape[-1]


class InterpTensorHuge(_Fc):
    def forward(self, x):
        y = self.fc(x)
        return y * F.interpolate(y.unsqueeze(1), size=6 + _huge(x)).shape[-1]


class AdaptivePoolTensorHuge(_Fc):
    def forward(self, x):
        y = self.fc(x)
        return y * F.adaptive_avg_pool1d(y.unsqueeze(1), 3 + _huge(x)).shape[-1]


class OneHotTensorHuge(_Fc):
    def forward(self, x):
        y = self.fc(x)
        return y * F.one_hot(torch.zeros_like(y, dtype=torch.long), 4 + _huge(x)).shape[-1]


class PadIntControl(_Fc):
    """Control: a pad of plain integers keeps its static extent (a literal, as before)."""

    def forward(self, x):
        y = self.fc(x)
        return y * F.pad(y, (0, 1)).shape[-1]


WITNESSES = [PadTensorLate2, PadTensorHuge, PadTensorCount, InterpTensorHuge, AdaptivePoolTensorHuge,
             OneHotTensorHuge]


def _inputs(batch: int, huge: bool) -> torch.Tensor:
    x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
    if huge:
        x[0, 0] = 1e7
    return x


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", WITNESSES, ids=lambda f: f.__name__)
def test_a_tensor_size_argument_is_exact_at_every_batch_and_input(factory, contract):
    model = native(factory)
    flow = convert(model, (8,), contract)
    assert not [name for name, (operator, _) in flow.shape_literals.items() if operator.startswith("getitem")], \
        flow.shape_literals  # no literal for z.shape[-1]
    for batch in ALL_BATCHES:
        for huge in (False, True):
            x = _inputs(batch, huge)
            with torch.no_grad():
                err = relative_error(flow(x.clone()), model(x.clone())).relative_error
            assert err <= EXACT, (batch, huge, err)


@pytest.mark.parametrize("factory", WITNESSES, ids=lambda f: f.__name__)
def test_the_value_program_of_a_tensor_size_argument_is_refused_by_name(factory):
    """PadTensorLate2's MVM stage 2 passed with its value program wrong: the extent is opaque now."""
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, copy.deepcopy(native(factory)), (8,), 10, packaging=MVM_PACKAGING,
                value_program_stage=True)
    assert "OpaqueBatchValueError" in str(excinfo.value), str(excinfo.value)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_pad_of_integers_keeps_its_static_extent(contract):
    flow = assert_converted_exact(native(PadIntControl), (8,), contract, batches=(1, 2, 3, 7, 13))
    assert flow.shape_literals.get("getitem") == ("getitem", "7"), flow.shape_literals


def test_a_tensor_operand_where_a_kernel_reads_a_number_makes_the_rule_state_nothing():
    zero_dim = Shape(())
    y = Shape((BATCH, 6))
    assert padded(Call("pad", (y, (0, zero_dim)))) is None
    assert padded(Call("pad", (y, (0, 1)))) == 1
    rule = leading(None, lambda call: 1)
    assert rule(Call("interpolate", (y,), {"size": zero_dim}, ((7, 7),))) is None
    assert rule(Call("interpolate", (y,), {"size": 7}, ((7, 7),))) == [(BATCH, 7)]
    linear = leading(1, operands=("input", "weight", "bias"))
    assert linear(Call("linear", (y, Shape((4, 6)), Shape((4,))), {}, ((7, 4),))) == [(BATCH, 4)]


def test_an_extent_from_the_seeded_traces_is_never_stated():
    """Where torch's FakeTensor run raised, the pass hands the rule ``Opaque`` extents for ``out``: a rule that reads
    torch's extent states an opaque one, never the batch-1 trace's."""
    stand_in = Opaque("an extent the seeded traces show (a sample)")
    found = leading(None, lambda call: 1)(Call("pad", (Shape((BATCH, 6)), (0, 1)), {}, ((stand_in, stand_in),)))
    assert found == [(BATCH, stand_in)]
