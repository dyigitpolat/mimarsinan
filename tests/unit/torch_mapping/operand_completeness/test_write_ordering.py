"""Writes in place in the native order of reads and writes, with no copy (review round 16, targeted pass).

The round-16 reviewer's witnesses (``review_r16/scripts/r16_clone.py``, ``r16_hw.py``):

- a helper whose operand every other reader reads before it received a copy (``Tensor.clone``), which keeps the values
  and not the layout (a view's strides, its offset): LayoutCloned, StrideCloned and OffsetCloned were refused at the
  probe, LayoutClonedLate and ClonedLateRealistic certified and wrong from batch 7, all exact on ``w0-base``;
- a declared write whose operand another node reads before it acted on ``torch.clone(operand)``, and every later
  reader read the clone, in a new storage and a contiguous layout: PureAsStridedAfterWriteCopyLate (a pure program) and
  DeclaredWriteCopyLayoutLate certified and wrong from batch 7;
- a helper's write the run at batch 1 observes, of an operand read after it, was refused (``OperandWriteError``):
  HWReturnsOperandAndRead, HWReadAfterAddDep, HWTwoHelpers and HWTwoHelpersNoOtherRead, exact on ``w0-base``.

Now every node that reads a value sharing the written operand's storage before a write (or before a call outside the
declared operator set) runs before it in the converted model (an ordering edge, ``Mapper.run_after``), the write acting
on the operand itself; and every reader of the operand after a call whose write the run observes reads it through a
value ordered after the call (``compute_modules.read_after``)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper, InputMapper, ModelRepresentation
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import relative_error
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING

from .harness import (
    CONTRACTS,
    EXACT,
    IMPURE_BATCHES,
    assert_converted_exact,
    assert_refused_by_name,
    assert_value_program_exact,
    convert,
    native,
    quietly,
)

ALL_BATCHES = tuple(range(1, 65))
RUN_AFTER = "mimarsinan_run_after"
"""``operand_writes.RUN_AFTER``, named here so the witnesses run on a tree without it (tests first)."""


def _read_after():
    from mimarsinan.mapping.support.compute_modules import read_after

    return read_after


class _Fc(nn.Module):
    def __init__(self, width: int = 6):
        super().__init__()
        self.fc = nn.Linear(8, width)


def _mapper(flow, name: str):
    return next(m for m in flow.get_mapper_repr().execution_order() if getattr(m, "name", None) == name)


def _assert_exact_at_every_batch(model: nn.Module, contract: str) -> None:
    flow = convert(model, (8,), contract)
    for batch in ALL_BATCHES:
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            err = relative_error(flow(x.clone()), model(x.clone())).relative_error
        assert err <= EXACT, (batch, err)


# --- a helper reads its operand's layout (blocker 1) ---------------------------------------------------------------


@torch.fx.wrap
def _by_layout(t):
    return t * (2.0 if t.is_contiguous() else 1.0)


@torch.fx.wrap
def _by_layout_late(t):
    return t * (2.0 if (t.is_contiguous() and t.shape[0] >= 7) else 1.0)


@torch.fx.wrap
def _by_stride(t):
    return t * float(t.stride(0))


@torch.fx.wrap
def _by_offset(t):
    return t * float(1 + t.storage_offset())


class LayoutCloned(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        return z[:, :3] + _by_layout(y[:, :3])


class LayoutClonedLate(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        return z[:, :3] + _by_layout_late(y[:, :3])


class ClonedLateRealistic(_Fc):
    """A helper choosing a code path by contiguity, with another result."""

    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        return z[:, 1:4] + _by_layout_late(y[:, 1:4])


class StrideCloned(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        return z[:, :3] + _by_stride(y[:, :3])


class OffsetCloned(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        return z[:, 3:] + _by_offset(y[:, 3:])


LAYOUT_HELPERS = [LayoutCloned, LayoutClonedLate, ClonedLateRealistic, StrideCloned, OffsetCloned]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", LAYOUT_HELPERS, ids=lambda f: f.__name__)
def test_a_helper_reading_its_operands_layout_after_every_other_read_is_exact(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=IMPURE_BATCHES, kinds=("normal", "wide"))


def test_the_helper_runs_after_the_earlier_reader_on_its_operand_itself():
    from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import RUN_AFTER as declared

    assert declared == RUN_AFTER
    gm = trace_model(native(LayoutClonedLate), (8,))
    helper = next(n for n in gm.graph.nodes if n.name == "_by_layout_late")
    assert helper.meta.get(RUN_AFTER) == ("mul",), helper.meta.get(RUN_AFTER)
    assert "mimarsinan_cloned_operands" not in helper.meta and "mimarsinan_kept_operands" not in helper.meta
    flow = convert(native(LayoutClonedLate), (8,), "mvm")
    mapper, reader = _mapper(flow, "_by_layout_late"), _mapper(flow, "mul")
    assert any(found is reader for found in mapper.run_after)
    order = flow.get_mapper_repr().execution_order()
    assert order.index(reader) < order.index(mapper)


# --- a declared write after another read (blocker 2) ---------------------------------------------------------------


class PureAsStridedAfterWriteCopyLate(_Fc):
    """Pure: ``torch.as_strided`` reads the written view's storage (natively ``y``'s) at an offset 3 from batch 7."""

    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        t = y[:, :3]
        F.relu(t, inplace=True)
        w = torch.as_strided(t, (1, 3), (3, 1), 3 * (t.shape[0] >= 7))
        return z[:, :3] + w


class PureAsStridedAfterWriteCopy(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        t = y[:, :3]
        F.relu(t, inplace=True)
        return z[:, :3] + torch.as_strided(t, (t.shape[0], 3), (3, 1))


class DeclaredWriteCopyLayoutLate(_Fc):
    """The in-place ReLU of a view another node read before; a helper reads the write's result's layout after it."""

    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        t = y[:, :3]
        F.relu(t, inplace=True)
        return z[:, :3] + _by_layout_late(t)


class DeclaredWriteCopyLayout(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 5
        t = y[:, :3]
        F.relu(t, inplace=True)
        return z[:, :3] + _by_layout(t)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [PureAsStridedAfterWriteCopyLate, PureAsStridedAfterWriteCopy],
                         ids=lambda f: f.__name__)
def test_a_pure_read_of_the_written_storage_after_the_write_is_exact_at_every_batch(factory, contract):
    _assert_exact_at_every_batch(native(factory), contract)


@pytest.mark.parametrize("factory", [PureAsStridedAfterWriteCopyLate, PureAsStridedAfterWriteCopy],
                         ids=lambda f: f.__name__)
def test_the_value_program_of_a_strided_read_of_the_storage_is_refused_by_name(factory):
    """``torch.as_strided`` reads a storage, which a value program, one buffer per value, does not hold: stage 2 is
    refused by name, as for the reviewer's control PureAsStridedNoWrite (no write)."""
    with pytest.raises(ToolchainRefusal, match="identity value program"):
        quietly(convert_torch_model, native(factory), (8,), 10, packaging=MVM_PACKAGING, value_program_stage=True)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [DeclaredWriteCopyLayoutLate, DeclaredWriteCopyLayout], ids=lambda f: f.__name__)
def test_a_helper_reading_the_layout_of_a_write_after_another_read_is_exact(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=IMPURE_BATCHES, kinds=("normal", "wide"))


def test_the_write_acts_on_its_operand_after_the_earlier_reader():
    gm = trace_model(native(PureAsStridedAfterWriteCopyLate), (8,))
    assert not [n for n in gm.graph.nodes if n.target is torch.clone]
    relu = next(n for n in gm.graph.nodes if n.target is F.relu)
    assert relu.args[0].name == "getitem" and relu.meta.get(RUN_AFTER) == ("mul",), relu.meta.get(RUN_AFTER)
    flow = convert(native(PureAsStridedAfterWriteCopyLate), (8,), "spiking")
    mapper, reader = _mapper(flow, relu.name), _mapper(flow, "mul")
    order = flow.get_mapper_repr().execution_order()
    assert any(found is reader for found in mapper.run_after) and order.index(reader) < order.index(mapper)


class FusionAcrossWrite(nn.Module):
    """``fc1`` reads ``y`` before the in-place ReLU, and its one reader ``fc2`` comes after it: fusing the pair would
    read ``y`` after the write."""

    def __init__(self):
        super().__init__()
        self.fc, self.fc1, self.fc2 = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        a = self.fc1(y)
        r = F.relu(y, inplace=True)
        return self.fc2(a) + r[:, :6]


@pytest.mark.parametrize("contract", CONTRACTS)
def test_no_linear_fusion_moves_a_read_across_a_write(contract):
    assert_converted_exact(native(FusionAcrossWrite), (8,), contract, batches=(1, 2, 3, 7, 13))
    gm = normalize_fx_graph(trace_model(native(FusionAcrossWrite), (8,)))
    assert [n.target for n in gm.graph.nodes if n.op == "call_module"] == ["fc", "fc1", "fc2"]


def test_the_value_program_of_a_write_after_another_read_is_exact():
    assert_value_program_exact(native(FusionAcrossWrite), (8,), batches=(1, 2, 3, 7))


def test_an_ordering_edge_orders_the_converted_forward():
    """``run_after``: a mapper runs after the mappers it names, which it does not read."""
    calls = []

    def recorder(label):
        def call(v):
            calls.append(label)
            return v
        return call

    source = InputMapper((4,))
    first = ComputeOpMapper(source, ComputeAdapter(recorder("first")), name="first")
    second = ComputeOpMapper(source, ComputeAdapter(recorder("second")), name="second")
    out = ComputeOpMapper([second, first], ComputeAdapter(torch.add), name="out", elementwise_inputs=False)
    assert [m.name for m in ModelRepresentation(out).execution_order()[1:]] == ["second", "first", "out"]
    second.run_after = (first,)
    model = ModelRepresentation(out)
    assert [m.name for m in model.execution_order()[1:]] == ["first", "second", "out"]
    model(torch.ones(2, 4))
    assert calls == ["first", "second"]


def test_a_cycle_of_ordering_edges_is_found():
    from mimarsinan.torch_mapping.converter_handlers.layout_mixin import _cycle

    source = InputMapper((4,))
    first = ComputeOpMapper(source, ComputeAdapter(torch.relu), name="first")
    second = ComputeOpMapper(first, ComputeAdapter(torch.relu), name="second")
    assert _cycle(second) is None
    first.run_after = (second,)
    assert _cycle(second) in (first, second)


# --- a helper's write the run observes, read after it (major 1) ----------------------------------------------------


@torch.fx.wrap
def _double_(t):
    t.mul_(2.0)
    return t.sum(1, keepdim=True)


@torch.fx.wrap
def _double_ret(t):
    t.mul_(2.0)
    return t


@torch.fx.wrap
def _add_one_(t):
    t.add_(1.0)
    return t * 0.5


@torch.fx.wrap
def _plain(t):
    return t * 0.25 + 1.0


class _Fc2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc, self.fc2 = nn.Linear(8, 8), nn.Linear(8, 6)


class HWReturnsOperandAndRead(_Fc2):
    def forward(self, x):
        y = self.fc(x)
        z = _double_ret(y)
        return z + y


class HWReadAfterAddDep(_Fc2):
    def forward(self, x):
        y = self.fc(x)
        s = _double_(y)
        return y + s


class HWTwoHelpers(_Fc2):
    def forward(self, x):
        y = self.fc(x)
        a = _add_one_(y)
        b = _double_(y)
        return a + b


class HWTwoHelpersNoOtherRead(_Fc2):
    def forward(self, x):
        y = self.fc(x)
        a = _add_one_(y)
        b = _plain(y)
        return a + b


class HWReadAfterMul(_Fc2):
    """Wrong on ``w0-base``."""

    def forward(self, x):
        y = self.fc(x)
        s = _double_(y)
        return y * 3 + s


class HWReadAfterLinear(_Fc2):
    def forward(self, x):
        y = self.fc(x)
        s = _double_(y)
        return self.fc2(y) + s


class HWBeforeAndAfter(_Fc2):
    def forward(self, x):
        y = self.fc(x)
        w = y * 5
        s = _double_(y)
        return w + y + s


class HWWritesThenReturnsOperand(_Fc2):
    """The helper's result is not read; the program returns the operand it wrote."""

    def forward(self, x):
        y = self.fc(x)
        _double_(y)
        return y


OBSERVED_WRITES = [HWReturnsOperandAndRead, HWReadAfterAddDep, HWTwoHelpers, HWTwoHelpersNoOtherRead, HWReadAfterMul,
                   HWReadAfterLinear, HWBeforeAndAfter, HWWritesThenReturnsOperand]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", OBSERVED_WRITES, ids=lambda f: f.__name__)
def test_an_observed_helper_write_read_after_is_exact(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=IMPURE_BATCHES, kinds=("normal", "wide"))


def test_a_reader_after_an_observed_write_reads_the_operand_ordered_after_the_call():
    gm = trace_model(native(HWReadAfterMul), (8,))
    helper = next(n for n in gm.graph.nodes if n.name == "_double_")
    after = [n for n in gm.graph.nodes if n.target is _read_after()]
    assert len(after) == 1 and after[0].args == (helper.args[0], helper), after
    mul = next(n for n in gm.graph.nodes if n.name == "mul")
    assert mul.args[0] is after[0] and "mimarsinan_kept_operands" not in helper.meta
    flow = convert(native(HWReadAfterMul), (8,), "mvm")
    order = flow.get_mapper_repr().execution_order()
    assert order.index(_mapper(flow, "_double_")) < order.index(_mapper(flow, after[0].name)) < order.index(
        _mapper(flow, "mul"))


def test_read_after_is_its_first_operand():
    t = torch.ones(2)
    assert _read_after()(t, object(), None) is t


@torch.fx.wrap
def _double_view_(t):
    t.mul_(2.0)
    return t.sum(1, keepdim=True)


class HWViewWriteReadBase(_Fc2):
    """The helper doubles a view of ``y``, and ``y`` itself, another value of the view's storage, is read after."""

    def forward(self, x):
        y = self.fc(x)
        s = _double_view_(y[:, :3])
        return y * 3 + s


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_read_of_another_value_of_the_written_storage_after_an_observed_write_is_refused_by_name(contract):
    assert_refused_by_name(lambda: convert(native(HWViewWriteReadBase), (8,), contract), ToolchainRefusal,
                           "_double_view_ (FX node '_double_view_')", "undeclared write", "'fc'", "after the write")
