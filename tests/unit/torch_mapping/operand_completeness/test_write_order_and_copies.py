"""Writes in place against the native order of reads and writes, copies a may-alias annotation calls an alias, and
torch.inference_mode() (review round 16).

The round-15 reviewer's witnesses (``review_r15/scripts/r15_mut.py``, ``r15_inference_mode.py``), each exact on
``w0-base`` and refused or crashing on round 15:

- a user helper's write after every other read of its operand (HelperWriteAfterRead, observed at batch 1, refused at
  conversion; HelperWriteAfterReadLate, from batch 7 only, refused in the converted forward): the helper's host op now
  runs after every other read of the operand, on the operand itself, so its write lands after every other read, as
  natively, at every batch [changed in review round 16, targeted pass: the host op received a copy, which keeps the
  values and not the layout; ``test_write_ordering.py``];
- a ``reshape`` or ``contiguous()`` torch's schema lets share its operand's storage, which copies at every batch
  (ReshapeCopyWrite, ContiguousCopyWriteSource; ContiguousCopyWrite, wrong on ``w0-base``): refused by
  ``OperandWriteError``; now taken as the copy the traced run shows, and guarded at run time;
- ``torch.inference_mode()``: every conversion failed (an inference tensor keeps no version counter).

Boundary witnesses of hypothesis T3 (``value_parity/certificate.py``), not findings: HelperAliasLate and
HelperAliasLateModule (a helper returning its operand itself from batch 7 only), HelperDataWriteLate and
HelperNumpyWriteLate (a write through ``Tensor.data`` or a numpy view, which the version counter does not count).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.mapping.support import compute_modules
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import relative_error

from .harness import (
    CONTRACTS,
    EXACT,
    IMPURE_BATCHES,
    assert_converted_exact,
    assert_value_program_exact,
    convert,
    native,
)

ALL_BATCHES = tuple(range(1, 65))


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)


@torch.fx.wrap
def _clamp_(v):
    v.clamp_(-0.1, 0.1)
    return v.sum(1, keepdim=True)


@torch.fx.wrap
def _clamp_from_7(v):
    if v.shape[0] >= 7:
        v.clamp_(-0.1, 0.1)
    return v.sum(1, keepdim=True)


class HelperWriteAfterRead(_Fc):
    """Every other reader of ``t`` reads it before the helper clamps it, natively."""

    def forward(self, x):
        t = self.fc(x)
        z = t * 2
        s = _clamp_(t)
        return z + s


class HelperWriteAfterReadLate(_Fc):
    def forward(self, x):
        t = self.fc(x)
        z = t * 2
        s = _clamp_from_7(t)
        return z + s


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [HelperWriteAfterRead, HelperWriteAfterReadLate], ids=lambda f: f.__name__)
def test_a_helper_write_after_every_other_read_is_exact(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=IMPURE_BATCHES, kinds=("normal", "wide"))


def test_the_helpers_host_op_runs_after_every_other_read_of_its_operand():
    """[changed in review round 16, targeted pass: the host op received a copy, ``CLONED_OPERANDS``]"""
    from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import RUN_AFTER

    gm = trace_model(native(HelperWriteAfterReadLate), (8,))
    helper = next(n for n in gm.graph.nodes if n.name == "_clamp_from_7")
    assert helper.meta.get(RUN_AFTER) == ("mul",) and "mimarsinan_kept_operands" not in helper.meta
    assert "mimarsinan_cloned_operands" not in helper.meta
    flow = convert(native(HelperWriteAfterReadLate), (8,), "mvm")
    order = flow.get_mapper_repr().execution_order()
    mapper = next(m for m in order if getattr(m, "name", "") == "_clamp_from_7")
    reader = next(m for m in order if getattr(m, "name", "") == "mul")
    assert mapper.kept_inputs == () and any(m is reader for m in mapper.run_after)
    assert order.index(reader) < order.index(mapper)


def test_the_helpers_host_op_receives_its_inputs_themselves():
    """[changed in review round 16, targeted pass: a tensor given twice was given as one copy twice]"""
    seen = []
    t = torch.ones(2)
    compute_modules.unwritten_call(lambda given: seen.extend(given), (t, t), (), "f", "f")
    assert seen[0] is t and seen[1] is t


class ContiguousCopyWrite(_Fc):
    """``contiguous()`` copies at every batch (a transpose of 2 x 3 per sample); the in-place ReLU writes the copy."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fc(x)
        z = y.view(-1, 2, 3).transpose(1, 2).contiguous()
        z = self.act(z)
        return z.flatten(1) + y


class ReshapeCopyWrite(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y.view(-1, 2, 3).permute(0, 2, 1).reshape(-1, 6)
        z = F.relu(z, inplace=True)
        return z + y


class ContiguousCopyWriteSource(_Fc):
    """The source is written in place after a copying ``contiguous()`` of a view of it; the copy is read after."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fc(x)
        z = y.view(-1, 2, 3).transpose(1, 2).contiguous()
        y = self.act(y)
        return z.flatten(1) + y


COPIES = [ContiguousCopyWrite, ReshapeCopyWrite, ContiguousCopyWriteSource]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", COPIES, ids=lambda f: f.__name__)
def test_a_copy_a_may_alias_annotation_calls_an_alias_is_exact_at_every_batch(factory, contract):
    model = native(factory)
    flow = convert(model, (8,), contract)
    for batch in ALL_BATCHES:
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            err = relative_error(flow(x.clone()), model(x.clone())).relative_error
        assert err <= EXACT, (batch, err)


@pytest.mark.parametrize("factory", COPIES, ids=lambda f: f.__name__)
def test_the_value_program_of_a_guarded_copy_is_exact(factory):
    assert_value_program_exact(native(factory), (8,), batches=(1, 2, 3, 7))


def test_a_copy_the_rule_relies_on_is_the_guarded_host_op_of_its_native_method():
    from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import COPY_GUARD

    gm = trace_model(native(ReshapeCopyWrite), (8,))
    guarded = {n.name: n.meta[COPY_GUARD] for n in gm.graph.nodes if COPY_GUARD in n.meta}
    assert guarded == {"reshape": ("permute", (6, 1, 3))}, guarded
    flow = convert(native(ReshapeCopyWrite), (8,), "mvm")
    mapper = next(m for m in flow.get_mapper_repr().execution_order() if getattr(m, "name", "") == "reshape")
    assert isinstance(mapper, ComputeOpMapper) and mapper.copy_guard == (0, (6, 1, 3), "reshape")


def test_a_copy_no_rule_relies_on_is_lowered_as_before():
    """The common form: a copying reshape with no write keeps its static reshape mapper and no guard."""

    class PermuteReshape(_Fc):
        def forward(self, x):
            return self.fc(x).view(-1, 2, 3).permute(0, 2, 1).reshape(-1, 6) * 2

    gm = trace_model(native(PermuteReshape), (8,))
    assert not [n for n in gm.graph.nodes if "mimarsinan_copy_guard" in n.meta]
    flow = assert_converted_exact(native(PermuteReshape), (8,), "mvm")
    assert not [m for m in flow.get_mapper_repr().execution_order() if getattr(m, "copy_guard", None) is not None]


def test_the_copy_guard_refuses_a_view_and_another_layout_by_name():
    from mimarsinan.mapping.support.compute_modules import HostOpAliasError, checked_copy

    base = torch.randn(4, 2, 3, dtype=torch.float64)
    permuted = base.permute(0, 2, 1)
    guard = (0, (6, 1, 3), "reshape")
    checked_copy(permuted.reshape(-1, 6), (permuted,), guard, "reshape")  # a copy at the traced strides
    with pytest.raises(HostOpAliasError, match=r"reshape \(FX node 'reshape'\): it returned a value sharing"):
        checked_copy(permuted, (permuted,), guard, "reshape")
    contiguous = permuted.contiguous()
    with pytest.raises(HostOpAliasError, match="has the strides"):
        checked_copy(contiguous.reshape(-1, 6), (contiguous,), guard, "reshape")


def test_sum_to_size_may_return_its_operand():
    from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes.declared import view_aliases

    class SumToSize(_Fc):
        def forward(self, x):
            t = self.fc(x)
            return t.sum_to_size(t.shape[0], 6)

    gm = trace_model(native(SumToSize), (8,))
    node = next(n for n in gm.graph.nodes if n.target == "sum_to_size")
    assert [a.name for a in view_aliases(node)] == ["fc"]


@torch.fx.wrap
def _row_sum(t):
    return t.sum(1, keepdim=True)


class HelperKept(_Fc):
    def forward(self, x):
        t = self.fc(x)
        return t * 2 + _row_sum(t)


class ReluInplaceReadBefore(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = y * 2
        return z + F.relu(y, inplace=True)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(8, 6), nn.ReLU(inplace=True), nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [Mlp, HelperKept, ReluInplaceReadBefore, HelperWriteAfterRead],
                         ids=lambda f: f.__name__)
def test_conversion_and_the_converted_forward_work_under_inference_mode(factory, contract):
    model = native(factory)
    with torch.inference_mode():
        flow = convert(model, (8,), contract)
    for batch in (1, 2, 3, 7, 9):
        x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            want = model(x.clone())
        for context in (torch.no_grad, torch.inference_mode):
            with context():
                err = relative_error(flow(x.clone()), want).relative_error
            assert err <= EXACT, (batch, context.__name__, err)


def test_an_inference_tensors_write_counter_is_its_bytes():
    from mimarsinan.mapping.support.compute_modules import write_counter

    with torch.inference_mode():
        t = torch.zeros(3)
        before = write_counter(t)
        t.add_(1)
        assert write_counter(t) != before and isinstance(before, bytes)
    assert write_counter(torch.zeros(3)) == 0
