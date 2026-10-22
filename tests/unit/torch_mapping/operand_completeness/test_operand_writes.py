"""A call that writes a tensor operand in place is a new definition of that operand (review round 15).

The round-14 reviewer's witnesses (``review_r14/scripts/r14_mut.py``): a tensor resized in place by a public torch
function kept the shape its own node recorded, and FX's dead-code elimination dropped ``torch.resize_as_(z, y)``, so
``z.shape[0] < 7`` was folded to True and every conversion was wrong from batch 7 (MVM stage 2 passing); an in-place
clamp or ``nan_to_num_`` whose result was unused was dropped (wrong on inputs with a 1e9 or inf entry, on ``w0-base``
too). Now every reader after a write reads the write's result (``operand_writes.define_writes``); a value sharing the
written operand's storage read after the write is refused by name; a read before the write runs before it in the
converted model, the write acting on its operand itself [changed in review round 16, targeted pass: the write acted on
a copy, ``torch.clone``, which keeps the values and not the layout or the storage].
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
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import relative_error

from .harness import (CONTRACTS, EXACT, assert_converted_exact, assert_refused_by_name, assert_value_program_exact,
                      convert, native, quietly)

ALL_BATCHES = tuple(range(1, 65))


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)


class ResizeAsResize(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros(6, 6, dtype=y.dtype)
        torch.resize_as_(z, y)
        return y * (1 + (z.shape[0] < 7))


class AsStridedResize(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros(6, dtype=y.dtype)
        torch.as_strided_(z, (x.size(0),), (0,))
        return y * (1 + (z.shape[0] < 7))


class ResizeAsScratch(_Fc):
    def forward(self, x):
        y = self.fc(x)
        buf = torch.empty(0, dtype=y.dtype)
        torch.resize_as_(buf, y)
        return y * (1 + (buf.shape[0] < 7))


class AtenResize(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros(6, 6, dtype=y.dtype)
        torch.ops.aten.resize_.default(z, [x.size(0), 6])
        return y * (1 + (z.shape[0] < 7))


class AtenSetSource(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros(6, 6, dtype=y.dtype)
        torch.ops.aten.set_.source_Tensor(z, y)
        return y * (1 + (z.shape[0] < 7))


class ResizeAsUsed(_Fc):
    """The write's result is read too, in its first row: the reviewer's form sums every row, which from batch 7 reads
    memory ``resize_as_`` leaves uninitialized, natively too (no conversion can equal it there)."""

    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros(6, 6, dtype=y.dtype)
        w = torch.resize_as_(z, y)
        return y * (1 + (z.shape[0] < 7)) + 0 * torch.sum(w[:1])


@torch.fx.wrap
def _scale(v):
    return v * 1.0


class ResizeAsImpure(_Fc):
    """Made impure by a user helper: nothing is folded, and the resize ran nowhere in the conversion."""

    def forward(self, x):
        y = _scale(self.fc(x))
        z = torch.zeros(6, 6, dtype=y.dtype)
        torch.resize_as_(z, y)
        return y * (1 + (z.shape[0] < 7))


class AtenUnsqueezeMasked(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros_like(y)
        torch.ops.aten.unsqueeze_.default(z, 0)
        return y * (1 + (z.shape[1] < 7))


class ResizeAsView(_Fc):
    def forward(self, x):
        y = self.fc(x)
        z = torch.zeros(6, 6, dtype=y.dtype)
        torch.resize_as_(z, y.transpose(0, 1))
        return F.softmax(y.view(-1, 1, 6 // (1 + (z.shape[1] >= 7))), -1).flatten(1)


class InplaceClampHuge(_Fc):
    def forward(self, x):
        y = self.fc(x)
        torch.clamp_(y, -1e5, 1e5)
        return y * 2


class InplaceNanToNum(_Fc):
    def forward(self, x):
        y = self.fc(x)
        torch.nan_to_num_(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return F.softmax(y, -1)


RESIZES = [ResizeAsResize, AsStridedResize, ResizeAsScratch, AtenResize, AtenSetSource, ResizeAsUsed, ResizeAsImpure,
           AtenUnsqueezeMasked]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", RESIZES, ids=lambda f: f.__name__)
def test_a_resize_in_place_is_exact_at_every_batch(factory, contract):
    """Certified and wrong from batch 7 on the round-14 tree (the literals getitem=6, lt=True, add=2); now exact at
    batches 1 to 64, with no literal: ``z.shape`` reads the write's result, an opaque batch value (or the program is
    impure)."""
    flow = assert_converted_exact(native(factory), (8,), contract, batches=ALL_BATCHES, kinds=("normal", "uniform"))
    assert flow.shape_literals == {}, flow.shape_literals


@pytest.mark.parametrize("factory", RESIZES, ids=lambda f: f.__name__)
def test_the_value_program_of_a_resize_is_refused_by_name(factory):
    """MVM stage 2 passed with its value program wrong (ResizeAsResize, AsStridedResize, ResizeAsScratch): the resized
    extent is opaque now (or the program impure), so stage 2 is refused by name."""
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, copy.deepcopy(native(factory)), (8,), 10, packaging=MVM_PACKAGING,
                value_program_stage=True)
    assert "OpaqueBatchValueError" in str(excinfo.value) or "ImpureShapeProgramError" in str(excinfo.value)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_resized_extent_in_a_view_is_refused_by_name(contract):
    """ResizeAsView: the stale ``z.shape[1]`` made a static view wrong from batch 7, stage 2 passing; now the resize
    of a value whose batch axis is not axis 0 is refused by name."""
    assert_refused_by_name(lambda: convert(native(ResizeAsView), (8,), contract), ToolchainRefusal, "transpose")


def _huge_inputs(batch: int, value: float) -> torch.Tensor:
    x = torch.randn(batch, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(batch))
    x[0, 0] = value
    return x


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("value", [1e9, float("inf"), -float("inf")])
@pytest.mark.parametrize("factory", [InplaceClampHuge, InplaceNanToNum], ids=lambda f: f.__name__)
def test_an_in_place_value_write_is_kept(factory, value, contract):
    """Dead-code elimination dropped the write: wrong by rel 2290 (or inf) on inputs with a 1e9 or inf entry, on
    ``w0-base`` too. Exact at batches 1 to 64 now."""
    model = native(factory)
    flow = convert(model, (8,), contract)
    for batch in ALL_BATCHES:
        x = _huge_inputs(batch, value)
        with torch.no_grad():
            err = relative_error(flow(x.clone()), model(x.clone())).relative_error
        assert err <= EXACT, (batch, err)


class ReluInplaceMultiUser(_Fc):
    """``y + relu_(y)`` is ``2 relu(y)`` natively; its value program was wrong on the round-14 tree."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fc(x)
        a = self.act(y)
        return y + a


class ReluInplaceReadBefore(_Fc):
    """``b = y * 3`` reads ``y`` before the write: refused by the probe on the round-14 tree (the converted model ran
    the in-place host op first)."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fc(x)
        b = y * 3
        a = self.act(y)
        return a + b


class ReluFunctionalInplaceUnused(_Fc):
    def forward(self, x):
        y = self.fc(x)
        F.relu(y, True)
        return y * 2


class ResNetStyleInplace(_Fc):
    """The common form: an in-place activation whose operand nothing else reads."""

    def __init__(self):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(6, 4)

    def forward(self, x):
        return self.fc2(self.act(self.fc(x)))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", [ReluInplaceMultiUser, ReluInplaceReadBefore, ReluFunctionalInplaceUnused,
                                     ResNetStyleInplace], ids=lambda f: f.__name__)
def test_an_in_place_activation_is_exact(factory, contract):
    assert_converted_exact(native(factory), (8,), contract, batches=(1, 2, 3, 7, 13))


def test_a_read_before_the_write_runs_before_it():
    """[changed in review round 16, targeted pass: the write acted on ``torch.clone(fc)``] The write acts on its
    operand itself, ordered after the node that reads it before (``RUN_AFTER``)."""
    from mimarsinan.torch_mapping.converter_handlers.symbolic.operand_writes import RUN_AFTER

    gm = trace_model(native(ReluInplaceReadBefore), (8,))
    act = next(node for node in gm.graph.nodes if node.name == "act")
    assert act.args[0].name == "fc" and act.meta.get(RUN_AFTER) == ("mul",), (act.args, act.meta.get(RUN_AFTER))
    assert not [node for node in gm.graph.nodes if node.target is torch.clone]


def test_the_common_form_is_traced_as_before():
    """No copy and no rewrite where nothing else reads the written operand: the traced graph is FX's own."""
    gm = trace_model(native(ResNetStyleInplace), (8,))
    assert [(n.op, str(n.target), [str(a) for a in n.args]) for n in gm.graph.nodes] == [
        ("placeholder", "x", []), ("call_module", "fc", ["x"]), ("call_module", "act", ["fc"]),
        ("call_module", "fc2", ["act"]), ("output", "output", ["fc2"])]


def test_the_value_programs_of_in_place_activations_are_exact():
    for factory in (ReluInplaceMultiUser, ReluInplaceReadBefore):
        assert_value_program_exact(native(factory), (8,), batches=(1, 2, 3, 7))


class ReluInplaceOnView(_Fc):
    """A write through a view changes ``y`` natively."""

    def forward(self, x):
        y = self.fc(x)
        torch.relu_(y[:, :3])
        return y * 2


class ReluInplaceViewReadAfter(_Fc):
    """A view taken before the write is read after it."""

    def forward(self, x):
        y = self.fc(x)
        v = y.view(-1, 2, 3)
        torch.relu_(y)
        return v.flatten(1) * 2


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory, read", [(ReluInplaceOnView, "'fc'"), (ReluInplaceViewReadAfter, "'flatten'")],
                         ids=["through-a-view", "view-read-after"])
def test_a_value_sharing_the_storage_read_after_the_write_is_refused_by_name(factory, read, contract):
    assert_refused_by_name(lambda: convert(native(factory), (8,), contract), ToolchainRefusal,
                           "relu_ (FX node 'relu_')", read, "after the write")


@torch.fx.wrap
def _clamp_in_place(v):
    v.clamp_(-1e5, 1e5)
    return v.sum()


class HelperWritesReadOperand(_Fc):
    """A user helper clamps its operand in place, and ``y`` is read after it, through a value ordered after the call
    (review round 16, targeted pass)."""

    def forward(self, x):
        y = self.fc(x)
        s = _clamp_in_place(y)
        return y * 2 + 0 * s


@torch.fx.wrap
def _relu_in_place(v):
    return v.relu_()


class HelperWritesOwnOperand(_Fc):
    """A user helper writes an operand nothing else reads (``w0-base``'s exact form)."""

    def forward(self, x):
        return _relu_in_place(self.fc(x)) * 2


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_undeclared_write_of_an_operand_read_after_it_is_exact(contract):
    """[changed in review round 16, targeted pass: refused by name, ``OperandWriteError``; a read of another value of
    the written storage after the write still is, ``test_write_ordering.py``]"""
    assert_converted_exact(native(HelperWritesReadOperand), (8,), contract, batches=(1, 2, 3, 7, 13))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_undeclared_write_nothing_else_reads_is_exact(contract):
    assert_converted_exact(native(HelperWritesOwnOperand), (8,), contract, batches=(1, 2, 3, 7))


@torch.fx.wrap
def _clamp_from_batch_7(v):
    if v.shape[0] >= 7:
        v.clamp_(-1e5, 1e5)
    return v.sum()


class HelperWritesLate(_Fc):
    """A user helper that writes its operand from batch 7 only: the run at batch 1 shows no write."""

    def forward(self, x):
        y = self.fc(x)
        s = _clamp_from_batch_7(y)
        return y * 2 + 0 * s


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_undeclared_write_the_run_does_not_see_is_refused_in_the_converted_forward(contract):
    """Exact below batch 7; from batch 7 the helper's host op writes an operand another node reads, and the converted
    forward refuses the input by name (``HostOpWriteError``) where it would otherwise read ``y`` before the write."""
    from mimarsinan.common.refusal import refusal_in_chain
    from mimarsinan.mapping.support.compute_modules import HostOpWriteError

    model = native(HelperWritesLate)
    flow = assert_converted_exact(model, (8,), contract, batches=(1, 2, 3, 6))
    with pytest.raises(RuntimeError) as excinfo:  # a refusal in a forward reaches the caller wrapped, by name
        with torch.no_grad():
            flow(_huge_inputs(7, 1e9))
    refusal = refusal_in_chain(excinfo.value)
    assert isinstance(refusal, HostOpWriteError), excinfo.value
    assert "_clamp_from_batch_7 (FX node '_clamp_from_batch_7')" in str(refusal), str(refusal)
