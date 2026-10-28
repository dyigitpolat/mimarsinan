"""Metadata reads (coverage W8): a factory of an activation's dtype and device is a batch-free value.

On the base (W6), ``x.new_zeros(size)`` read the input, so it was no batch-free value; with no method rule for it, the
model was refused by name (``RepresentabilityError``). Swin's shifted-window attention builds its mask this way. Now a
receiver factory (``new_zeros``, ``new_ones``, ``new_full``) whose operands but the receiver are literals reads only
the receiver's dtype and device: it is batch-free (its receiver is not part of the computation), computed at every
call on a zero-element stand-in of the receiver's traced dtype in the converted model's state, so it follows the
model's casts, and a constant of the deployed program. A factory whose size the program reads from the input, or whose
receiver a call outside the declared set reaches, is not batch-free: the factory is the host op of its method, exact,
and a value without a batch axis is refused by name at stage 2.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_free import BatchFreeValues, batch_free_nodes
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_value_program_exact,
    convert,
    inputs,
    native,
    quietly,
)

C = 6
SHAPE = (4, C)


class _Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(C, C)
        self.head = nn.Linear(4 * C, 3)

    def finish(self, y):
        return self.head(torch.flatten(y, 1))


class Zeros(_Base):
    def forward(self, x):
        y = self.a(x)
        return self.finish(y + (y.new_zeros((4, C)) + 0.5).cumsum(0))


class Ones(_Base):
    def forward(self, x):
        y = self.a(x)
        return self.finish(y * y.new_ones(C))


class Full(_Base):
    def forward(self, x):
        y = self.a(x)
        return self.finish(y + y.new_full((4, 1), 0.1))


class FromTheInput(_Base):
    """A size the program reads from the input: not a metadata read."""

    def forward(self, x):
        y = self.a(x)
        return self.finish(y + y.new_ones(y.size(-1)))


@torch.fx.wrap
def opaque_cast(t):
    if bool(t.reshape(-1)[:1].any()):
        pass
    return t


class OpaqueReceiver(_Base):
    """The receiver is the value of a call outside the declared set: its dtype is that call's."""

    def forward(self, x):
        y = opaque_cast(self.a(x))
        return self.finish(y + y.new_ones(C))


class CollideMetaFirst(_Base):
    """A root parameter with the name a float64 stand-in would take (``_receiver_float64``), read by a batch-free
    value after the metadata read: the value reads the parameter, never the stand-in (soundness review S1)."""

    def __init__(self):
        super().__init__()
        self._receiver_float64 = nn.Parameter(torch.randn(C))

    def forward(self, x):
        y = self.a(x)
        return self.finish((y + y.new_ones(C)) * self._receiver_float64.sum())


class CollideConeFirst(_Base):
    """The same name on a parameter a batch-free value reads before the metadata read: the stand-in is still a
    zero-element tensor of its receiver's dtype, beside the parameter (S1)."""

    def __init__(self):
        super().__init__()
        self._receiver_float64 = nn.Parameter(torch.arange(C, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        y = self.a(x)
        y = y + self._receiver_float64 * 2.0
        return self.finish(y + y.new_full((4, 1), 0.5))


class PinnedFactory(_Base):
    """The receiver's dtype pinned by a declared factory with a ``dtype`` argument (soundness review S2,
    ``11_wave2/swin/review_soundness/p3_standin_cast.py``, case pinned_factory): a cast of the model leaves it float32."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(C))

    def forward(self, x):
        y = self.a(x)
        return self.finish(y + torch.zeros_like(self.p, dtype=torch.float32).new_full((4, 1), 0.1))


class MixedPromotion(_Base):
    """A receiver whose dtype follows some casts of the model only: the state promoted with a value the program pins
    to float32 (built in float64 it is float64; a float32 cast gives float32, a float16 cast float32 too)."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(C))

    def forward(self, x):
        y = self.a(x)
        return self.finish(y + (self.p + torch.zeros_like(self.p, dtype=torch.float32)).new_full((4, 1), 0.1))


class FreeBesideOpaque(_Base):
    """A batch-free metadata read (of a parameter) beside one whose receiver a call outside the declared set reaches
    (no meta evaluation runs that call): the dtype rule is decided for the batch-free read alone, which follows."""

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.randn(C))

    def forward(self, x):
        z = opaque_cast(self.a(x))
        return self.finish(z + self.p.new_full((4, 1), 0.1) + z.new_ones(C))


READS = {"zeros": Zeros, "ones": Ones, "full": Full}
COLLIDING = {"meta_first": CollideMetaFirst, "cone_first": CollideConeFirst}


def _free(factory) -> tuple:
    gm = trace_model(native(factory), SHAPE)
    gm = normalize_fx_graph(gm, program_purity(gm).foreign_modules)
    free = batch_free_nodes(gm)
    return gm, [node for node in free if node.op == "call_method" and str(node.target).startswith("new_")]


@pytest.mark.parametrize("name", sorted(READS))
def test_a_factory_of_the_receivers_dtype_is_batch_free(name):
    _, found = _free(READS[name])
    assert len(found) == 1, found


@pytest.mark.parametrize("factory", [FromTheInput, OpaqueReceiver], ids=lambda f: f.__name__)
def test_a_factory_reading_more_is_not_batch_free(factory):
    _, found = _free(factory)
    assert found == []


@pytest.mark.parametrize("name", sorted(READS))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_metadata_read_converts_exactly(name, contract):
    assert_converted_exact(native(READS[name]), SHAPE, contract, batches=(1, 2, 3, 9))


@pytest.mark.parametrize("name", sorted(READS))
def test_the_value_program_of_a_metadata_read_is_exact(name):
    assert_value_program_exact(native(READS[name]), SHAPE, batches=(2, 3))
    flow = quietly(convert_torch_model, native(READS[name]), SHAPE, 4, packaging=MVM_PACKAGING,
                   value_program_stage=True)
    assert flow.conversion_certificate.certified


def test_the_stand_in_follows_the_converted_models_cast():
    """Converted in float32, cast to float64: the factory's value is float64, as the native model's is."""
    torch.manual_seed(0)
    model = Full().eval()
    flow = convert(model, SHAPE, "mvm").double()
    x = inputs(SHAPE, 3, "normal")
    with torch.no_grad():
        assert torch.allclose(flow(x), model.double()(x), atol=1e-12, rtol=0)
    gm, found = _free(Full)
    values = BatchFreeValues(gm)
    assert torch.equal(values.evaluate(found[0]), torch.full((4, 1), 0.1, dtype=torch.float64))
    stand_ins = {name.rsplit(".", 1)[-1]: (b.dtype, b.numel()) for name, b in values._state.named_buffers()
                 if "_receiver_" in name}
    assert stand_ins == {"_receiver_float64": (torch.float64, 0)}, stand_ins


@pytest.mark.parametrize("name", sorted(COLLIDING))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_state_value_named_as_a_stand_in_converts_exactly(name, contract):
    _, found = _free(COLLIDING[name])
    assert len(found) == 1, found
    assert_converted_exact(native(COLLIDING[name]), SHAPE, contract, batches=(1, 2, 3, 9, 20))


@pytest.mark.parametrize("name", sorted(COLLIDING))
def test_the_value_program_of_a_state_value_named_as_a_stand_in_is_exact(name):
    assert_value_program_exact(native(COLLIDING[name]), SHAPE, batches=(2, 3))


def test_the_stand_ins_live_apart_from_every_copied_value():
    """The stand-in's path is named apart from every attribute of the program; the parameter of its name is copied at
    its own path, unchanged (S1)."""
    gm, found = _free(CollideMetaFirst)
    values = BatchFreeValues(gm)
    values.evaluate(found[0])
    [read] = [node for node in batch_free_nodes(gm) if node.op == "call_method" and node.target == "sum"]
    assert torch.equal(values.evaluate(read), gm._receiver_float64.detach().sum())
    stand_ins = [name for name, b in values._state.named_buffers() if b.numel() == 0]
    assert len(stand_ins) == 1 and "." in stand_ins[0] and not hasattr(gm, stand_ins[0].split(".")[0]), stand_ins


def test_a_copied_path_holding_another_value_is_refused():
    """``_install`` keeps an attribute already at a path only when it is the copy of the program's value there."""
    gm, found = _free(CollideMetaFirst)
    values = BatchFreeValues(gm)
    values._state.register_buffer("_receiver_float64", torch.zeros(0, dtype=torch.float64))
    [read] = [node for node in batch_free_nodes(gm) if node.op == "call_method" and node.target == "sum"]
    with pytest.raises(RuntimeError, match="not the copy"):
        values.evaluate(read)


def test_a_factory_reading_the_input_keeps_the_base_path_and_stage_2_is_refused_by_name():
    assert_converted_exact(native(FromTheInput), SHAPE, "mvm", batches=(1, 2, 3))
    with pytest.raises(ToolchainRefusal):
        quietly(convert_torch_model, copy.deepcopy(native(FromTheInput)), SHAPE, 4, packaging=MVM_PACKAGING,
                value_program_stage=True)



@pytest.mark.parametrize("built", ["float64", "float32"])
def test_a_receiver_whose_dtype_the_program_pins_keeps_it_under_a_cast_of_the_converted_model(built):
    """S2: the converted model cast to float64 equals the native model cast alike, at either converted dtype (the
    stand-in of a pinned receiver is no cast-following buffer)."""
    torch.manual_seed(0)
    model = PinnedFactory().eval() if built == "float32" else native(PinnedFactory)
    flow = convert(model, SHAPE, "mvm").double()
    x = inputs(SHAPE, 3, "normal")
    with torch.no_grad():
        assert torch.allclose(flow(x), model.double()(x), atol=1e-12, rtol=0)


def test_a_pinned_stand_in_keeps_its_dtype_and_a_following_one_follows_the_states_cast():
    for factory, follows in ((PinnedFactory, False), (Full, True)):
        gm, found = _free(factory)
        values = BatchFreeValues(gm)
        values.evaluate(found[0])
        for dtype in (torch.float32, torch.float64, torch.float16):
            values._program(found[0]).to(dtype)  # the value's program holds the state it reads (as a flow's cast does)
            got = values.evaluate(found[0]).dtype
            assert got == (dtype if follows else torch.float32), (factory.__name__, dtype, got)


def test_a_receiver_whose_dtype_follows_some_casts_only_is_refused_by_name():
    with pytest.raises(ToolchainRefusal, match="metadata read"):
        convert(native(MixedPromotion), SHAPE, "mvm")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_pinned_receiver_converts_exactly(contract):
    assert_converted_exact(native(PinnedFactory), SHAPE, contract, batches=(1, 2, 3, 9))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_read_outside_the_batch_free_values_takes_no_part_in_the_dtype_rule(contract):
    assert_converted_exact(native(FreeBesideOpaque), SHAPE, contract, batches=(1, 2, 3))
