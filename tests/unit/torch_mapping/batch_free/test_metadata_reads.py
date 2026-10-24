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


READS = {"zeros": Zeros, "ones": Ones, "full": Full}


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
    stand_ins = {name: (b.dtype, b.numel()) for name, b in values._state.named_buffers() if "_receiver_" in name}
    assert stand_ins == {"_receiver_float64": (torch.float64, 0)}, stand_ins


def test_a_factory_reading_the_input_keeps_the_base_path_and_stage_2_is_refused_by_name():
    assert_converted_exact(native(FromTheInput), SHAPE, "mvm", batches=(1, 2, 3))
    with pytest.raises(ToolchainRefusal):
        quietly(convert_torch_model, copy.deepcopy(native(FromTheInput)), SHAPE, 4, packaging=MVM_PACKAGING,
                value_program_stage=True)
