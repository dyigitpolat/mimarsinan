"""Batch-free values (coverage W6): a value computed from the program's state and literals only has no batch axis.

On the base such a value had no conversion rule: a permute of the gathered relative-position bias (Swin, MaxViT) was
refused as a layout of a value that is not an activation (``BatchAxisLayoutError``), and a Linear applied to a stored
coordinate table (Swin V2) as a module call without an activation (``ModuleCallArgumentError``). Now the value is
computed, at every call of the converted model, by the traced operators that compute it natively, from the converted
model's own copy of the state it reads, as a bound operand of every operator that reads it: gradients reach that state
(the relative-position table is trained as natively), and the probe compares the computation itself with the native
model's (no fold for both). At IR lowering the deployed program holds it as a constant. A value whose computation
draws a random number, writes in place, is ordered around a write, or calls outside the declared operator set keeps
the base's path, as does a parameter read as it is.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from batch_free.models import (
    SHAPE,
    T,
    AttentionMask,
    ClassToken,
    ConstantOutput,
    CoordinateBias,
    OpaqueCone,
    OpaqueConsumer,
    PlainParameter,
    RandomCone,
    RandomMask,
    RelativeBias,
    StateShape,
    StateViewToLaterOpaqueCall,
    StateViewToOpaqueCall,
    TwoConsumers,
    WrappedBias,
    WrittenByOpaqueCall,
    WrittenState,
)
from operand_completeness.harness import (
    CONTRACTS,
    EXACT,
    assert_converted_exact,
    assert_impure_stage_2_refused,
    assert_value_program_exact,
    convert,
    inputs,
    native,
    quietly,
    worst_error,
)

PURE = [RelativeBias, WrappedBias, CoordinateBias, TwoConsumers, ClassToken]
MASK_SHAPE = (T, 4)


def _batch_free_value_type():
    from mimarsinan.mapping.support.batch_free_value import BatchFreeValue

    return BatchFreeValue


def _named_state(flow: nn.Module, suffix: str) -> nn.Parameter:
    """The converted model's own copy of a state tensor (a parameter whose name ends with ``suffix``), exactly one."""
    found = {id(p): p for name, p in flow.named_parameters() if name.endswith(suffix)}
    assert len(found) == 1, sorted(name for name, _ in flow.named_parameters())
    return next(iter(found.values()))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("factory", PURE, ids=lambda f: f.__name__)
def test_batch_free_value_converts_exact(factory, contract):
    """The bias permute (Swin, MaxViT), the leaf helper, the coordinate MLP (Swin V2), a value read twice and a
    computed class token convert, exact against the native fp64 model at batches 1, 2, 3 and 5."""
    assert_converted_exact(native(factory), SHAPE, contract)


@pytest.mark.parametrize("factory", PURE, ids=lambda f: f.__name__)
def test_batch_free_value_program_exact(factory):
    """Stage 2: the identity value program of the converted model, exact (the value a constant of the program)."""
    assert_value_program_exact(native(factory), SHAPE)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_attention_mask_of_a_module_call(contract):
    """A batch-free float mask given to ``nn.MultiheadAttention``: the module call's bound operand."""
    assert_converted_exact(native(AttentionMask), MASK_SHAPE, contract)


def test_bias_read_by_an_opaque_call_impure_program():
    """Swin's form: the bias read by a call that stays opaque. Stage 1 exact; stage 2 refused by name (impure)."""
    model = native(OpaqueConsumer)
    assert_converted_exact(model, SHAPE, "mvm")
    assert_impure_stage_2_refused(model, SHAPE, "_opaque_attention", "_opaque_attention")


@pytest.mark.parametrize("factory, table", [(RelativeBias, "table"), (WrappedBias, "table"),
                                            (OpaqueConsumer, "table"), (TwoConsumers, "table"),
                                            (CoordinateBias, "cpb.0.weight"), (ClassToken, "cls")],
                         ids=lambda v: v if isinstance(v, str) else v.__name__)
def test_gradients_reach_the_state(factory, table):
    """The converted model's copy of the table is a trainable parameter, and its gradient equals the native one."""
    model = native(factory)
    flow = convert(model, SHAPE, "mvm")
    x = inputs(SHAPE, 3, "normal")
    model(x).square().sum().backward()
    flow(x).square().sum().backward()
    ours = _named_state(flow, table)
    want = dict(model.named_parameters())[table].grad
    assert ours.requires_grad and ours.grad is not None
    assert torch.allclose(ours.grad, want, rtol=1e-12, atol=1e-12), (ours.grad - want).abs().max()


@pytest.mark.parametrize("factory, table", [(RelativeBias, "table"), (CoordinateBias, "cpb.2.weight")],
                         ids=lambda v: v if isinstance(v, str) else v.__name__)
def test_value_is_computed_at_every_call(factory, table):
    """Not a fold: the same change of the native table and of the converted model's copy keeps the two equal."""
    model = native(factory)
    flow = convert(model, SHAPE, "mvm")
    x = inputs(SHAPE, 2, "normal")
    with torch.no_grad():
        before = flow(x).clone()
        delta = torch.linspace(-1.0, 1.0, dict(model.named_parameters())[table].numel(), dtype=torch.float64)
        dict(model.named_parameters())[table].add_(delta.view_as(dict(model.named_parameters())[table]))
        _named_state(flow, table).add_(delta.view_as(_named_state(flow, table)))
        after = flow(x)
        assert (after - before).abs().max() > 1e-6
        assert torch.allclose(after, model(x), rtol=1e-12, atol=1e-12)


def test_converted_model_owns_its_state():
    """The converted model holds its own copy (as a perceptron holds its weights): a later change of the source
    model's table does not reach it."""
    model = native(RelativeBias)
    flow = convert(model, SHAPE, "mvm")
    x = inputs(SHAPE, 2, "normal")
    with torch.no_grad():
        before = flow(x).clone()
        model.table.add_(1.0)
        assert torch.equal(flow(x), before)


def test_value_is_a_constant_of_the_deployed_program():
    """IR lowering: the host op that reads the bias holds it as a constant (no batch-free computation in the IR), equal
    to the native bias; the converted model keeps computing it."""
    batch_free_value = _batch_free_value_type()
    model = native(RelativeBias)
    flow = convert(model, SHAPE, "mvm")
    mapper_repr = flow.get_mapper_repr()
    try:
        ir = map_model_to_ir(mapper_repr)
    finally:
        mapper_repr.clear_ir_caches()
    payloads = [op.params.get("module") for op in ir.get_compute_ops()]
    assert not any(isinstance(m, batch_free_value) for p in payloads if p is not None for m in p.modules())
    from batch_free.models import relative_bias

    want = relative_bias(model.table, model.index).detach()[0]
    held = [t for p in payloads if p is not None for t in p.parameters() if tuple(t.shape) == tuple(want.shape)]
    assert any(torch.equal(t, want) and not t.requires_grad for t in held)
    assert any(isinstance(m, batch_free_value) for m in flow.modules())


def test_probe_checks_the_computation_itself(monkeypatch):
    """The probe compares the converted model's own computation of the bias with the native one: a wrong batch-free
    computation (here the bias transposed) is refused by the value-parity probe, never certified."""
    batch_free_value = _batch_free_value_type()
    forward = batch_free_value.forward
    monkeypatch.setattr(batch_free_value, "forward", lambda self: forward(self).transpose(-1, -2))
    with pytest.raises(ToolchainRefusal, match="relative error"):
        quietly(convert_torch_model, native(RelativeBias), SHAPE, 10, packaging=MVM_PACKAGING)


def test_batch_free_output_refused_by_name():
    """An output computed from the state alone has no batch axis: refused by name (the base failed loudly)."""
    with pytest.raises(ToolchainRefusal, match="output"):
        quietly(convert_torch_model, native(ConstantOutput), (3,), 10, packaging=MVM_PACKAGING)


def test_parameter_read_as_it_is_keeps_the_frozen_bound_tensor():
    """A parameter operand keeps the base's path: a frozen bound tensor of its operator, no batch-free value."""
    flow = assert_converted_exact(native(PlainParameter), SHAPE, "mvm")
    frozen = [p for p in flow.parameters() if tuple(p.shape) == SHAPE]
    assert frozen and not any(p.requires_grad for p in frozen)
    assert not any(type(m).__name__ == "BatchFreeValue" for m in flow.modules())


def test_shape_of_a_batch_free_value_is_a_literal():
    """A shape read from a batch-free value is a static shape value, folded at conversion as on the base."""
    flow = assert_converted_exact(native(StateShape), SHAPE, "mvm")
    assert any(value == repr(T) for _, value in flow.shape_literals.values()), flow.shape_literals


@pytest.mark.parametrize("factory", [RandomCone, WrittenState, OpaqueCone, WrittenByOpaqueCall,
                                     StateViewToOpaqueCall, StateViewToLaterOpaqueCall], ids=lambda f: f.__name__)
def test_excluded_computations_keep_the_base_path(factory):
    """A random draw, a write in place of the state read, an opaque call in the computation, or a value an opaque call
    writes in place that a later operator reads (``read_after``; computing it again for that reader would read it
    unwritten): no batch-free value; the conversion is the base's (a refusal by name, or an exact host op of an impure
    program), never a value claimed for every call."""
    model = native(factory)
    try:
        flow = quietly(convert_torch_model, copy.deepcopy(model), SHAPE, 10, packaging=MVM_PACKAGING)
    except ToolchainRefusal:
        return
    assert not any(type(m).__name__ == "BatchFreeValue" for m in flow.modules())
    assert worst_error(flow.eval(), model, SHAPE, batches=(1, 2), kinds=("normal",)) <= EXACT


def test_flow_with_batch_free_values_pickles_and_copies():
    """The pipeline cache pickles a converted flow (``torch.save``): a flow holding batch-free values round-trips, and
    so does a deep copy, each computing the same output and keeping one copy of a table read twice."""
    import io as _io

    model = native(TwoConsumers)
    flow = convert(model, SHAPE, "mvm")
    x = inputs(SHAPE, 2, "normal")
    buffer = _io.BytesIO()
    torch.save(flow, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False).eval()
    copied = copy.deepcopy(flow).eval()
    with torch.no_grad():
        want = flow(x)
        assert torch.equal(loaded(x), want) and torch.equal(copied(x), want)
    for candidate in (loaded, copied):
        _named_state(candidate, "table")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_layout_of_an_excluded_state_value_is_refused_by_name(contract):
    """A view of a value of the state the batch-free rule excludes (a random draw) has no per-sample mapper: its
    reader refuses it by name (on the base the view became a ``ReshapeMapper`` with no source, and the converted
    forward failed loudly)."""
    with pytest.raises(ToolchainRefusal, match="view"):
        convert(native(RandomMask), MASK_SHAPE, contract)
