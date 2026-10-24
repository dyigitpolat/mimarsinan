"""Shape-specialized inlining (coverage W8): a body's writes to tensors it allocates itself, rewritten out of place.

On the base, a leaf call whose body writes in place stayed opaque (rule (d): ``its body writes in place``, or
``its body calls setitem``), so the program was impure and stage 2 refused it by name: torchvision's
``shifted_window_attention`` builds Swin's attention mask with ``x.new_zeros`` and ``setitem`` and zeroes a slice of a
clone of Swin V2's ``qkv_bias`` in place. Now a write to a tensor the body allocates (a factory, a clone, an operator
returning a new tensor), at a literal basic index (integers and slices with literal bounds), of a literal number, with
no other read of that tensor or a view of it before its last write, is the same value computed out of place:
``torch.slice_scatter`` and ``torch.select_scatter`` of ``torch.full_like`` of the indexed view, nested by axis. Any
other write keeps the call opaque with its reason: a write to an operand or to a value that may share its storage (a
``contiguous()`` of the operand), a view read after a write, an index or value computed from the batch, a value read
from the written tensor, an in-place arithmetic operator.
"""

from __future__ import annotations

import copy
import itertools

import pytest
import torch
import torch.fx
import torch.fx as fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.fx_rewrites.body_rules.local_writes import assignment_nodes
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_stage_2_refused,
    assert_value_program_exact,
    native,
    quietly,
)

C = 4
SHAPE = (4, 4, C)
BATCHES = (1, 2, 3, 5, 9)


class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(C, C), nn.Linear(16 * C, 3)
        self.b = nn.Parameter(torch.randn(C))


# ----------------------------------------------------------------------------- writes the rewrite takes
@torch.fx.wrap
def mask_rows(x):
    """Swin's mask: a fresh tensor of the input's dtype filled region by region with literal counts."""
    m = x.new_zeros((4, 4))
    count = 0
    for h in ((0, -2), (-2, -1), (-1, None)):
        for w in ((0, -2), (-2, -1), (-1, None)):
            m[h[0]:h[1], w[0]:w[1]] = count
            count += 1
    return x + m.unsqueeze(-1)


@torch.fx.wrap
def zero_middle(x, b):
    """Swin V2's ``qkv_bias``: a clone of a parameter with a slice zeroed in place."""
    b = b.clone()
    b[1:3].zero_()
    return x + b


@torch.fx.wrap
def fill_strided(x, b):
    b = b.clone()
    b[::2].fill_(0.5)
    return x * b


@torch.fx.wrap
def write_local_activation(x):
    """Writes to a tensor computed from the input, on per-sample axes: the scattered value carries the batch."""
    y = x * 2.0
    y[:, 1:3, 0] = 0.0
    y[:, -1] = 7.0
    return y


@torch.fx.wrap
def step_and_negative(x):
    m = x.new_ones((4, 4))
    m[-3:, ::2] = 2.0
    m[1:-1:2] = 3.0
    m[:, -1] = -1.0
    m[2] = 5.0
    return x * m.unsqueeze(-1)


@torch.fx.wrap
def int_into_float(x):
    m = x.new_zeros((4, 4))
    m[0:2, 0:2] = 3
    m[2:, 2:] = True
    return x - m.unsqueeze(-1)


@torch.fx.wrap
def full_value(x):
    m = x.new_full((4, 4), 0.1)
    m[1:3, 1:3] = 0.25
    return x + m.unsqueeze(-1)


# ----------------------------------------------------------------------------- writes that keep the call opaque
@torch.fx.wrap
def write_operand(x):
    x[:, 0] = 0.0
    return x * 2.0


@torch.fx.wrap
def write_operand_view(x):
    v = x[:, :2]
    v.zero_()
    return x + 1.0


@torch.fx.wrap
def write_contiguous_alias(x):
    """``contiguous()`` of a contiguous operand is the operand: the write reaches it natively."""
    t = x.contiguous()
    t[:, 0] = 0.0
    return t * 2.0 + x


@torch.fx.wrap
def write_type_as_alias(x):
    """``type_as`` of a tensor of that type is the tensor itself (its schema declares no alias): the write reaches
    the operand natively."""
    t = x.type_as(x)
    t[:, 0] = 0.0
    return t * 2.0 + x


@torch.fx.wrap
def write_to_dense_alias(x):
    """``to_dense`` of a strided tensor is the tensor itself."""
    t = x.to_dense()
    t[:, :, 1:] = 3.0
    return t + x


@torch.fx.wrap
def view_read_after_write(x):
    m = x.new_zeros((4, 4))
    u = m[0:2]
    m[0] = 5.0
    return x + u.sum()


@torch.fx.wrap
def view_kept_across_write(x):
    m = x.new_zeros((4, 4))
    out = m.view(2, 8)
    m[1] = 1.0
    return x + out.reshape(4, 4).unsqueeze(-1)


@torch.fx.wrap
def tensor_value(x):
    m = x.new_ones((4, 4))
    m[0:2] = m[2:4] * 3.0
    return x + m.unsqueeze(-1)


@torch.fx.wrap
def in_place_arithmetic(x):
    m = x.new_ones((4, 4))
    m.add_(1.0)
    return x * m.unsqueeze(-1)


@torch.fx.wrap
def batch_index(x):
    """``m[:B] = 1`` holds min(B, 4) ones: an index computed from the batch."""
    m = x.new_zeros((4, 4))
    m[: x.shape[0]] = 1.0
    return x + m.unsqueeze(-1)


@torch.fx.wrap
def batch_value(x):
    """``m[0] = B``: a value computed from the batch."""
    m = x.new_zeros((4, 4))
    m[0] = x.shape[0]
    return x + m.unsqueeze(-1)


@torch.fx.wrap
def write_batch_first(x):
    """Sample 0 of a tensor the body computes from its input, overwritten with a literal: the native model treats
    sample 0 apart."""
    y = x * 2.0
    y[0] = 7.0
    return y


@torch.fx.wrap
def write_batch_last(x):
    y = x * 2.0
    y[-1] = 7.0
    return y


@torch.fx.wrap
def write_batch_tail(x):
    y = x * 2.0
    y[1:] = 0.0
    return y


@torch.fx.wrap
def write_batch_step(x):
    y = x * 2.0
    y[::2, 0] = 1.0
    return y


def _net(helper, *operands):
    namespace: dict = {}
    extra = "".join(f", self.{name}" for name in operands)
    exec(f"def forward(self, x):\n    return self.fc2(torch.flatten({helper.__name__}(self.fc1(x){extra}), 1))\n",
         globals(), namespace)
    return type(f"Net_{helper.__name__}", (_Fc,), {"forward": namespace["forward"]})


TAKEN = {helper.__name__: (helper, operands) for helper, operands in [
    (mask_rows, ()), (zero_middle, ("b",)), (fill_strided, ("b",)), (write_local_activation, ()),
    (step_and_negative, ()), (int_into_float, ()), (full_value, ())]}
KEPT = {helper.__name__: (helper, reason) for helper, reason in [
    (write_operand, "writes in place"), (write_operand_view, "writes in place"),
    (write_contiguous_alias, "writes in place"), (write_type_as_alias, "writes in place"),
    (write_to_dense_alias, "writes in place"), (view_read_after_write, "writes in place"),
    (view_kept_across_write, "writes in place"), (tensor_value, "writes in place"),
    (in_place_arithmetic, "writes in place"), (batch_index, ""), (batch_value, "writes in place")]}


def _traced(factory) -> fx.GraphModule:
    return trace_model(native(factory), SHAPE)


def _targets(gm: fx.GraphModule) -> set:
    return {node.target if node.op == "call_method" else getattr(node.target, "__name__", node.target)
            for node in gm.graph.nodes if node.op in ("call_function", "call_method")}


@pytest.mark.parametrize("name", sorted(TAKEN))
def test_a_write_to_a_tensor_the_body_allocates_is_inlined_out_of_place(name):
    helper, operands = TAKEN[name]
    gm = _traced(_net(helper, *operands))
    assert helper not in {node.target for node in gm.graph.nodes}, gm.graph
    assert program_purity(gm).pure, program_purity(gm).impurity
    targets = _targets(gm)
    assert not targets & {"setitem", "zero_", "fill_"}, targets
    assert targets & {"slice_scatter", "select_scatter"}, targets


@pytest.mark.parametrize("name", sorted(TAKEN))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_inlined_write_converts_exactly_at_every_batch(name, contract):
    helper, operands = TAKEN[name]
    flow = assert_converted_exact(native(_net(helper, *operands)), SHAPE, contract, batches=BATCHES)
    assert flow.shape_impurity is None


@pytest.mark.parametrize("name", sorted(TAKEN))
def test_the_value_program_of_an_inlined_write_is_exact(name):
    helper, operands = TAKEN[name]
    assert_value_program_exact(native(_net(helper, *operands)), SHAPE, batches=(2, 3, 9))


@pytest.mark.parametrize("name", sorted(KEPT))
def test_any_other_write_keeps_the_call_opaque_and_exact(name):
    helper, reason = KEPT[name]
    factory = _net(helper)
    gm = _traced(factory)
    calls = [node for node in gm.graph.nodes if node.target is helper]
    assert calls and reason in calls[0].meta[OPAQUE_BODY], calls[0].meta.get(OPAQUE_BODY)
    assert_converted_exact(native(factory), SHAPE, "mvm", batches=BATCHES)
    assert_impure_stage_2_refused(native(factory), SHAPE, calls[0].name, helper.__name__)


# ----------------------------------------------------------------------------- the out-of-place form itself
_BOUNDS = (None, 0, 1, -1, -3, 5, -9, 2)
_STEPS = (None, 1, 2, 3)
_INDICES = [slice(a, b, s) for a, b, s in itertools.product(_BOUNDS, _BOUNDS, _STEPS)] + [0, 3, -1, -5]


def _assign(t: torch.Tensor, index, value) -> torch.Tensor:
    """The out-of-place form the rewrite emits, as an FX graph run on ``t``."""
    graph = fx.Graph()
    source = graph.placeholder("t")
    graph.output(assignment_nodes(graph, source, index if isinstance(index, tuple) else (index,), value))
    return fx.GraphModule(nn.Module(), graph)(t)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64, torch.bool])
def test_the_out_of_place_form_equals_setitem_for_every_basic_index(dtype):
    torch.manual_seed(0)
    base = (torch.randn(5, 7, 6) * 4).to(dtype)
    checked = 0
    for first, second in itertools.product(_INDICES, [slice(None), slice(1, -1, 2), 2, -1]):
        for index in ((first,), (first, second), (slice(None), first), (second, slice(None), first)):
            for value in (3, 0.5, True, -2):
                native_value = base.clone()
                native_value[index] = value
                assert torch.equal(_assign(base, index, value), native_value), (index, value, dtype)
                checked += 1
    assert checked > 1000


def test_the_out_of_place_form_leaves_its_operand_unchanged():
    t = torch.arange(12.0).reshape(3, 4)
    before = t.clone()
    out = _assign(t, (slice(0, 2), 1), 9.0)
    assert torch.equal(t, before) and out[0, 1] == 9.0 and out.data_ptr() != t.data_ptr()


class ScatterRoot(_Fc):
    """Root code: ``torch.select_scatter`` of an activation and its slice, operands of ranks 3 and 2. At the trace's
    batch of 1 they broadcast to the output, so the host op was marked elementwise and its run-time broadcast check
    failed loudly from batch 2. [coverage W8: ``fx_shape_utils.broadcasts_to`` checks a batch of 2 too]"""

    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(torch.flatten(torch.select_scatter(y, y[:, 0] * 2.0, 1, 3), 1))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_host_op_of_operands_of_two_ranks_runs_at_every_batch(contract):
    assert_converted_exact(native(ScatterRoot), SHAPE, contract, batches=BATCHES)


# ----------------------------------------------------------------------------- a literal at a position of the batch
@pytest.mark.parametrize("helper", [write_batch_first, write_batch_last, write_batch_tail, write_batch_step],
                         ids=lambda h: h.__name__)
def test_a_literal_written_at_a_batch_position_is_exact_and_never_certified(helper):
    """The native model writes sample 0 (the last, every other one) apart from the rest. The converted model computes
    that at every batch, and stage 2 is refused by name: an inlined ``y[0] = 7`` leaves a view of one sample, which
    has no buffer form (``HostValueBatchLayoutError``); ``y[1:]`` and ``y[::2]`` decide on the batch (``B - 1 == 1``,
    ``(B + 1) // 2``) outside the domain, so the call stays opaque (``ImpureShapeProgramError``)."""
    factory = _net(helper)
    assert_converted_exact(native(factory), SHAPE, "mvm", batches=BATCHES)
    with pytest.raises(ToolchainRefusal, match="identity value program") as refused:
        quietly(convert_torch_model, copy.deepcopy(native(factory)), SHAPE, 10, packaging=MVM_PACKAGING,
                value_program_stage=True)
    assert "HostValueBatchLayoutError" in str(refused.value) or "ImpureShapeProgramError" in str(refused.value), \
        refused.value
