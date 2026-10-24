"""Leaf-call inlining: an opaque call of a Python function (an FX leaf function, ``torch.fx.wrap``) whose operands
other than tensors are literals is replaced by its body as Dynamo captures it, spliced into the program.

On the base every such call stayed opaque (``a call outside the declared operator set``): the program was impure, so
stage 2 refused it (``ImpureShapeProgramError``) whatever the helper computed; torchvision's ``stochastic_depth`` (an
identity at the eval-mode literal) made EfficientNet, ConvNeXt, Swin and MaxViT impure this way. A body Dynamo does not
capture as one graph (a branch on a tensor value), or whose capture decides on the batch (a branch on a shape), keeps
the call opaque, its reason extended; Dynamo interprets nested helpers itself; tensor constants of a body become
attributes of the program; a container literal (list, tuple, dict) is held constant and the body's output rebuilt by
Dynamo's residual code. A traced operand that is not a tensor at every trace batch (a shape, a value read from one), or
one computed from a call that stays opaque, keeps the call opaque. [restated in fix round 2 of the coverage fixes:
round 1 traced the body with FX on Proxies; ``batch_dependent_values`` and the dataflow precondition are gone, the
witnesses below that FX misread now inline and convert exactly]
"""

from __future__ import annotations

import copy
import math
import operator

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torch.overrides import handle_torch_function, has_torch_function
from torchvision.ops import StochasticDepth

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_free import batch_free_nodes
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.graph_normalization import normalize_fx_graph
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_stage_2_refused,
    assert_value_program_exact,
    native,
    quietly,
)

W = 6
SHAPE = (W,)


# ----------------------------------------------------------------------------- helpers (FX leaf functions)
@torch.fx.wrap
def drop_rows(x, p, mode, training):
    """A stochastic-depth helper (torchvision's form): the identity under the eval-mode literal."""
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ("batch", "row"):
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return x
    keep = torch.empty([x.shape[0]] + [1] * (x.ndim - 1), dtype=x.dtype, device=x.device).bernoulli_(1.0 - p)
    return x * keep / (1.0 - p)


@torch.fx.wrap
def flip_by_sum(x):
    """A branch on a tensor value: FX cannot trace the body."""
    return x if bool(x.sum() >= 0) else -x


def flip_by_sum_specialized(x):
    """``flip_by_sum`` with its branch fixed at one arm: what an inline under the tensor-value branch would be."""
    return x


@torch.fx.wrap
def halve_above_8(x):
    """A branch on the batch extent (a shape): FX cannot trace the body."""
    return x if x.shape[0] <= 8 else x * 0.5


@torch.fx.wrap
def scaled(x, factor):
    return x * factor


@torch.fx.wrap
def scaled_twice(x, factor, twice):
    """Two nested leaf calls under a literal branch."""
    y = scaled(x, factor)
    return scaled(y, factor) if twice else y


_OFFSET = torch.tensor([0.25, -0.5, 1.0, 2.0, -1.0, 0.75])
"""A tensor a helper closes over: FX records it as a constant of the body's root."""

_ROOT_OFFSET = torch.tensor([1.0, 0.5, -0.25, 0.0, 0.125, -2.0])
"""A tensor the root code closes over: a constant of the program's root (``_tensor_constant0`` in both traces)."""


@torch.fx.wrap
def shifted(x):
    return x + _OFFSET


_MODE = {"scaled": True}
"""A global flag a helper reads: evaluated at trace time, as root code reads it."""


@torch.fx.wrap
def flag_scaled(x):
    return x * 3.0 if _MODE["scaled"] else x


@torch.fx.wrap
def halves(x, at):
    """Two values: the call's users index it."""
    return x[:, :at], x[:, at:]


@torch.fx.wrap
def groups_for(n):
    """2 from batch 9 on (the round-6 reviewer's helper): ``isinstance`` of a size proxy is False while FX traces, so a
    trace of the body returns the constant 1 and never reads ``n``."""
    return 2 if isinstance(n, int) and n > 8 else 1


@torch.fx.wrap
def pad_from_9(t):
    """Pads two features from batch 9 on; its trace computes ``t.shape[0]`` and never uses it."""
    n = t.shape[0]
    return F.pad(t, (0, 2)) if isinstance(n, int) and n > 8 else t


@torch.fx.wrap
def doubled_row_sum(t):
    """Writes its operand in place."""
    t.mul_(2.0)
    return t.sum(1, keepdim=True)


@torch.fx.wrap
def inverse_root_numel(s):
    """A dataflow body on a static shape (review round 6's ``numel_scale``): its operand is a shape."""
    return s.numel() ** -0.5


@torch.fx.wrap
def tensor_scaled(t, u):
    """Reads its operand ``u`` only in Python: ``isinstance`` of a proxy is False while FX traces, True natively."""
    return t * (2.0 if isinstance(u, torch.Tensor) else 1.0)


@torch.fx.wrap
def sum_typed(t):
    """Computes a value it then reads only in Python."""
    s = t.sum()
    return t * (2.0 if isinstance(s, torch.Tensor) else 1.0)


@torch.fx.wrap
def pair_sum(pair):
    """Its operand is a container holding traced values: not a literal, so the call stays opaque."""
    return pair[0] + pair[1]


@torch.fx.wrap
def add_scaled_first(x, ks):
    """A list literal: FX traces one placeholder per entry and returns the body's output flattened to its leaves."""
    return x + ks[0] * ks[1]


@torch.fx.wrap
def shift_by(x, ab):
    """A tuple literal, a tensor returned."""
    return x * ab[0] + ab[1]


@torch.fx.wrap
def cfg_pair(x, cfg, *, bias=0.25):
    """A dict literal and a keyword-only literal; a dict returned."""
    return {"a": x * cfg["s"] + bias, "b": x - bias}


@torch.fx.wrap
def pair_of(x, ks):
    """A list literal; a tuple returned."""
    return x * ks[0], x + ks[1]


@torch.fx.wrap
def gate(t):
    """Doubles from batch 9 on, computing with the batch extent it reads (review round 1's counterexample): FX
    evaluates ``isinstance`` of the size proxy as False, so a trace of the body keeps the batch-8 arm and uses ``n``,
    a dataflow function of its operand."""
    n = t.shape[0]
    k = 2.0 if isinstance(n, int) and n > 8 else 1.0
    return t.reshape(n, -1) * k


@torch.fx.wrap
def regroup(t, n):
    """``gate`` with the extent an operand: the size the caller reads."""
    k = 2.0 if isinstance(n, int) and n > 8 else 1.0
    return t.reshape(n, -1) * k


@torch.fx.wrap
def batch_of(t):
    """Returns the batch extent: its body reads a shape, so its call stays opaque, a result of unknown kind."""
    return t.shape[0]


@torch.fx.wrap
def doubled_above_4(t, v):
    """Doubles once ``v`` exceeds 4 (``isinstance`` of a proxy is False while FX traces); ``v`` is used."""
    return t * (2.0 if isinstance(v, float) and v > 4.0 else 1.0) + v * 0.0


def selfrec(x):
    """Not wrapped: FX records it through ``__torch_function__``, a body whose trace is a call of itself."""
    if has_torch_function((x,)):
        return handle_torch_function(selfrec, (x,), x)
    return x * 2.0


# ----------------------------------------------------------------------------- programs
class _Fc(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4)


class DropNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(drop_rows(self.fc1(x), 0.25, "row", self.training)))


class DropKeywordNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(drop_rows(self.fc1(x), mode="row", training=self.training, p=0.25)))


class FlipNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(self.fc1(flip_by_sum(x))))


class HalveNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(halve_above_8(self.fc1(x))))


class NestedNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(scaled_twice(self.fc1(x), 0.5, True)))


class ShiftNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(shifted(self.fc1(x) + _ROOT_OFFSET)))


class FlagNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(flag_scaled(self.fc1(x))))


class RootFlagNet(_Fc):
    """``FlagNet`` with the helper's body written in the root code."""

    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(y * 3.0 if _MODE["scaled"] else y))


class HalvesNet(_Fc):
    def forward(self, x):
        a, b = halves(self.fc1(x), 2)
        return self.fc2(self.act(torch.cat([b, a], 1)))


class PairNet(_Fc):
    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(pair_sum([y, y * 2.0])))


class GroupsNet(_Fc):
    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(y * groups_for(x.size(0))))


class PadNet(_Fc):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(W + 2, 4)

    def forward(self, x):
        y = pad_from_9(self.act(self.fc1(x)))
        return self.fc2(F.pad(y, (0, W + 2 - y.shape[1])))


class WriteNet(_Fc):
    def forward(self, x):
        y = self.act(self.fc1(x))
        return self.fc2(y + doubled_row_sum(y.clone()))


class TypeReadNet(_Fc):
    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(tensor_scaled(y, y)))


class UnusedNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(sum_typed(self.fc1(x))))


class NumelNet(_Fc):
    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(y * inverse_root_numel(y.shape[1:])))


class ListNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(add_scaled_first(self.fc1(x), [0.5, 2.0])))


class TupleNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(shift_by(self.fc1(x), (2.0, -1.0))))


class DictNet(_Fc):
    def forward(self, x):
        d = cfg_pair(self.fc1(x), {"s": 1.5}, bias=-0.5)
        return self.fc2(self.act(d["a"] * d["b"]))


class PairOfNet(_Fc):
    def forward(self, x):
        a, b = pair_of(self.fc1(x), [2.0, -1.0])
        return self.fc2(self.act(a * b))


class GateNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(gate(self.fc1(x))))


class RegroupNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(regroup(self.fc1(x), x.size(0))))


class OpaqueSizeNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(regroup(self.fc1(x), batch_of(x))))


class MathNet(_Fc):
    """``math.sqrt`` of a traced value (FX records ``math`` calls): a Python value read from it, growing with the
    batch (natively 2.0 from batch 9 on the probe's inputs)."""

    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(doubled_above_4(y, math.sqrt(x.abs().sum()))))


class ResidualNet(_Fc):
    """A tensor computed with an opaque call's result is a tensor: the helper given it inlines, the helper given the
    result itself stays opaque."""

    def forward(self, x):
        y = self.fc1(x)
        z = y + flip_by_sum(y)
        return self.fc2(self.act(drop_rows(z, 0.25, "row", self.training) + scaled(flip_by_sum(z), 0.5)))


class SelfRecNet(_Fc):
    def forward(self, x):
        return self.fc2(self.act(selfrec(self.fc1(x))))


CONTAINER_LITERALS = {"list": (ListNet, add_scaled_first), "tuple": (TupleNet, shift_by),
                      "dict_and_keyword": (DictNet, cfg_pair), "list_tuple_output": (PairOfNet, pair_of)}
BATCH_DEPENDENT = {
    "body_reads_a_shape": (GateNet, gate, "decides on the batch"),
    "operand_is_a_size": (RegroupNet, regroup, "its operand 'n' is not a strided tensor (a value of type int)"),
    "operand_is_an_extent_returned_by_a_helper": (OpaqueSizeNet, regroup,
                                                  "its operand 'n' is not a strided tensor (a value of type int)"),
    "operand_is_a_math_value": (MathNet, doubled_above_4,
                                "its operand 'v' is not a strided tensor (a value of type float)"),
}
"""[restated in fix round 2: the reasons were FX's ("which FX would read in Python at its trace-time value");
``batch_of`` (it returns the batch extent) now inlines as the operand's ``size``, and ``regroup`` given it stays
opaque]"""


class StochasticDepthNet(_Fc):
    """torchvision's ``StochasticDepth`` module: its forward calls the leaf function ``stochastic_depth``."""

    def __init__(self):
        super().__init__()
        self.drop = StochasticDepth(0.2, "row")

    def forward(self, x):
        y = self.act(self.fc1(x))
        return self.fc2(y + self.drop(y))


def _traced(factory) -> fx.GraphModule:
    return trace_model(native(factory), SHAPE)


def _called(gm: fx.GraphModule) -> set:
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


def _x(batch: int = 3) -> torch.Tensor:
    return torch.randn(batch, W, generator=torch.Generator().manual_seed(7), dtype=torch.float64)


# ----------------------------------------------------------------------------- inlined
@pytest.mark.parametrize("factory", [DropNet, DropKeywordNet], ids=["positional", "keywords"])
def test_a_helper_with_an_eval_mode_literal_is_inlined_and_the_program_is_pure(factory):
    gm = _traced(factory)
    called = _called(gm)
    assert drop_rows not in called, gm.graph
    # FX's guards on the literals it specialized (``p == 0.25``, ``torch._assert``) fold away with the literals
    assert torch._assert not in called and operator.eq not in called, gm.graph
    assert program_purity(gm).pure
    with torch.no_grad():
        assert torch.equal(gm(_x()), native(factory)(_x()))


@pytest.mark.parametrize("factory", [DropNet, StochasticDepthNet, NestedNet, ShiftNet, HalvesNet],
                         ids=["drop_rows", "torchvision_stochastic_depth", "nested", "constants", "tuple"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_inlined_program_converts_exactly(factory, contract):
    flow = assert_converted_exact(native(factory), SHAPE, contract)
    assert flow.shape_impurity is None
    assert flow.conversion_certificate.certified
    proof = flow.conversion_certificate.shape_proof
    assert proof is None or proof.startswith("static shape proof"), proof


@pytest.mark.parametrize("factory", [DropNet, StochasticDepthNet, NestedNet, ShiftNet, HalvesNet],
                         ids=["drop_rows", "torchvision_stochastic_depth", "nested", "constants", "tuple"])
def test_the_value_program_of_an_inlined_program_is_exact(factory):
    assert_value_program_exact(native(factory), SHAPE)
    flow = quietly(convert_torch_model, native(factory), SHAPE, 4, packaging=MVM_PACKAGING, value_program_stage=True)
    certificate = flow.conversion_certificate
    assert certificate.certified and certificate.missing_stages == (), certificate
    assert [verdict.stage for verdict in certificate.stages] == ["converted model", "identity value program"]


def test_torchvisions_stochastic_depth_is_inlined():
    gm = _traced(StochasticDepthNet)
    names = {getattr(target, "__name__", str(target)) for target in _called(gm)}
    assert "stochastic_depth" not in names, gm.graph
    assert program_purity(gm).pure


def test_nested_helpers_are_inlined_to_a_fixpoint():
    gm = _traced(NestedNet)
    assert not {scaled, scaled_twice} & _called(gm), gm.graph
    assert program_purity(gm).pure
    with torch.no_grad():
        assert torch.equal(gm(_x()), native(NestedNet)(_x()))


def test_the_tensor_constants_of_a_body_become_attributes_of_the_program():
    gm = _traced(ShiftNet)
    assert shifted not in _called(gm), gm.graph
    constants = [getattr(gm, str(node.target)) for node in gm.graph.nodes if node.op == "get_attr"]
    held = [t for t in constants if t.shape == _OFFSET.shape]
    assert any(torch.equal(t.double(), _OFFSET.double()) for t in held) and any(
        torch.equal(t.double(), _ROOT_OFFSET.double()) for t in held), held  # the body's and the root's, both kept
    buffers = dict(gm.named_buffers())
    assert all(str(node.target) in buffers for node in gm.graph.nodes if node.op == "get_attr"), buffers
    with torch.no_grad():
        assert torch.equal(gm(_x()), native(ShiftNet)(_x()))


def test_the_witness_program_binding_holds_for_a_body_with_constants():
    """``convert_torch_model`` compares the witness program with the conversion's own traced program, node by node and
    in every tensor a node reads (``program_binding``): an inlined constant is one of those tensors."""
    flow = quietly(convert_torch_model, native(ShiftNet), SHAPE, 4, packaging=MVM_PACKAGING, value_program_stage=True)
    assert flow.conversion_certificate.certified and flow.conversion_certificate.reason is None


def test_a_global_flag_a_body_reads_is_evaluated_at_trace_time_as_root_code_is():
    gm, root = _traced(FlagNet), _traced(RootFlagNet)
    assert flag_scaled not in _called(gm)
    x = _x()
    try:
        _MODE["scaled"] = False  # after tracing: both programs keep the trace-time arm
        with torch.no_grad():
            assert torch.equal(gm(x), root(x))
            assert not torch.equal(gm(x), native(FlagNet)(x))
    finally:
        _MODE["scaled"] = True


def test_a_body_returning_a_tuple_is_spliced_into_the_users_of_the_call():
    gm = _traced(HalvesNet)
    assert halves not in _called(gm), gm.graph
    with torch.no_grad():
        assert torch.equal(gm(_x()), native(HalvesNet)(_x()))


@pytest.mark.parametrize("name", sorted(CONTAINER_LITERALS))
def test_a_container_literal_is_held_constant_and_the_call_inlined(name):
    """FX traces a container literal as one placeholder per leaf and returns the body's output flattened; the output
    is rebuilt with its tree, so every user reads the body's own value [review round 1 of the coverage fixes: the
    flattened list was spliced in, a tensor read as ``[tensor]``, ``TracingError`` at shape propagation]."""
    factory, helper = CONTAINER_LITERALS[name]
    gm = _traced(factory)
    assert helper not in _called(gm) and program_purity(gm).pure, gm.graph
    with torch.no_grad():
        assert torch.equal(gm(_x()), native(factory)(_x()))


@pytest.mark.parametrize("name", sorted(CONTAINER_LITERALS))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_container_literal_program_converts_exactly_at_every_batch(name, contract):
    flow = assert_converted_exact(native(CONTAINER_LITERALS[name][0]), SHAPE, contract, batches=(1, 2, 9, 13))
    assert flow.shape_impurity is None and flow.conversion_certificate.certified


@pytest.mark.parametrize("name", sorted(CONTAINER_LITERALS))
def test_the_value_program_of_a_container_literal_program_is_exact(name):
    factory = CONTAINER_LITERALS[name][0]
    assert_value_program_exact(native(factory), SHAPE)
    flow = quietly(convert_torch_model, native(factory), SHAPE, 4, packaging=MVM_PACKAGING, value_program_stage=True)
    assert flow.conversion_certificate.certified and flow.conversion_certificate.missing_stages == ()


@pytest.mark.timeout(240)
@pytest.mark.parametrize("builder,side", [("swin_t", 224), ("swin_v2_t", 256)])
def test_swin_traces_with_every_leaf_call_inlined_and_every_value_batch_leading(builder, side):
    """torchvision Swin calls ``_get_relative_position_bias`` with a list literal (its window size): inlined, it
    traces, never a ``TracingError`` [review round 1: the flattened output made shape propagation fail on every Swin
    and Swin V2]. [changed in coverage W6: the bias permute and Swin V2's coordinate MLP are batch-free values]
    [changed in coverage W7 and W8: Swin-T was refused at its window partition (``x_3``, the batch merged into the
    window axis) and Swin V2-T's attention calls stayed opaque (``zero_``; Swin-T's shifted blocks, ``setitem``). Now
    every leaf call is inlined (writes to tensors a body allocates computed out of place), the program is pure, and
    after normalization every value computed from the input carries the batch on axis 0 (the merged regions carried
    split). The refusal witnesses of a merged value an op mixes are in ``batch_merge/``; both stages of torchvision's
    Swin and Swin V2 at toy size, in ``batch_merge/test_swin_family.py``; the census sides, in the coverage
    reproducers]"""
    torch.manual_seed(0)
    model = getattr(tvm, builder)(weights=None).double().eval()
    gm = trace_model(copy.deepcopy(model), (3, side, side))
    names = {getattr(node.target, "__name__", "") for node in gm.graph.nodes if node.op == "call_function"}
    assert "_get_relative_position_bias" not in names and "shifted_window_attention" not in names, sorted(names)
    assert not [node.name for node in gm.graph.nodes if OPAQUE_BODY in node.meta]
    purity = program_purity(gm)
    assert purity.pure, purity.impurity
    gm = normalize_fx_graph(gm, purity.foreign_modules)
    free = batch_free_nodes(gm)  # the attention masks: metadata reads of an activation, batch-free
    reached, merged = set(), []
    for node in gm.graph.nodes:
        if node not in free and (node.op == "placeholder" or any(arg in reached for arg in node.all_input_nodes)):
            reached.add(node)
            shape = getattr(node.meta.get("tensor_meta"), "shape", None)
            if shape and shape[0] != 1:
                merged.append((node.name, tuple(shape)))
    assert not merged, merged
    assert any(node.op == "call_method" and node.target == "new_zeros" for node in free)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_shape_operand_keeps_the_call_opaque_and_exact_as_on_the_base(contract):
    """``Size.numel()`` of a shape operand: the call stays opaque (its operand is a shape), the host op of its own
    call, exact at every batch as on the base [review round 1: inlined, root code refused it by name]."""
    gm = _traced(NumelNet)
    assert inverse_root_numel in _called(gm)
    assert program_purity(gm).impurity.reason == (
        "a call outside the declared operator set; its operand 's' is not a strided tensor (a value of type Size)")
    assert_converted_exact(native(NumelNet), SHAPE, contract, batches=(1, 2, 9, 13))


# ----------------------------------------------------------------------------- kept opaque
BRANCH_REASONS = {"flip_by_sum": "Dynamo does not capture its body as one graph (UserError: ",
                  "halve_above_8": "its body decides on the batch: Dynamo's guard "}


@pytest.mark.parametrize("factory,helper", [(FlipNet, flip_by_sum), (HalveNet, halve_above_8)],
                         ids=["tensor_value_branch", "shape_branch"])
def test_a_body_branching_on_a_traced_value_stays_opaque_with_its_reason_extended(factory, helper):
    gm = _traced(factory)
    assert helper in _called(gm)
    impurity = program_purity(gm).impurity
    assert impurity is not None and impurity.operator == helper.__name__, impurity
    assert impurity.reason.startswith("a call outside the declared operator set; ") and (
        BRANCH_REASONS[helper.__name__] in impurity.reason), impurity.reason  # [restated in fix round 2: "its body
    # branches on a traced value", FX's]


@pytest.mark.parametrize("factory,node,helper", [(FlipNet, "flip_by_sum", flip_by_sum),
                                                 (HalveNet, "halve_above_8", halve_above_8)],
                         ids=["tensor_value_branch", "shape_branch"])
def test_stage_2_of_a_program_with_an_untraceable_body_is_refused_by_name(factory, node, helper):
    message = assert_impure_stage_2_refused(native(factory), SHAPE, node, helper.__name__)
    assert BRANCH_REASONS[helper.__name__] in message, message


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_untraceable_body_still_converts_exactly_at_stage_1(contract):
    flow = assert_converted_exact(native(FlipNet), SHAPE, contract)
    assert flow.shape_impurity is not None and flow.shape_impurity.operator == "flip_by_sum"


def test_a_call_given_a_container_of_traced_values_stays_opaque_and_names_the_operand():
    gm = _traced(PairNet)
    assert pair_sum in _called(gm)
    impurity = program_purity(gm).impurity
    assert impurity is not None and impurity.operator == "pair_sum", impurity
    assert impurity.reason == ("a call outside the declared operator set; its operand 'pair' holds a traced value "
                               "inside a container"), impurity.reason


def test_a_body_writing_in_place_stays_opaque():
    """A write in place stays the opaque call's observed write (T3)."""
    gm = _traced(WriteNet)
    assert doubled_row_sum in _called(gm), gm.graph
    impurity = program_purity(gm).impurity
    assert impurity is not None and impurity.operator == "doubled_row_sum", impurity
    assert impurity.reason == ("a call outside the declared operator set; its body writes in place (mul_: mul_)"), (
        impurity.reason)


@pytest.mark.parametrize("factory,helper", [(TypeReadNet, tensor_scaled), (UnusedNet, sum_typed)],
                         ids=["operand_read_in_python", "value_read_in_python"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_body_reading_a_tensor_in_python_is_inlined_and_exact(factory, helper, contract):
    """``isinstance`` of a tensor: Dynamo answers it as for a real tensor, so the inlined body takes the native arm
    [restated in fix round 2: kept opaque by round 1's dataflow precondition ("its body does not compute with its
    traced operand 'u'", "its body computes a value it never uses (sum_1)"), FX answering it False on a Proxy]."""
    gm = _traced(factory)
    assert helper not in _called(gm) and program_purity(gm).pure, gm.graph
    flow = assert_converted_exact(native(factory), SHAPE, contract, batches=(1, 2, 6, 9, 13))
    assert flow.conversion_certificate.certified


BATCH_DEPENDENT_ROUND_6 = {
    "round_6_size_operand": (GroupsNet, groups_for, "its operand 'n' is not a strided tensor (a value of type int)"),
    "round_6_body_reads_a_shape": (PadNet, pad_from_9, "decides on the batch"),
}


@pytest.mark.parametrize("name", sorted({**BATCH_DEPENDENT, **BATCH_DEPENDENT_ROUND_6}))
def test_a_batch_dependent_python_value_keeps_the_call_opaque(name):
    """A shape of a traced value, a value computed from one, or an opaque call's result may read differently in Python
    at another batch; FX keeps the trace-time read, so the call stays opaque, named [review round 1: ``gate`` passed
    the dataflow precondition, both stages were certified and the conversion wrong by 0.5 from batch 9]."""
    factory, helper, reason = {**BATCH_DEPENDENT, **BATCH_DEPENDENT_ROUND_6}[name]
    gm = _traced(factory)
    calls = [node for node in gm.graph.nodes if node.op == "call_function" and node.target is helper]
    assert len(calls) == 1, gm.graph
    assert reason in calls[0].meta[OPAQUE_BODY], calls[0].meta[OPAQUE_BODY]
    assert program_purity(gm).impurity is not None


@pytest.mark.parametrize("name", sorted(BATCH_DEPENDENT))
def test_stage_2_of_a_batch_dependent_helper_is_refused_by_name(name):
    factory, helper, reason = BATCH_DEPENDENT[name]
    first = program_purity(_traced(factory)).impurity.operator
    assert first == helper.__name__, first  # [restated in fix round 2: ``batch_of`` was first, opaque]
    message = assert_impure_stage_2_refused(native(factory), SHAPE, first, first)
    assert reason in message, message


@pytest.mark.parametrize("factory", [TypeReadNet, UnusedNet, WriteNet, GroupsNet, PadNet, GateNet, RegroupNet,
                                     OpaqueSizeNet, MathNet, ResidualNet],
                         ids=["unread_operand", "unused_value", "write_in_place", "round_6_size_operand",
                              "round_6_body_reads_a_shape", "body_reads_a_shape", "operand_is_a_size",
                              "operand_is_an_opaque_result", "operand_is_a_math_value", "residual_of_an_opaque_result"])
@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_body_kept_opaque_converts_exactly_at_every_batch(factory, contract):
    assert_converted_exact(native(factory), SHAPE, contract, batches=(1, 2, 6, 9, 13))


def test_a_value_computed_from_an_opaque_result_keeps_the_call_opaque():
    """``y + flip_by_sum(y)`` and ``flip_by_sum(z)`` are computed from a call that stays opaque: their extents are not
    in the batch domain, so a helper given either stays opaque, named [restated in fix round 2: round 1 inlined the
    helper given the sum, a tensor]."""
    gm = _traced(ResidualNet)
    called = _called(gm)
    assert drop_rows in called and scaled in called and flip_by_sum in called, gm.graph
    kept = {node.target.__name__: node.meta[OPAQUE_BODY] for node in gm.graph.nodes
            if node.op == "call_function" and node.target in (drop_rows, scaled)}
    assert kept == {"drop_rows": "its operand 'x' is computed from a call that stays opaque (node 'flip_by_sum')",
                    "scaled": "its operand 'x' is computed from a call that stays opaque (node 'flip_by_sum_1')"}, kept


def test_a_function_dispatching_to_its_own_call_takes_its_tensor_path_and_is_inlined():
    """FX records ``selfrec`` through ``__torch_function__``; on plain tensors its body takes its tensor path, which
    Dynamo captures [review round 1: inlined 1000 times to the interpreter's recursion limit, the node renamed
    ``selfrec_1000``; then opaque, "its body calls itself"; restated in fix round 2]."""
    gm = _traced(SelfRecNet)
    assert selfrec not in _called(gm) and program_purity(gm).pure, gm.graph
    flow = assert_converted_exact(native(SelfRecNet), SHAPE, "mvm", batches=(1, 2, 6, 9, 13))
    assert flow.conversion_certificate.certified


# ----------------------------------------------------------------------------- mutation
def test_an_inline_under_a_tensor_value_branch_is_refused(monkeypatch):
    """Mutation: an inliner that fixes the tensor-value branch of ``flip_by_sum`` at one arm (what tracing the body at
    one example would record) makes the program pure; the probe's negated inputs take the other arm natively, so the
    conversion is refused, never certified."""
    from mimarsinan.torch_mapping.fx_rewrites import leaf_call_inlining

    trace_body = leaf_call_inlining.trace_body

    def specialized(function, arguments, samples):
        body = flip_by_sum_specialized if function is flip_by_sum else function
        return trace_body(body, arguments, samples)

    monkeypatch.setattr(leaf_call_inlining, "trace_body", specialized)
    assert program_purity(_traced(FlipNet)).pure  # the mutation took effect
    with pytest.raises(ToolchainRefusal, match="value parity"):
        quietly(convert_torch_model, copy.deepcopy(native(FlipNet)), SHAPE, 4, packaging=MVM_PACKAGING,
                value_program_stage=True)
