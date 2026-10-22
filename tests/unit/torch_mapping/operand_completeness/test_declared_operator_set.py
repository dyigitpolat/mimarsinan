"""The declared operator set: only declared operators run on symbolic values or fold (review round 7).

The symbolic-batch pass runs only the declared operator set (``symbolic/declared_operators.py``: torch and aten tensor
functions and methods, the scalar operators of ``operator``, ``math`` and ``torch.sym_*``, and the ``torch.nn`` leaf
modules with a declared shape rule, by exact type). Any other call (a ``torch.fx.wrap`` helper, a user function,
method or module, a module with hooks) is opaque: never run in the pass or at conversion time, no shape of its result
and of anything computed from it is proved, and it runs at run time as a host op, as on w0-base. Before, the pass ran
every helper on symbolic values: a helper that checks the Python type of a size took another branch on a ``SymInt``
than on any real batch, installed no guard, and its output shape counted as proved (PadWidth certified and wrong by
0.25 from batch 9).

Every node runs with guards suppressed and records the guards it requests on the batch symbol: nothing narrows the
range of B, so a request belongs to the node that made it and none hides another. A request anywhere leaves the whole
program without a static shape proof (nothing is folded; the certificate records ``no static shape proof: batch guard
... at ...``). A batch decision is a declared call that requests a guard, or fails, while reading a symbolic size (B
included), or an opaque call returning a non-tensor from a run-time value; it reaches every layout computed from it
through the data flow and the layout is refused by name. Before, a helper's guard on the bare batch size was not a
decision and proofs ended at the first guard in traced order, so six layouts decided by ``pick(x.size(0))`` were
certified and wrong at batch 13, depending on the order of two independent lines; and a guard implied by an earlier
one was never installed, so the same gate written twice (``GateTwice``) was certified and wrong at batch 13 too.

[restated in review round 9: a program with a call outside the declared set is impure (``symbolic/purity.py``): the
symbolic pass never runs for it, nothing is folded, and every layout is the host op of its native method, so the
helper-decided layouts of rounds 6 and 7, refused by name here before, are exact at every batch, and their value
programs are refused by name (``ImpureShapeProgramError``, naming the helper). A receiver's type is the run-time type
the tracer recorded, by exact type, and a module's reason names the attribute it holds.]

[restated in review round 14: the pass computes every batch-dependent value in the affine batch domain, never with
torch's guards; a value outside it is opaque, computed on the host at run time with every layout it reaches, and the
statement names the first one (``static shape proof: every batch-dependent size in the affine batch domain ...;
opaque batch values computed on the host at run time (first: ...)``); no guard is recorded.]
"""

from __future__ import annotations

import math
import operator
import time
from operator import is_

import numpy as np
import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.models.nn.layers import ChannelsLastBatchNorm1d
from mimarsinan.torch_mapping.converter import convert_torch_model, trace_and_convert
from mimarsinan.torch_mapping.converter_handlers import shape_value_mixin
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BATCH_SIZE, BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import (
    CONVERSION_TIME_OPERATORS,
    declare_leaf_module,
    is_conversion_time_operator,
    opaque_reason,
)
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity.certificate import CERTIFICATE_HYPOTHESES

from .harness import (
    CONTRACTS,
    PACKAGINGS,
    assert_converted_exact,
    assert_impure_exact,
    assert_impure_stage_2_refused,
    convert,
    native,
    quietly,
)

EVERY_BATCH = (1, 2, 3, 6, 7, 9, 13)
torch.fx.wrap("is_")


# ----------------------------------------------------------------------------- the set
def _call_node(target, *args, op: str = "call_function", **kwargs) -> fx.Node:
    graph = fx.Graph()
    x = graph.placeholder("x")
    return graph.create_node(op, target, (x, *args), kwargs)


@torch.fx.wrap
def _helper(n):
    return n


class _UserLSTM(nn.LSTM):
    """A user subclass of a declared leaf: its forward is the user's, so it is not declared."""


class _UserLinear(nn.Linear):
    pass


class _OwnLeaf(nn.Module):
    def forward(self, x):
        return x * 2


class _SubTensor(torch.Tensor):
    """A tensor subclass: its methods dispatch through its own ``__torch_function__``."""


class TestTheDeclaredSet:
    @pytest.mark.parametrize("fn", [
        torch.add, torch.cat, torch.reshape, torch.flatten, torch.unflatten, F.softmax, F.linear, F.pad,
        torch.Tensor.view, torch.ops.aten.add, torch.ops.aten.add.Tensor, torch.sym_max, torch._assert,
        operator.add, operator.getitem, operator.floordiv, math.floor, math.prod, getattr,
    ], ids=lambda fn: getattr(fn, "__name__", str(fn)))
    def test_torch_aten_operator_and_math_callables_are_declared(self, fn):
        receiver = torch.Tensor if fn is getattr else None  # round 9: the receiver's run-time type
        args = ("shape",) if fn is getattr else ()
        assert opaque_reason(_call_node(fn, *args), receiver, {}) is None

    @pytest.mark.parametrize("fn", [_helper, lambda n: n, np.prod, len, isinstance, str, repr, print],
                             ids=["fx_wrap_helper", "lambda", "numpy", "len", "isinstance", "str", "repr", "print"])
    def test_every_other_callable_is_opaque(self, fn):
        assert opaque_reason(_call_node(fn), None, {}) == "a call outside the declared operator set"

    @pytest.mark.parametrize("fn", [operator.is_, operator.is_not], ids=["is_", "is_not"])
    def test_identity_tests_are_opaque(self, fn):
        """An object's identity is not a function of its value (``n is m`` for two equal sizes holds up to 256)."""
        assert opaque_reason(_call_node(fn), None, {}) == "a call outside the declared operator set"
        assert fn not in CONVERSION_TIME_OPERATORS

    def test_a_declared_function_given_a_user_callable_is_opaque(self):
        node = _call_node(torch.vmap, lambda t: t)
        assert opaque_reason(node, None, {}) == "a declared call given a callable outside the declared operator set"

    def test_methods_and_attributes_of_tensors_and_shapes_are_declared(self):
        """Round 9: on a receiver of the exact run-time type ``torch.Tensor``, ``nn.Parameter`` or ``torch.Size``."""
        assert opaque_reason(_call_node("view", 6, op="call_method"), torch.Tensor, {}) is None
        assert opaque_reason(_call_node("view", 6, op="call_method"), nn.Parameter, {}) is None
        assert opaque_reason(_call_node("numel", op="call_method"), torch.Size, {}) is None
        assert opaque_reason(_call_node(getattr, "T"), torch.Tensor, {}) is None
        assert opaque_reason(_call_node("index", 1, op="call_method"), tuple, {}) == (
            "a method of a value that is not a tensor or a shape")
        assert opaque_reason(_call_node(getattr, "shape"), tuple, {}) == (
            "an attribute of a value that is not a tensor or a shape")
        assert opaque_reason(_call_node("view", 6, op="call_method"), _SubTensor, {}) == (
            "a method of a value that is not a tensor or a shape")

    @pytest.mark.parametrize("make", [
        lambda: nn.Linear(2, 2), lambda: nn.LSTM(2, 2), lambda: nn.MultiheadAttention(2, 1),
        lambda: nn.Conv2d(1, 1, 1), lambda: ChannelsLastBatchNorm1d(2),
    ], ids=["Linear", "LSTM", "MultiheadAttention", "Conv2d", "ChannelsLastBatchNorm1d"])
    def test_torch_nn_leaves_and_the_tracers_leaves_are_declared(self, make):
        assert opaque_reason(_call_node("m", op="call_module"), None, {"m": make()}) is None

    def test_a_module_is_declared_by_exact_type_and_without_hooks(self):
        hooked, prehooked, parametrized = nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)
        hooked.register_forward_hook(lambda module, args, out: out)
        prehooked.register_forward_pre_hook(lambda module, args: args)
        parametrize.register_parametrization(parametrized, "weight", nn.Identity())
        user_activation = nn.TransformerEncoderLayer(4, 1, 8, activation=lambda t: t * (t.shape[0] > 8))
        user_submodule = nn.TransformerEncoderLayer(4, 1, 8)
        user_submodule.linear1 = _UserLinear(4, 8)
        no_rule = "a module class without a declared shape rule"
        reasons = {name: opaque_reason(_call_node("m", op="call_module"), None, {"m": module}) for name, module in (
            ("subclass", _UserLSTM(2, 2)), ("own", _OwnLeaf()), ("hooked", hooked), ("prehooked", prehooked),
            ("parametrized", parametrized), ("user_activation", user_activation), ("user_submodule", user_submodule),
            ("module_activation", nn.TransformerEncoderLayer(4, 1, 8, activation=nn.GELU())))}
        assert reasons == {"subclass": no_rule, "own": no_rule, "hooked": "a module with a forward hook",
                           "prehooked": "a module with a forward hook", "parametrized": no_rule,
                           "user_activation": "a module whose attribute 'activation' is a callable outside the "
                                              "declared operator set",
                           "user_submodule": no_rule, "module_activation": None}

    def test_a_leaf_module_class_is_declared_in_one_place(self):
        class Declared(nn.Module):
            def forward(self, x):
                return x + 1

        node = _call_node("m", op="call_module")
        assert opaque_reason(node, None, {"m": Declared()}) is not None
        assert declare_leaf_module(Declared) is Declared
        assert opaque_reason(node, None, {"m": Declared()}) is None

    def test_folding_uses_the_scalar_operators_of_the_same_set(self):
        assert shape_value_mixin.CONVERSION_TIME_OPERATORS is CONVERSION_TIME_OPERATORS
        assert is_conversion_time_operator(operator.add) and is_conversion_time_operator(torch.sym_max)
        assert not is_conversion_time_operator(torch.add) and not is_conversion_time_operator(_helper)


# ----------------------------------------------------------------------------- opaque calls never run symbolically
SEEN = []


@torch.fx.wrap
def spy(t):
    """Records the type of every value it is called with."""
    SEEN.append((type(t).__name__, type(t.shape[0]).__name__))
    return t * 2


class SpyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        z = spy(self.fc(x))
        return z * z.shape[1]


def test_an_opaque_call_never_runs_on_symbolic_values():
    """Traced, propagated, sampled and probed: always a real tensor of a real batch; never a FakeTensor or SymInt.
    Round 8: its value in the pass is the one the seeded traces show ((B, 6)), unproved, as is every value computed
    from it (round 7: no value). [restated in review round 9: the program is impure, so the pass never runs for it and
    no value of it has a layout, ``fc``'s included]"""
    SEEN.clear()
    model = native(SpyModel)
    flow = convert(model, (8,), "spiking")
    assert SEEN and set(SEEN) == {("Tensor", "int")}
    gm = trace_model(native(SpyModel), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    assert layouts.impurity.node == "spy" and proof.batch is None and proof.shapes == {}
    assert layouts.layout(nodes["fc"]) is None and layouts.layout(nodes["spy"]) is None
    assert flow.shape_literals == {}


# ----------------------------------------------------------------------------- the round-6 blocker witnesses
@torch.fx.wrap
def groups_for(n):
    """2 groups from batch 9 on; 1 when the size is not a plain int (the round-6 reviewer's helper)."""
    return 2 if isinstance(n, int) and n > 8 else 1


@torch.fx.wrap
def groups_by_digits(n):
    return 2 if len(str(n)) == 2 else 1


@torch.fx.wrap
def pad_big(t):
    """Pads two features from batch 9 on; reads the tensor's batch extent."""
    n = t.shape[0]
    return F.pad(t, (0, 2)) if isinstance(n, int) and n > 8 else t


class HelperOfARunTimeValue(nn.Module):
    """The round-6 reviewer's witnesses (``r6_mods.py``): helpers given the batch size or an activation."""

    FORMS = {
        "pad_width": lambda x, y: y * pad_big(y).shape[1],
        "group_width_scale": lambda x, y: y * torch.reshape(y, (x.size(0), groups_for(x.size(0)), -1)).shape[2],
        "group_width_unflatten": lambda x, y: y * torch.unflatten(y, 1, (groups_for(x.size(0)), -1)).shape[2],
        "group_scale_host": lambda x, y: _grouped_softmax(x, y, groups_for(x.size(0))),
        "group_scale_digits": lambda x, y: _grouped_softmax(x, y, groups_by_digits(x.size(0))),
        "group_softmax_unflatten": lambda x, y: _unflattened_softmax(y, groups_for(x.size(0))),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


def _grouped_softmax(x, y, g):
    z = torch.reshape(y, (x.size(0), g, -1))
    return torch.reshape(torch.softmax(z * z.shape[-1] ** -0.5, -1), (x.size(0), -1))


def _unflattened_softmax(y, g):
    z = torch.unflatten(y, 1, (g, -1))
    return torch.flatten(torch.softmax(z * z.shape[-1] ** -0.5, -1), 1)


# form -> the host ops of the converted model (w0-base's shape queries at run time), in execution order
HELPER_HOST_OPS = {
    "pad_width": ["pad_big", "getattr_1", "getitem", "mul"],
    "group_width_scale": ["size", "size_1", "groups_for", "reshape", "getattr_1", "getitem", "mul"],
    "group_width_unflatten": ["size", "groups_for", "unflatten", "getattr_1", "getitem", "mul"],
    "group_scale_host": ["size_1", "size", "groups_for", "reshape", "getattr_1", "getitem", "pow_1", "mul",
                         "softmax", "size_2", "reshape_1"],
    "group_scale_digits": ["size_1", "size", "groups_by_digits", "reshape", "getattr_1", "getitem", "pow_1", "mul",
                           "softmax", "size_2", "reshape_1"],
}
# form -> the first impure FX node (its operator is its name), which the stage-2 refusal names (round 9; before: a
# run-time shape value, or in round 8 the opaque call whose tensor result's per-sample shape is not proved)
HELPER_STAGE_2 = {
    "pad_width": "pad_big",
    "group_width_scale": "groups_for",
    "group_width_unflatten": "groups_for",
    "group_scale_host": "groups_for",
    "group_scale_digits": "groups_by_digits",
}


def _host_ops(flow) -> list:
    return [m.name for m in flow.get_mapper_repr().execution_order()
            if isinstance(m, ComputeOpMapper) and m.name != "fc"]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(HELPER_HOST_OPS))
def test_a_helper_of_a_run_time_value_is_a_host_op_and_exact_at_every_batch(how, contract):
    """Certified and wrong from batch 9 (pad_width rel 0.25, the group widths rel 1.0) or at batch 13 (the grouped
    softmaxes rel 0.11) before; pad_width was exact on w0-base, the group widths refused there by name."""
    flow = assert_converted_exact(native(lambda: HelperOfARunTimeValue(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert _host_ops(flow) == HELPER_HOST_OPS[how] and flow.shape_literals == {}


@pytest.mark.parametrize("how", sorted(HELPER_STAGE_2))
def test_a_helper_of_a_run_time_value_is_refused_by_name_at_stage_2(how):
    node = HELPER_STAGE_2[how]
    assert_impure_stage_2_refused(native(lambda: HelperOfARunTimeValue(how)), (8,), node, node)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_layout_of_a_helpers_scalar_of_the_batch_is_exact(contract):
    """group_softmax_unflatten: certified and wrong at batch 13 by rel 0.11 in round 6, refused by name in rounds 7 and
    8 (naming ``groups_for``). [restated in review round 9: the program is impure; its flatten is the host op of
    ``torch.flatten`` on the run-time value, exact at every batch]"""
    assert_impure_exact(native(lambda: HelperOfARunTimeValue("group_softmax_unflatten")), (8,), contract,
                        "groups_for", "groups_for")


# ----------------------------------------------------------------------------- the round-6 major witnesses
@torch.fx.wrap
def pick(n):
    """1 for n <= 7, else 2 (Python control flow on the bare batch size)."""
    return 1 if n <= 7 else 2


@torch.fx.wrap
def half_if_big(n):
    return 3 if n > 7 else 6


class HelperOfTheBatch(nn.Module):
    """A layout whose argument a helper of the bare batch size decides (``r6_mods.py``); ``fc`` before or after."""

    FORMS = {
        "pick_first_view": lambda m, x: _pick_view(m, x, pick, first=True),
        "pick_first_reshape": lambda m, x: _pick_reshape(m, x),
        "pick_after_view": lambda m, x: _pick_after(m, x),
        "pick_proved_view": lambda m, x: _pick_view(m, x, pick, first=False),
        "pick_permute": lambda m, x: _pick_view(m, x, half_if_big, first=True),
        "pick_first_nn_flatten": lambda m, x: m.fl(F.softmax(_split(m, x, half_if_big), -1)),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)
        self.fl = nn.Flatten(1)

    def forward(self, x):
        return self.FORMS[self.how](self, x)


def _pick_view(m, x, helper, first):
    if first:
        g = helper(x.size(0))
        y = m.fc(x)
    else:
        y = m.fc(x)
        g = helper(x.size(0))
    return F.softmax(y.view(x.size(0), g, -1), -1).flatten(1)


def _pick_reshape(m, x):
    g = pick(x.size(0))
    y = m.fc(x)
    return F.softmax(y.reshape(x.size(0), g, -1), -1).reshape(x.size(0), -1)


def _pick_after(m, x):
    y = m.fc(x)
    g = pick(x.size(0))
    w = y * 1.0
    return F.softmax(w.view(x.size(0), g, -1), -1).flatten(1)


def _split(m, x, helper):
    g = helper(x.size(0))
    y = m.fc(x)
    return y.view(x.size(0), g, -1)


class PickInConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 12)

    def forward(self, x):
        g = pick(x.size(0))
        y = self.fc(F.relu(self.c(x)).flatten(1))
        return F.softmax(y.view(x.size(0), g, -1), -1).flatten(1)


PICKED = {  # form -> (refused operator and FX node, the helper)
    "pick_first_view": ("view", "pick"), "pick_first_reshape": ("reshape", "pick"),
    "pick_after_view": ("view", "pick"), "pick_proved_view": ("view", "pick"),
    "pick_permute": ("view", "half_if_big"), "pick_first_nn_flatten": ("view", "half_if_big"),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(PICKED))
def test_a_layout_a_helper_of_the_batch_decides_is_exact_whatever_the_line_order(how, contract):
    """Five certified and wrong at batch 13 in round 6 (rel 0.41 to 1.0; on w0-base too), the order of two independent
    lines deciding; pick_proved_view refused by name in round 6 (by the guard ``pick`` installed when it ran); all six
    refused by name in rounds 7 and 8, naming the helper. [restated in review round 9: impure, every layout the host op
    of its native method: exact at every batch; the stage-2 refusal names the helper]"""
    _, helper = PICKED[how]
    model = native(lambda: HelperOfTheBatch(how))
    assert_impure_exact(model, (8,), contract, helper, helper)
    if contract == "mvm":
        assert_impure_stage_2_refused(model, (8,), helper, helper)


# PickFlattenStart (``torch.flatten(z.view(B, 6 // g, g), 1)``), refused by name in round 7 as a declared change, is
# w0-base's program since round 8: ``test_shape_decisions.py``.


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_layout_a_helper_of_the_batch_decides_in_a_conv_net_is_exact(contract):
    """Certified and wrong at batch 13 by rel 0.13 to 0.97 in round 6 (on w0-base too), refused by name in rounds 7 and
    8. [restated in review round 9: impure; exact at every batch]"""
    assert_impure_exact(native(PickInConvNet), (1, 4, 4), contract, "pick", "pick")


# ----------------------------------------------------------------------------- decisions no earlier guard hides
@torch.fx.wrap
def count_big(t):
    """A scalar of an activation's batch extent."""
    return 2 if t.shape[0] > 8 else 1


class Decided(nn.Module):
    """A gate decided again after an earlier guard made it statically known, a helper's scalar of an activation, or
    the identity of two run-time sizes (``torch.fx.wrap("is_")``)."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)
        self.rnn = nn.LSTM(6, 6, batch_first=True)

    def forward(self, x):
        y = self.fc(x)
        if self.how == "gate_twice":  # the second gate's B <= 7 is implied by the first one's
            s = y * (1 + (x.size(0) > 7))
            return s + F.softmax(y.view(x.size(0), 1 + (x.size(0) > 7), -1), -1).flatten(1)
        if self.how == "lstm_then_gate":  # the leaf's Eq(B, 7) replaced B by 7 for every later node
            h = self.rnn(y.view(x.size(0), 1, 6))[0].flatten(1)
            return h + F.softmax(y.view(x.size(0), 1 + (x.size(0) > 7), -1), -1).flatten(1)
        if self.how == "data_gate":  # a gate on a data-dependent count: its symbolic run raises
            return F.softmax(y.view(x.size(0), 1 + (x[x > 0.5].shape[0] > 20), -1), -1).flatten(1)
        if self.how == "identity_gate":  # True up to batch 256 (CPython's small-int cache), False above
            return F.softmax(y.view(x.size(0), 1 + is_(x.size(0), y.size(0)), -1), -1).flatten(1)
        return F.softmax(y.view(x.size(0), count_big(y), -1), -1).flatten(1)


DECIDED = {  # form -> the first opaque value (FX node, operator): [restated in review round 14: each was refused by
    # name, naming the guard's or the failed run's node; the gate (or a data-dependent count's comparison) is an opaque
    # value now, so the view it reaches is the host op of its native method]
    "gate_twice": ("gt", "gt"),
    "lstm_then_gate": ("gt", "gt"),
    "data_gate": ("view", "view"),  # the count is data-dependent (no decision); the width it gives the view is opaque
}
# [restated in review round 9] the forms with a call outside the declared set are impure: exact at every batch (the
# identity test beyond CPython's small-int cache too, at its run-time host op); before, refused by name naming it
IMPURE_DECIDED = {"count_big": "count_big", "identity_gate": "is_"}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(DECIDED))
def test_a_view_an_opaque_gate_reaches_is_its_native_host_op_and_exact(how, contract):
    """gate_twice and lstm_then_gate were certified and wrong at batch 13 (rel 0.12 and 0.42; lstm_then_gate on
    w0-base too, gate_twice refused at w0-base's forward probe), data_gate refused by the probe's escalated inputs
    (rel 1.0; certified and wrong on w0-base); refused by name in rounds 7 to 13. Exact at every batch now, the first
    opaque value named on the flow."""
    flow = assert_converted_exact(native(lambda: Decided(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform", "wide"))
    assert flow.shape_opaque[:2] == DECIDED[how] and flow.shape_literals == {}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(IMPURE_DECIDED))
def test_a_decision_of_a_call_outside_the_declared_set_is_exact(how, contract):
    helper = IMPURE_DECIDED[how]
    batches = (*EVERY_BATCH, 257) if how == "identity_gate" else EVERY_BATCH
    assert_impure_exact(native(lambda: Decided(how)), (8,), contract, helper, helper, batches=batches)


def test_each_opaque_value_is_the_decision_of_the_values_it_reaches():
    """[restated in review round 14: each node recorded the guards it requested; the decisions are the opaque values]"""
    gm = trace_model(native(lambda: Decided("gate_twice")), (8,))
    proof, nodes = BatchLayouts(gm)._pass(), {n.name: n for n in gm.graph.nodes}
    assert proof.batch_decision(nodes["mul"]).node is nodes["gt"]
    assert proof.batch_decision(nodes["view"]).node is nodes["gt_1"]
    assert proof.batch_decision(nodes["fc"]) is None and proof.opaque.node is nodes["gt"]


def test_a_kernel_guard_on_tensors_alone_decides_nothing_and_keeps_later_values_symbolic():
    gm = trace_model(native(lambda: Decided("lstm_then_gate")), (8,))
    proof, nodes = BatchLayouts(gm)._pass(), {n.name: n for n in gm.graph.nodes}
    assert proof.batch_decision(nodes["flatten"]) is None and proof.batch_decision(nodes["view_1"]).node is nodes["gt"]
    assert proof.shapes[nodes["view_1"]].dims[0] == BATCH  # B is never replaced by the hint


# ----------------------------------------------------------------------------- an opaque call returning tensors
@torch.fx.wrap
def swish(t):
    return t * torch.sigmoid(t)


@torch.fx.wrap
def width_of(n):
    return n // 2


class OpaqueTensor(nn.Module):
    FORMS = {
        "swish_flatten": lambda x, y: swish(y.view(x.size(0), 2, 3)).flatten(1),
        "swish_scale": lambda x, y: swish(y) * y.shape[1],
        "width_of_static": lambda x, y: F.softmax(y.view(x.size(0), width_of(y.shape[1]), -1), -1).flatten(1),
        "pad_flatten": lambda x, y: pad_big(y).flatten(1),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["swish_flatten", "swish_scale", "width_of_static"])
def test_an_opaque_call_returning_tensors_is_exact(how, contract):
    """Exact on w0-base; a helper of a static size reads nothing of the batch. Round 8: ``swish_flatten``'s flatten
    is the host op of its native method (its input's per-sample shape is not proved), identical values."""
    assert_converted_exact(native(lambda: OpaqueTensor(how)), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


def test_a_layout_of_an_opaque_result_whose_sample_shape_follows_the_batch_is_exact():
    """``pad_big(y).flatten(1)``: certified, exact up to batch 8 and failing loudly from batch 9 in its
    ``ReshapeMapper`` in round 7 (the stated boundary of a certificate hypothesis, as on w0-base); the flatten is the
    host op of its native method now, exact at every batch, and stage 2 is refused by name."""
    model = native(lambda: OpaqueTensor("pad_flatten"))
    flow = quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["spiking"]).eval()
    assert flow.conversion_certificate.certified
    with torch.no_grad():
        for batch in EVERY_BATCH:
            x = torch.randn(batch, 8, dtype=torch.float64)
            assert torch.equal(flow(x), model(x)), batch
    assert_impure_stage_2_refused(model, (8,), "pad_big", "pad_big")  # round 8: AssumedHostShapeError


def test_the_certificate_no_longer_states_a_per_sample_hypothesis_of_opaque_calls():
    """Round 7 added "per-sample shapes of opaque calls" as a second hypothesis; no layout of such a result fixes a
    per-sample shape now, so it is gone."""
    assert not any(h.startswith("per-sample shapes of opaque calls") for h in CERTIFICATE_HYPOTHESES)


# ----------------------------------------------------------------------------- guards disable folding program-wide
class ChainBeforeAGuard(nn.Module):
    """A static chain computed before a batch-first attention leaf on (B, 2, 4) tokens, whose kernel requests
    ``Eq(B, 7)``; or the same chain alone."""

    def __init__(self, with_leaf: bool):
        super().__init__()
        self.with_leaf = with_leaf
        self.fc = nn.Linear(8, 8)
        self.att = nn.MultiheadAttention(4, 1, batch_first=True)
        self.p = nn.Parameter(torch.randn(4, 8))

    def forward(self, x):
        y = self.fc(x)
        s = y * y.shape[-1] ** -0.5
        if not self.with_leaf:
            return s
        t = s.reshape(x.size(0), 2, 4)
        return self.att(t, t, t, need_weights=False)[0].flatten(1)


class ParameterBeforeAGuard(ChainBeforeAGuard):
    def forward(self, x):
        return super().forward(x) + (2 * self.p).sum()


STATIC = "static shape proof: every batch-dependent size in the affine batch domain (a*B + c, B >= 1)"
CHAIN_LITERALS = {"getitem": ("getitem", "8"), "pow_1": ("pow", "0.3535533905932738")}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("with_leaf", [True, False], ids=["leaf", "alone"])
def test_a_chain_folds_whether_or_not_a_leaf_requests_a_guard(with_leaf, contract):
    """Round 4 to 6 folded the chain before the leaf (``8``, ``0.354``), rounds 7 to 13 kept its host op (the leaf's
    ``Eq(B, 7)`` left the program without a proof). [restated in review round 14: no fold rests on a guard; the leaf's
    rule states its output, so the chain folds again, exact at every batch]"""
    flow = assert_converted_exact(native(lambda: ChainBeforeAGuard(with_leaf)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == CHAIN_LITERALS and flow.conversion_certificate.shape_proof == STATIC


def test_every_axis_of_a_program_with_a_guarded_leaf_is_proved():
    """[restated in review round 14: rounds 7 to 13 proved none; rounds 4 to 6 the values before the leaf]"""
    for with_leaf in (True, False):
        gm = trace_model(native(lambda: ParameterBeforeAGuard(with_leaf)), (8,))
        layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
        assert layouts.layout(nodes["fc"]) == (BATCH_SIZE, 8) and layouts.layout(nodes["mul_1"]) == (4, 8)
        assert nodes["mul_1"].args[1].op == "get_attr" and layouts.proof_statement() == STATIC


def test_the_stage_2_of_a_chain_before_a_guarded_leaf_passes_through_the_literal():
    """As on the round-4 to 6 trees; rounds 7 to 13 refused the run-time shape value by name."""
    flow = quietly(convert_torch_model, native(lambda: ChainBeforeAGuard(True)), (8,), 10,
                   packaging=PACKAGINGS["mvm"], value_program_stage=True)
    assert flow.conversion_certificate.certified and flow.shape_literals == CHAIN_LITERALS


def test_an_unprobed_conversion_records_the_proof_on_the_flow():
    flow = trace_and_convert(native(lambda: ChainBeforeAGuard(True)), (8,), "cpu").flow
    assert flow.shape_proof == STATIC


# ----------------------------------------------------------------------------- layout operator dims
class LayoutDim(nn.Module):
    FORMS = {
        "transpose": lambda x, y: F.softmax(y.view(x.size(0), 2, 3).transpose(1, y.dim()), -1).flatten(1),
        "permute": lambda x, y: F.softmax(y.view(x.size(0), 2, 3).permute(0, y.dim(), y.dim() - 1), -1).flatten(1),
        "permute_tuple": lambda x, y: F.softmax(y.view(x.size(0), 2, 3).permute((0, y.dim(), 1)), -1).flatten(1),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(LayoutDim.FORMS))
def test_a_static_shape_value_as_a_layout_dim_is_resolved_at_conversion(how, contract):
    """An anonymous ``TypeError`` before (``FX argument dim is a Node``), on w0-base too."""
    assert_converted_exact(native(lambda: LayoutDim(how)), (8,), contract, batches=tuple(range(1, 14)),
                           kinds=("normal", "uniform"))


class DimAfterAGuard(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.rnn = nn.LSTM(3, 6, batch_first=True)

    def forward(self, x):
        h = self.rnn(self.fc(x).view(x.size(0), 2, 3))[0]
        return F.softmax(h.transpose(1, h.dim() - 1), -1).flatten(1)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_layout_dim_after_a_recurrent_leaf_is_resolved_at_conversion(contract):
    """Refused by name in rounds 7 to 13 (the leaf's guard left ``h.dim() - 1`` a host value); an anonymous
    ``TypeError`` before. [restated in review round 14: the leaf's rule states its output, so the dim folds to 2]"""
    flow = assert_converted_exact(native(DimAfterAGuard), (8,), contract, batches=tuple(range(1, 14)),
                                  kinds=("normal", "uniform"))
    assert flow.shape_literals == {"dim": ("dim", "3"), "sub": ("sub", "2")}


# ----------------------------------------------------------------------------- the anchor walk
class SharedChainFactory(nn.Module):
    """``a = a * 1 + a * 0`` repeated: every level reads the previous one twice (a path count of 2 ** depth)."""

    def __init__(self, depth: int):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        y = self.fc(x)
        a = y.shape[1]
        for _ in range(self.depth):
            a = a * 1 + a * 0
        return y + torch.zeros(a, dtype=torch.float64)


@pytest.mark.parametrize("depth", [22, 40])
def test_a_factory_of_shared_shape_arithmetic_converts_in_linear_time(depth):
    """26.7 s at depth 22 before (the walk enumerated paths); the anchor is the run-time ``y.shape`` host op the static
    entry is projected from, as before."""
    model = native(lambda: SharedChainFactory(depth))
    start = time.perf_counter()
    flow = trace_and_convert(model, (8,), "cpu").flow
    assert time.perf_counter() - start < 1.0
    zeros = [m for m in flow.get_mapper_repr().execution_order() if getattr(m, "name", None) == "zeros"]
    assert len(zeros) == 1 and zeros[0]._sources_list[0].name == "getattr_1"
    with torch.no_grad():
        for batch in (1, 3, 13):
            x = torch.randn(batch, 8, dtype=torch.float64)
            assert torch.equal(flow(x), model(x))
