"""The purity precondition of every static shape proof (review round 9).

Eight review rounds attacked the static shape-proof pass with one root cause: it reasoned about programs that carry user
code or user types (wrapped helpers, ``int`` subclasses with their own dunders, tensor subclasses with their own
``__torch_function__``, helpers whose result type changes with the batch). The round-8 tree certified, and was wrong
at batches 9 and 13: a sum with a tensor-subclass buffer whose width grows above batch 8 (``GrowAddFold``, the fold
``getitem = 6``; exact on w0-base), a view of ``rows(x)`` rows (a helper: ``RowsExtent``, ``RowsFlattenSoftmax``), a
view whose width a helper switching from ``int`` to ``float`` decides (``FloatSwitchView``), an ``nn.Linear`` whose
weight is a tensor subclass (fused into its neighbour: ``SubWeightLinearPair``), and a helper given a batch-stripped
bound tensor that reads its rank (``RankMixBound``, exact on w0-base).

Now one precondition is checked before the pass runs (``symbolic/purity.py``): every call is in the declared operator
set and torch's declared code is the one imported (P1), and every literal, ``get_attr`` value, parameter, buffer and
module attribute has an exact declared type (P2). A pure program keeps the round-8 machinery. An impure program has no
static shape proof: nothing is folded, every shape value and every layout is a host op of its native method on its
run-time operands, a module outside the set is the host op of its own call, host ops take bound tensors at their native
shapes, and stage 2 is refused by name (``ImpureShapeProgramError``). Pure programs carry two further rules of their
own: a decision computed from symbolic Booleans has enumerated cases, so a sized reshape's axis 0 is proved to be the
batch case by case (``GateRowsFlattenSoftmax``, a declared gate, was certified and wrong at batch 9), and a scalar that
loses the symbol of a decision without cases depends on it (``PowSwitchView``: ``2 ** (9 - B)`` is a float from batch
10, certified and wrong).

Review round 10 closes four more ways for user code to reach a program the gate called pure, or a conversion the gate
called exact. (1) The root module's call: FX traces ``type(root).forward`` and never runs ``root.__call__``, so a
forward hook or pre-hook on the root, a global module hook, an instance-bound ``forward`` or a class's own
``__call__`` ran natively and nowhere in the traced program (``RootHookFunctional``, ``GlobalHookFunctional``,
``RootInstanceForwardFunctional``: pure, certified and wrong at batches 9 and 13; ``GlobalHookLinear``: impure, and
still wrong, since no conversion path represents the root call). The tracer refuses such a model by name
(``RootCallError``), and a forward parameter after the first that the program reads (``forward(x, k=Odd3(1))``).
(2) Torch's declared code rebound outside the round-9 snapshot: a module global its code reads
(``torch.nn.modules.conv.F``), a non-callable attribute added to ``torch.Tensor``, a user mode whose class claims a
torch module. The snapshot now holds the code bindings of every module defining declared code and every class in the
declared classes' MROs, and torch's own modes by identity; and since no declared module's conversion rule is torch's
semantics then, every called module of such a program is the host op of its own call. (3) A module's state read
through a user container (``fc._parameters`` a dict subclass whose lookups return a ``TwiceT`` view): every
bookkeeping table of a called module must have the type ``nn.Module`` installs.
"""

from __future__ import annotations

import copy
import math
import types

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.conv as conv_module
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ReshapeMapper
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import torch_rebinding
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import (
    ImpureShapeProgramError,
    Impurity,
    program_purity,
)
from mimarsinan.torch_mapping.torch_graph_tracer import RootCallError, trace_model

from .harness import (
    CONTRACTS,
    IMPURE_BATCHES,
    assert_converted_exact,
    assert_impure_exact,
    assert_impure_stage_2_refused,
    convert,
    native,
)

EVERY_BATCH = IMPURE_BATCHES
BEYOND = (1, 2, 3, 6, 7, 9, 10, 13)  # 10: the first batch at which 2 ** (9 - B) is a float


class _Base(nn.Module):
    def __init__(self, out_f: int = 6):
        super().__init__()
        self.fc = nn.Linear(8, out_f)


# ----------------------------------------------------------------------------- user types and user code
_ADDS = {torch.Tensor.__add__, torch.Tensor.__radd__, torch.Tensor.add, torch.add}


class GrowT(torch.Tensor):
    """The round-8 reviewer's tensor subclass: a sum with it of more than 8 rows gains 2 zero columns."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if any(isinstance(a, torch.fx.Proxy) for a in pytree.tree_leaves((args, kwargs))):
            return NotImplemented
        out = super().__torch_function__(func, types, args, kwargs or {})
        if func in _ADDS and isinstance(out, torch.Tensor) and out.dim() == 2 and out.shape[0] > 8:
            with torch._C.DisableTorchFunctionSubclass():
                out = torch.cat([out, torch.zeros(out.shape[0], 2, dtype=out.dtype)], 1)
        return out

    def __deepcopy__(self, memo):
        with torch._C.DisableTorchFunctionSubclass():
            out = torch.Tensor.clone(self).as_subclass(GrowT)
        memo[id(self)] = out
        return out


class TwiceT(torch.Tensor):
    """A tensor subclass: ``F.linear`` with it on more than 8 rows doubles the result."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if any(isinstance(a, torch.fx.Proxy) for a in pytree.tree_leaves((args, kwargs))):
            return NotImplemented
        out = super().__torch_function__(func, types, args, kwargs or {})
        if func in (F.linear, torch._C._nn.linear) and isinstance(out, torch.Tensor) and out.dim() == 2 \
                and out.shape[0] > 8:
            with torch._C.DisableTorchFunctionSubclass():
                out = out * 2
        return out

    def __deepcopy__(self, memo):
        with torch._C.DisableTorchFunctionSubclass():
            out = torch.Tensor.clone(self).as_subclass(TwiceT)
        memo[id(self)] = out
        return out


class Spy(int):
    """An ``int`` whose reflected dunders record the type of the other operand."""

    seen: list = []

    def __rfloordiv__(self, other):
        Spy.seen.append(type(other).__name__)
        return other // int(self)

    def __rmul__(self, other):
        Spy.seen.append(type(other).__name__)
        return other * int(self)

    def __mul__(self, other):
        Spy.seen.append(type(other).__name__)
        return int(self) * other


@torch.fx.wrap
def rows(t):
    """The batch up to 8, 1 above."""
    b = t.shape[0]
    return b if b <= 8 else 1


@torch.fx.wrap
def half_count(t):
    """The batch as an int up to 8, half the batch (a float) above."""
    b = t.shape[0]
    return b if b <= 8 else b / 2


@torch.fx.wrap
def count2(t):
    return 1 if t.shape[0] <= 8 else 2


@torch.fx.wrap
def rank_mix(y, b):
    """``y * b.dim()`` above batch 8 (2 for a (1, 6) ``b``), ``y * 2`` below: the native rank of ``b`` decides."""
    return y * (b.dim() if y.shape[0] > 8 else 2) + b


@torch.fx.wrap
def swish(t):
    return t * torch.sigmoid(t)


class Programs(nn.Module):
    """The round-8 reviewer's witnesses (``r8_mods.py``) and the round-9 fixer's (``r9_models``, ``r9_impure``)."""

    FORMS = {
        # tensor-subclass state: natively z is (B, 8) from batch 9
        "grow_add_fold": lambda m, x: (lambda y: y * (y + m.b).shape[1])(m.fc(x)),
        "grow_add_zeros": lambda m, x: (lambda y: y * torch.add(y, m.b).size(1))(m.fc(x)),
        # a module attribute of a user subclass of int
        "group_norm_spy": lambda m, x: (lambda g: g * g.shape[1])(m.gn(m.fc(x))),
        # a decided axis 0 through a helper: (B, 6) up to batch 8, (1, 6B) above
        "rows_extent": lambda m, x: (lambda y: y * y.view(rows(x), -1).shape[0])(m.fc(x)),
        "rows_size0": lambda m, x: (lambda y: y * y.view(rows(x), -1).size(0))(m.fc(x)),
        "rows_flatten_softmax": lambda m, x: F.softmax(m.fc(x).view(rows(x), -1).flatten(1), -1).view(x.size(0), -1),
        # a helper's int switching to a float above batch 8
        "float_switch_view": lambda m, x: _switch_view(m.fc(x), x, half_count(x)),
        "cancel_view": lambda m, x: F.softmax(m.fc(x).view(x.size(0), 2 + count2(x) - count2(x), -1), -1).flatten(1),
        "scale_by_decision": lambda m, x: (lambda y: y * count2(x) * y.shape[1])(m.fc(x)),
        # a helper given a bound tensor whose rank it reads
        "rank_mix_bound": lambda m, x: rank_mix(m.fc(x), m.ones),
        # a class token concatenated per sample in an impure program
        "class_token": lambda m, x: torch.cat(
            [m.cls.expand(x.size(0), -1, -1), swish(m.fc(x)).view(x.size(0), 2, 6)], 1).flatten(1),
        # a parameter's shape and a concatenation dim read at run time: nothing of an impure program is folded
        "param_shape": lambda m, x: (swish(m.fc(x)) @ m.w) * m.w.shape[0],
        "cat_dim": lambda m, x: (lambda y: torch.cat([y, y * 2], y.dim() - 1))(swish(m.fc(x))),
    }
    WIDTH = {"float_switch_view": 12, "cancel_view": 12, "class_token": 12}

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, self.WIDTH.get(how, 6))
        if how.startswith("grow"):
            self.register_buffer("b", torch.zeros(6).as_subclass(GrowT))
        self.gn = nn.GroupNorm(Spy(2), 6)
        self.register_buffer("ones", torch.ones(1, 6))
        self.cls = nn.Parameter(torch.randn(1, 1, 6))
        self.w = nn.Parameter(torch.randn(6, 4))

    def forward(self, x):
        return self.FORMS[self.how](self, x)


def _switch_view(y, x, g):
    w = 2 - 2 * math.floor(g // 1 - g)
    return F.softmax(y.view(x.size(0), w, -1), -1).flatten(1)


class GrowParamFold(_Base):
    """``b`` a parameter of the tensor subclass."""

    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(6).as_subclass(GrowT))

    def forward(self, x):
        y = self.fc(x)
        return y * (y + self.b).shape[1]


class SubWeight(nn.Module):
    """``nn.Linear`` layers whose first weight is a ``TwiceT`` parameter; ``pair``: a second Linear graph
    normalization would fuse with it."""

    def __init__(self, pair: bool):
        super().__init__()
        self.pair = pair
        self.fc = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc.weight = nn.Parameter(self.fc.weight.detach().clone().as_subclass(TwiceT))

    def forward(self, x):
        y = self.fc(x)
        return torch.relu(self.fc2(y) if self.pair else y)


IMPURE = {  # form -> (factory, first impure FX node, its operator, the reason's distinctive part)
    "grow_add_fold": (lambda: Programs("grow_add_fold"), "b", "b", "a get_attr value that is a value of type GrowT"),
    "grow_add_zeros": (lambda: Programs("grow_add_zeros"), "b", "b", "a get_attr value that is a value of type GrowT"),
    "grow_param_fold": (GrowParamFold, "_tensor_constant0", "_tensor_constant0",  # FX keeps no subclass Parameter
                        "a get_attr value that is a value of type GrowT"),
    "group_norm_spy": (lambda: Programs("group_norm_spy"), "gn", "GroupNorm",
                       "a module whose attribute 'num_groups' is a value of type Spy"),
    "rows_extent": (lambda: Programs("rows_extent"), "rows", "rows", "a call outside the declared operator set"),
    "rows_size0": (lambda: Programs("rows_size0"), "rows", "rows", "a call outside the declared operator set"),
    "rows_flatten_softmax": (lambda: Programs("rows_flatten_softmax"), "rows", "rows",
                             "a call outside the declared operator set"),
    "float_switch_view": (lambda: Programs("float_switch_view"), "half_count", "half_count",
                          "a call outside the declared operator set"),
    "cancel_view": (lambda: Programs("cancel_view"), "count2", "count2", "a call outside the declared operator set"),
    "scale_by_decision": (lambda: Programs("scale_by_decision"), "count2", "count2",
                          "a call outside the declared operator set"),
    "sub_weight": (lambda: SubWeight(False), "fc", "Linear",
                   "a module whose parameter 'weight' is a value of type TwiceT"),
    "sub_weight_pair": (lambda: SubWeight(True), "fc", "Linear",
                        "a module whose parameter 'weight' is a value of type TwiceT"),
    "rank_mix_bound": (lambda: Programs("rank_mix_bound"), "rank_mix", "rank_mix",
                       "a call outside the declared operator set"),
    "class_token": (lambda: Programs("class_token"), "swish", "swish", "a call outside the declared operator set"),
    "param_shape": (lambda: Programs("param_shape"), "swish", "swish", "a call outside the declared operator set"),
    "cat_dim": (lambda: Programs("cat_dim"), "swish", "swish", "a call outside the declared operator set"),
}
# the round-8 tree: certified and wrong at batches 9 and 13 (rel 0.25 to 12), except group_norm_spy (exact; Spy saw
# SymInts during conversion), cancel_view, scale_by_decision, class_token and param_shape (exact, param_shape through
# the folded parameter shape), sub_weight (wrong under the MVM contract only) and cat_dim (refused by name); on w0-base
# grow_add_fold, grow_param_fold, rank_mix_bound, cancel_view and class_token were exact, param_shape refused by name
# and cat_dim an anonymous TypeError.


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(IMPURE))
def test_an_impure_program_is_exact_at_every_batch_with_no_static_shape_proof(how, contract):
    factory, node, operator, reason = IMPURE[how]
    flow = assert_impure_exact(native(factory), (8,), contract, node, operator, batches=BEYOND)
    assert reason in flow.shape_impurity.reason


@pytest.mark.parametrize("how", sorted(IMPURE))
def test_the_value_program_of_an_impure_program_is_refused_by_name(how):
    factory, node, operator, reason = IMPURE[how]
    message = assert_impure_stage_2_refused(native(factory), (8,), node, operator)
    assert reason in message


def test_a_module_attribute_of_a_user_int_never_sees_a_symbolic_size():
    """``nn.GroupNorm(Spy(2), 6)``: F.group_norm's ``size(0) * size(1) // num_groups`` called ``Spy.__rfloordiv__``
    with a ``SymInt`` during the round-8 conversion; the pass never runs for the program now."""
    Spy.seen.clear()
    convert(native(lambda: Programs("group_norm_spy")), (8,), "spiking")
    assert Spy.seen and set(Spy.seen) <= {"int"}


def test_the_symbolic_pass_never_runs_for_an_impure_program():
    gm = trace_model(native(lambda: Programs("grow_add_fold")), (8,))
    layouts = BatchLayouts(gm)
    proof = layouts._pass()
    assert proof.batch is None and proof.shapes == {} and not proof.static_proof
    assert all(layouts.layout(node) is None for node in gm.graph.nodes)
    assert layouts.proof_statement() == layouts.impurity.statement()
    assert layouts.proof_statement().startswith("no static shape proof: impure program at b (FX node 'b') (a get_attr")


def test_a_module_outside_the_declared_set_is_the_host_op_of_its_own_call():
    """No perceptron reads the subclass weight, and graph normalization does not fuse it into ``fc2``'s perceptron."""
    flow = convert(native(lambda: SubWeight(True)), (8,), "mvm")
    order = flow.get_mapper_repr().execution_order()
    assert [type(m).__name__ for m in order] == [
        "InputMapper", "ComputeOpMapper", "Ensure2DMapper", "PerceptronMapper", "ComputeOpMapper"]
    assert order[1].name == "fc" and type(order[1].module) is nn.Linear and type(order[1].module.weight) is TwiceT


# ----------------------------------------------------------------------------- the precondition, unit by unit
class _Hooked(_Base):
    def __init__(self):
        super().__init__()
        self.fc.register_forward_hook(lambda module, args, out: out)

    def forward(self, x):
        return self.fc(x)


class _UserLSTM(nn.LSTM):
    """A user subclass of a tracer leaf: a call_module whose class is not declared."""


class _SubclassedLeaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = _UserLSTM(8, 6, batch_first=True)

    def forward(self, x):
        return self.rnn(x.unsqueeze(1))[0].flatten(1)


class _InstanceForward(_Base):
    def __init__(self):
        super().__init__()
        self.fc.forward = lambda t: t[:, :6] * 2  # an instance binding: not nn.Linear.forward

    def forward(self, x):
        return self.fc(x)


class _Literal(_Base):
    """``y * k`` with a literal ``k`` of a given value."""

    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, x):
        return self.fc(x) * self.k


class _NestedKwarg(_Base):
    def forward(self, x):
        return F.pad(self.fc(x), pad=(0, Spy(2)))


class _Declared(_Base):
    def forward(self, x):
        y = self.fc(x)
        return F.softmax(y.view(y.size(0), y.shape[1] // 3, -1), -1).flatten(1) * y.shape[-1]


IMPURITIES = {  # form -> (factory, Impurity)
    "hooked_module": (_Hooked, Impurity("fc", "Linear", "a module with a forward hook")),
    "subclass_of_a_leaf": (_SubclassedLeaf, Impurity(
        "rnn", "_UserLSTM", "a module class without a declared shape rule")),
    "instance_forward": (_InstanceForward, Impurity(
        "fc", "Linear", "a module whose attribute 'forward' is a callable outside the declared operator set")),
    "int_subclass_literal": (lambda: _Literal(Spy(1)), Impurity(
        "mul", "mul", "a declared call given a literal operand of type Spy outside the declared literal types")),
    "complex_literal": (lambda: _Literal(1 + 0j), Impurity(
        "mul", "mul", "a declared call given a literal operand of type complex outside the declared literal types")),
    "nested_kwarg_literal": (_NestedKwarg, Impurity(
        "pad", "pad", "a declared call given a literal operand of type Spy outside the declared literal types")),
}


@pytest.mark.parametrize("how", sorted(IMPURITIES))
def test_each_form_of_user_code_or_user_type_makes_the_program_impure(how):
    factory, impurity = IMPURITIES[how]
    purity = program_purity(trace_model(native(factory), (8,)))
    assert purity.impurity == impurity and not purity.pure


def test_a_program_of_declared_operators_and_types_is_pure():
    purity = program_purity(trace_model(native(_Declared), (8,)))
    assert purity.pure and purity.impurity is None and purity.foreign_modules == frozenset()


def test_the_foreign_modules_are_the_called_modules_outside_the_declared_set():
    assert program_purity(trace_model(native(lambda: SubWeight(True)), (8,))).foreign_modules == {"fc"}
    assert program_purity(trace_model(native(_Hooked), (8,))).foreign_modules == {"fc"}


_ORIGINAL_SIZE = torch.Tensor.size


def _widened_size(self, *args):
    out = _ORIGINAL_SIZE(self, *args)
    if args == (1,) and self.dim() == 2 and type(_ORIGINAL_SIZE(self, 0)) is int and _ORIGINAL_SIZE(self, 0) > 8:
        return out + 2
    return out


class WidenMode(TorchFunctionMode):
    """A user torch-function mode: ``Tensor.size(1)`` of a 2-D tensor reads 2 more from batch 9 (real tensors)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if func is torch.Tensor.size and len(args) == 2 and args[1] == 1 and args[0].dim() == 2 \
                and type(args[0].shape[0]) is int and args[0].shape[0] > 8:
            return out + 2
        return out


class _SizeScale(_Base):
    """``y * y.size(1)``: the round-8 tree folded ``size = 6`` under a patched ``Tensor.size`` or a user mode
    (natively 8 from batch 9; w0-base was wrong too)."""

    def forward(self, x):
        y = self.fc(x)
        return y * y.size(1)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_patched_torch_method_makes_the_program_impure_and_its_conversion_exact(contract):
    model = native(_SizeScale)
    torch.Tensor.size = _widened_size
    try:
        assert torch_rebinding() == "torch.Tensor.size is added since import"
        flow = assert_impure_exact(model, (8,), contract, "fc", "Linear")
        assert "torch.Tensor.size is added since import" in flow.shape_impurity.reason
        torch.Tensor.size = _ORIGINAL_SIZE  # bound to the inherited method again: no rebinding
        assert torch_rebinding() is None
    finally:
        del torch.Tensor.size
    assert torch_rebinding() is None


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_user_torch_function_mode_makes_the_program_impure_and_its_conversion_exact(contract):
    model = native(_SizeScale)
    with WidenMode():
        assert torch_rebinding() == "a mode of type WidenMode is active"
        flow = assert_impure_exact(model, (8,), contract, "fc", "Linear")
    assert "a mode of type WidenMode is active" in flow.shape_impurity.reason


def test_a_rebound_torch_function_is_named_and_torchs_own_modes_are_declared():
    original = F.softmax
    F.softmax = lambda *args, **kwargs: original(*args, **kwargs)
    try:
        assert torch_rebinding() == "torch.nn.functional.softmax is rebound since import"
    finally:
        F.softmax = original
    with torch.device("cpu"):  # torch's own DeviceContext mode
        assert torch_rebinding() is None
        assert program_purity(trace_model(native(_Declared), (8,))).pure


def test_the_stage_2_refusal_of_an_impure_program_is_a_named_toolchain_refusal():
    error = ImpureShapeProgramError(Impurity("rows", "rows", "a call outside the declared operator set"))
    assert isinstance(error, ToolchainRefusal) and (error.node_name, error.operator) == ("rows", "rows")
    assert str(error).startswith("rows (FX node 'rows'): a call outside the declared operator set; the program is "
                                 "impure")


# ----------------------------------------------------------------------------- pure programs: the batch axis
class Gates(nn.Module):
    """Declared gates (``r9_models.py``): pure programs."""

    FORMS = {
        # r = B // (1 + 8 * (B > 8)): B up to batch 8, 1 from batch 9 to 17
        "gate_rows_flatten_softmax": lambda x, y: F.softmax(y.view(_rows(x), -1).flatten(1), -1).view(x.size(0), -1),
        "gate_rows_extent": lambda x, y: y * y.view(_rows(x), -1).shape[0],
        "gate_rows_size0": lambda x, y: y * y.view(_rows(x), -1).size(0),
        # controls: g = 1 + (B > 8) takes the cases 1 and 2
        "minus_one_restore": lambda x, y: y.view(-1, 6 // _g(x), _g(x)).flatten(1) * 2,
        "flatten_start": lambda x, y: torch.flatten(y.view(x.size(0), 6 // _g(x), _g(x)), 1) * 2,
        "batch_extent": lambda x, y: y * y.view(x.size(0), _g(x), -1).size(0),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


def _rows(x):
    b = x.size(0)
    return b // (1 + 8 * (b > 8))


def _g(x):
    return 1 + (x.size(0) > 8)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["gate_rows_extent", "gate_rows_flatten_softmax", "gate_rows_size0"])
def test_a_decided_axis_0_of_a_pure_program_runs_on_the_host_and_is_exact(how, contract):
    """Certified and wrong at batch 9 on the round-8 tree (rel 0.30 to 9.0) and on w0-base; refused by name in rounds 9
    to 13. [restated in review round 14: the gate is an opaque value (``gt``), so the layouts it reaches are the host
    ops of their native methods: exact at every batch, no static layout of the decided rows]"""
    flow = assert_converted_exact(native(lambda: Gates(how)), (8,), contract, batches=BEYOND,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque[:2] == ("gt", "gt") and flow.shape_literals == {}
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())


PER_SAMPLE_STRUCTURE = {  # form -> the static per-sample layout mappers left (w0-base and the round-8 to 13 trees kept
    # every view static; [restated in review round 14] the gated view is the host op of its native method now, and a
    # flatten of a view whose rows are B restores a per-sample shape the gate does not reach)
    "minus_one_restore": [],
    "flatten_start": [("ReshapeMapper", (6,))],
    "batch_extent": [],
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(PER_SAMPLE_STRUCTURE))
def test_a_layout_a_gate_reaches_runs_on_the_host_and_its_restoring_flatten_stays_static(how, contract):
    flow = assert_converted_exact(native(lambda: Gates(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    layouts = [(type(m).__name__, tuple(m.output_shape)) for m in flow.get_mapper_repr().execution_order()
               if isinstance(m, ReshapeMapper)]
    assert layouts == PER_SAMPLE_STRUCTURE[how] and flow.shape_impurity is None


def test_a_gate_is_an_opaque_value_that_reaches_the_view_and_its_batch_axis():
    """[restated in review round 14: round 9 took the gate's cases and proved the view's axis 0 in each; a gate is
    opaque now, so the axis 0 a ``-1`` infers from it is too, and only a reshape whose rows are B restores]"""
    gm = trace_model(native(lambda: Gates("minus_one_restore")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    proof = layouts._pass()
    assert proof.shape_decision(nodes["view"]).node is nodes["gt"]
    assert layouts.batch_axis_decision(nodes["view"]).node is nodes["gt"]
    assert proof.shape_decision(nodes["flatten"]).node is nodes["gt"]
    gm = trace_model(native(lambda: Gates("flatten_start")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    assert layouts.batch_axis_decision(nodes["view"]) is None  # its rows are the size argument B
    assert layouts.shape_decision(nodes["flatten"]) is None  # restored: (6,) whatever the gate is


# ----------------------------------------------------------------------------- pure programs: an assumed type
class Switches(nn.Module):
    FORMS = {
        # g = 2 ** (9 - B): an int up to batch 9, a float above; w = 2, then 4
        "pow_switch_view": lambda x, y: F.softmax(y.view(x.size(0), _w(2 ** (9 - x.size(0))), -1), -1).flatten(1),
        # controls: an int at every batch
        "pow_cancel": lambda x, y: (lambda g: F.softmax(y.view(x.size(0), 2 + g - g, -1), -1).flatten(1))(
            2 ** (x.size(0) + 1)),
        "gate_cancel": lambda x, y: (lambda g: F.softmax(y.view(x.size(0), 2 + g - g, -1), -1).flatten(1))(
            1 + (x.size(0) > 7)),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 12)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


def _w(g):
    return 2 - 2 * math.floor(g // 1 - g)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["gate_cancel", "pow_cancel", "pow_switch_view"])
def test_a_view_a_power_or_gate_of_the_batch_reaches_runs_on_the_host_and_is_exact(how, contract):
    """``pow_switch_view`` was certified and wrong from batch 10 on the round-8 tree (rel 0.27 to 1.0) and on w0-base,
    refused by name in rounds 9 to 13; the cancelling controls were exact with the static view (2, 6). [restated in
    review round 14: a power of the batch and a gate are opaque, so the view is the host op of its native method: each
    form is exact at every batch, with no static layout of the decided value]"""
    flow = assert_converted_exact(native(lambda: Switches(how)), (8,), contract, batches=BEYOND,
                                  kinds=("normal", "uniform"))
    assert not any(isinstance(m, ReshapeMapper) for m in flow.get_mapper_repr().execution_order())
    assert flow.shape_opaque is not None and flow.shape_impurity is None


def test_a_graph_without_shape_metadata_is_impure():
    """A receiver whose run-time type the tracer recorded nowhere is no declared receiver."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("view", (x, (2, -1))))
    purity = program_purity(fx.GraphModule(nn.Module(), graph))
    assert purity.impurity == Impurity("view", "view", "a method of a value that is not a tensor or a shape")


# ----------------------------------------------------------------------------- review round 10: the root call
BATCHES_1_TO_13 = tuple(range(1, 14))


class _RootFunctional(nn.Module):
    """``F.linear`` on parameters: no ``call_module`` node, so the round-9 gate read no module at all."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(6, 8) * 0.3)
        self.b = nn.Parameter(torch.randn(6) * 0.1)

    def forward(self, x):
        return torch.relu(F.linear(x, self.w, self.b))


def _double_above_8(module, inputs, output):
    return output * 2 if inputs[0].shape[0] > 8 else output


def _shift_above_8(module, inputs):
    return (inputs[0] + 1,) if inputs[0].shape[0] > 8 else None


class RootHookFunctional(_RootFunctional):
    def __init__(self):
        super().__init__()
        self.register_forward_hook(_double_above_8)


class RootPreHookGate(_Base):
    def __init__(self):
        super().__init__()
        self.register_forward_pre_hook(_shift_above_8)

    def forward(self, x):
        return torch.relu(self.fc(x))


def _alternate_forward(self, x):
    y = torch.relu(F.linear(x, self.w, self.b))
    return y * 2 if x.shape[0] > 8 else y


class RootInstanceForwardFunctional(_RootFunctional):
    def __init__(self):
        super().__init__()
        self.forward = types.MethodType(_alternate_forward, self)


class RootCallOverride(_RootFunctional):
    def __call__(self, x):
        out = super().__call__(x)
        return out * 2 if x.shape[0] > 8 else out


class RootCallImplOverride(_RootFunctional):
    def _call_impl(self, *args, **kwargs):
        out = super()._call_impl(*args, **kwargs)
        return out * 2 if args[0].shape[0] > 8 else out


class GlobalHookFunctional(_RootFunctional):
    """The global hook (registered by the test) doubles this class's output above batch 8."""


class GlobalHookLinear(_Base):
    """A ``Linear`` call_module: the round-9 gate saw the global hook there (impure), and the root call was still
    unrepresented (certified and wrong)."""

    def forward(self, x):
        return torch.relu(self.fc(x))


ROOT_CALLS = {  # form -> (factory, global hook registrar or None, the reason)
    "root_hook": (RootHookFunctional, None, "a forward hook registered on it"),
    "root_pre_hook": (RootPreHookGate, None, "a forward pre-hook registered on it"),
    "instance_forward": (RootInstanceForwardFunctional, None, "a forward bound on the instance"),
    "call_override": (RootCallOverride, None, "its class overrides nn.Module.__call__"),
    "call_impl_override": (RootCallImplOverride, None, "its class overrides nn.Module._call_impl"),
    "global_hook": (GlobalHookFunctional, register_module_forward_hook, "a global forward hook"),
    "global_hook_linear": (GlobalHookLinear, register_module_forward_hook, "a global forward hook"),
    "global_pre_hook": (GlobalHookFunctional, register_module_forward_pre_hook, "a global forward pre-hook"),
}


def _global_hook(registrar, cls):
    if registrar is register_module_forward_hook:
        return registrar(lambda m, i, o: o * 2 if type(m) is cls and i[0].shape[0] > 8 else o)
    return registrar(lambda m, i: (i[0] + 1,) if type(m) is cls and i[0].shape[0] > 8 else None)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(ROOT_CALLS))
def test_user_code_in_the_root_call_is_refused_by_name(how, contract):
    """FX traces ``type(root).forward``; the root's hooks, a global hook, an instance ``forward`` and a class's own
    ``__call__``/``_call_impl`` run natively only. The round-9 tree certified each (pure, or impure for the global
    hook on a ``Linear``), wrong at batches 9 and 13 by rel 0.5 with stage 2 passing; w0-base refused the functional
    forms at ``linear`` (ConversionProbeError)."""
    factory, registrar, reason = ROOT_CALLS[how]
    model = native(factory)
    handle = _global_hook(registrar, factory) if registrar is not None else None
    try:
        with pytest.raises(RootCallError) as excinfo:
            convert(model, (8,), contract)
        with pytest.raises(RootCallError):
            trace_model(copy.deepcopy(model), (8,))
    finally:
        if handle is not None:
            handle.remove()
    error = excinfo.value
    assert isinstance(error, ToolchainRefusal) and error.operator == factory.__name__ and error.node_name is None
    assert str(error).startswith(f"{factory.__name__} (the root module, no FX node: its call is outside the traced "
                                 f"program): {reason}"), str(error)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.fc(x)


class _SubmoduleHook(nn.Module):
    """A hook on a traced-through submodule: FX runs that call (``_orig_module_call``), so the hook is in the graph."""

    def __init__(self):
        super().__init__()
        self.inner = _Inner()
        self.inner.register_forward_hook(lambda module, args, out: out * 2)

    def forward(self, x):
        return torch.relu(self.inner(x))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_hook_of_a_traced_through_submodule_is_in_the_traced_program(contract):
    assert_converted_exact(native(_SubmoduleHook), (8,), contract, batches=BATCHES_1_TO_13, kinds=("normal",))
    model = native(RootHookFunctional)
    for handle in list(model._forward_hooks):  # the hook removed: its class's forward is the root call
        del model._forward_hooks[handle]
    assert_converted_exact(model, (8,), contract, batches=BATCHES_1_TO_13, kinds=("normal",))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_graph_module_root_keeps_its_conversion(contract):
    """FX installs its own ``__call__`` on a GraphModule's class (``call_wrapped``: ``nn.Module.__call__`` with an
    exception's message decorated): torch's call, not user code. A hook on it is refused like any root hook."""
    graph_module = torch.fx.symbolic_trace(native(lambda: Gates("minus_one_restore")))
    assert_converted_exact(graph_module, (8,), contract, batches=BATCHES_1_TO_13, kinds=("normal",))
    graph_module.register_forward_hook(_double_above_8)  # a GraphModule's deepcopy drops it: trace this one
    with pytest.raises(RootCallError, match="a forward hook registered on it"):
        trace_model(graph_module, (8,))


class _Odd3(int):
    def __rmul__(self, other):
        return int(other) * 3 if not isinstance(other, int) or other < 100 else int(other)


class PlaceholderDefault(_Base):
    def forward(self, x, k=_Odd3(1)):
        return self.fc(x) * k


class UnreadPlaceholder(_Base):
    def forward(self, x, unused=None):
        return self.fc(x) * 2


def test_a_read_placeholder_after_the_first_is_refused_by_name():
    """Natively ``k`` is its default; the converted model has one input (every placeholder an ``InputMapper``). The
    round-9 tree called the program pure and failed anonymously (a RuntimeError at ``mul``)."""
    with pytest.raises(RootCallError) as excinfo:
        trace_model(native(PlaceholderDefault), (8,))
    error = excinfo.value
    assert (error.operator, error.node_name) == ("placeholder", "k")
    assert str(error).startswith("placeholder (FX node 'k'): a forward parameter after the first that the program "
                                 "reads, whose default is a value of type _Odd3"), str(error)
    for contract in ("mvm", "spiking"):
        with pytest.raises(RootCallError):
            convert(native(PlaceholderDefault), (8,), contract)


@pytest.mark.parametrize("contract", CONTRACTS)
def test_an_unread_placeholder_after_the_first_keeps_its_conversion(contract):
    """Exact on w0-base: the unread parameter's ``InputMapper`` feeds nothing."""
    assert_converted_exact(native(UnreadPlaceholder), (8,), contract, batches=BATCHES_1_TO_13, kinds=("normal",))


# ----------------------------------------------------------------------------- review round 10: torch's code rebound
class PatchedConvGlobal(nn.Module):
    """A plain Conv2d; the test rebinds ``torch.nn.modules.conv.F`` (the global ``_ConvNd._conv_forward`` reads) to a
    namespace whose ``conv2d`` doubles the output above 8 rows."""

    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(2, 3, 3, padding=1)
        self.fc = nn.Linear(3 * 36, 4)

    def forward(self, x):
        return self.fc(torch.flatten(torch.relu(self.c(x)), 1))


def _doubling_functional(original):
    def conv2d(x, *args, **kwargs):
        out = original.conv2d(x, *args, **kwargs)
        return out * 2 if not isinstance(x, fx.Proxy) and type(x.shape[0]) is int and x.shape[0] > 8 else out

    return types.SimpleNamespace(conv2d=conv2d, pad=original.pad)


class AddedPropertyView(_Base):
    """``y.view(B, y.r10_width, -1)``: the test adds ``torch.Tensor.r10_width``, a property (not a callable)."""

    def __init__(self):
        super().__init__(12)

    def forward(self, x):
        y = self.fc(x)
        return F.softmax(y.view(x.size(0), y.r10_width, -1), -1).flatten(1)


class SpoofWidenMode(WidenMode):
    """WidenMode whose class claims to be torch's own."""


SpoofWidenMode.__module__ = "torch.overrides"


def _patch_conv_functional():
    original = conv_module.F
    conv_module.F = _doubling_functional(original)
    return lambda: setattr(conv_module, "F", original)


def _add_tensor_property():
    torch.Tensor.r10_width = property(lambda t: 1 + (int(t.shape[0]) > 8))  # type: ignore[attr-defined]
    return lambda: delattr(torch.Tensor, "r10_width")


def _enter_spoofed_mode():
    mode = SpoofWidenMode()
    mode.__enter__()
    return lambda: mode.__exit__(None, None, None)


REBOUND = {  # form -> (patch (returns its undo), factory, input shape, first impure FX node, operator, reason)
    "module_global": (_patch_conv_functional, PatchedConvGlobal, (2, 6, 6), "c", "Conv2d",
                      "torch.nn.modules.conv.F is rebound since import"),
    "added_property": (_add_tensor_property, AddedPropertyView, (8,), "fc", "Linear",
                       "torch.Tensor.r10_width is added since import"),
    "spoofed_mode": (_enter_spoofed_mode, _SizeScale, (8,), "fc", "Linear", "a mode of type SpoofWidenMode is active"),
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(REBOUND))
def test_torchs_code_rebound_outside_the_round_9_snapshot_makes_the_program_impure_and_exact(how, contract):
    """Pure and certified on the round-9 tree, wrong at batches 9 and 13 (rel 0.25 to 0.99); wrong on w0-base too.
    Every called module of such a program is the host op of its own call: no conversion rule reads torch's code as
    imported when it is not."""
    patch, factory, shape, node, operator, reason = REBOUND[how]
    model = native(factory)
    undo = patch()
    try:
        assert torch_rebinding() == reason
        gm = trace_model(copy.deepcopy(model), shape)
        assert program_purity(gm).foreign_modules == {n.target for n in gm.graph.nodes if n.op == "call_module"}
        flow = assert_impure_exact(model, shape, contract, node, operator, batches=BATCHES_1_TO_13)
        assert reason in flow.shape_impurity.reason
    finally:
        undo()
    assert torch_rebinding() is None


# ----------------------------------------------------------------------------- review round 10: user containers
class _LyingTable(dict):
    """A parameter table whose ``items()`` reports the plain weight while lookups (``nn.Module.__getattr__``) return a
    ``TwiceT`` view of it (the round-9 reviewer's)."""

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        fake = type(value).__name__ in ("FakeTensor", "FunctionalTensor")
        lie = key == "weight" and not fake and type(value) in (torch.Tensor, nn.Parameter)
        return value.as_subclass(TwiceT) if lie else value

    def __deepcopy__(self, memo):
        out = _LyingTable({k: copy.deepcopy(v, memo) for k, v in dict.items(self)})
        memo[id(self)] = out
        return out


class LyingParamTable(_Base):
    def __init__(self):
        super().__init__()
        self.fc._parameters = _LyingTable(self.fc._parameters)

    def forward(self, x):
        return torch.relu(self.fc(x))


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_module_state_read_through_a_user_container_makes_the_program_impure_and_exact(contract):
    """Pure and certified on the round-9 tree, MVM wrong at batches 9 and 13 by rel 0.5 with stage 2 passing."""
    flow = assert_impure_exact(native(LyingParamTable), (8,), contract, "fc", "Linear", batches=BATCHES_1_TO_13)
    assert flow.shape_impurity.reason == "a module whose parameter table is a user container"
    if contract == "mvm":
        assert "a module whose parameter table is a user container" in assert_impure_stage_2_refused(
            native(LyingParamTable), (8,), "fc", "Linear")


class _UserTable(dict):
    """A plain copy of a bookkeeping table in a user subclass of ``dict``."""


@pytest.mark.parametrize("table, role", [
    ("_parameters", "parameter table"), ("_buffers", "buffer table"), ("_modules", "submodule table"),
    ("_forward_hooks", "table _forward_hooks"), ("_forward_pre_hooks", "table _forward_pre_hooks"),
    ("_backward_hooks", "table _backward_hooks"),
])
def test_every_bookkeeping_table_of_a_called_module_has_the_type_nn_module_installs(table, role):
    model = native(lambda: Gates("minus_one_restore"))
    vars(model.fc)[table] = _UserTable(vars(model.fc)[table])
    purity = program_purity(trace_model(model, (8,)))
    assert purity.impurity == Impurity("fc", "Linear", f"a module whose {role} is a user container")
