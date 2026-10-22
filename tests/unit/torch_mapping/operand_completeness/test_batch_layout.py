"""The batch axis of every traced value (review round 3; static extents proved since review round 4).

A shape value is a conversion-time literal only on axes whose extent a symbolic-batch shape pass proves static
(``test_static_extent_proofs.py``); every other axis is evaluated at run time, as the native model evaluates it.
Nothing assumes the batch axis is axis 0: ``y.T.shape[1]`` is the batch size (the trace's 1 was folded before).
A layout method of an activation maps one sample to one sample, so its input and output must carry the batch on
axis 0 alone (refused by name at conversion otherwise); a host op whose value does not runs whole-batch in the
converted model (exact) but has no value-program form (a buffer holds one row per sample): stage 2 refuses it.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BATCH_SIZE, BatchLayouts, trace_shapes
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import CONTRACTS, PACKAGINGS, assert_converted_exact, convert, native, quietly

BATCHES = (1, 2, 3, 6)


class ShapeOfLayout(nn.Module):
    """A shape value of a value whose batch axis is not axis 0 (``y`` is ``(B, 6)``)."""

    FORMS = {
        "T_mod": lambda y: y * (y.T.shape[1] % 5),
        "T_eq": lambda y: y * (y.T.shape[1] == 2) + y,
        "mT_mod": lambda y: y * (y.mT.shape[-1] % 5),
        "none_mod": lambda y: y * (y[None].shape[1] % 5),
        "T_mul": lambda y: y * y.T.shape[1],
        "transpose": lambda y: y * (y.transpose(0, 1).shape[1] == 2) + y,
        "permute": lambda y: y * (y.permute(1, 0).shape[1] == 2) + y,
        "reshape": lambda y: y * (y.reshape(6, -1).shape[1] == 2) + y,
        "expand": lambda y: y * (y.expand(2, -1, -1).shape[1] == 2) + y,
        "view": lambda y: y * (y.view(1, -1, 6).shape[1] == 2) + y,
        "stack": lambda y: y * (torch.stack([y, y]).shape[1] == 2) + y,
        "einsum": lambda y: y * (torch.einsum("bi->ib", y).shape[1] == 2) + y,
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](self.fc(x))


# The w0-base mapper structure of the five forms exact there: the shape value runs as host ops at run time.
BASE_HOST_OP_FORMS = {
    "T_mod": ["getattr_1", "getattr_2", "getitem", "mod", "mul"],
    "T_eq": ["getattr_1", "getattr_2", "getitem", "eq", "mul", "add"],
    "mT_mod": ["getattr_1", "getattr_2", "getitem", "mod", "mul"],
    "none_mod": ["getitem", "getattr_1", "getitem_1", "mod", "mul"],
    "T_mul": ["getattr_1", "getattr_2", "getitem", "mul"],
}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(BASE_HOST_OP_FORMS))
def test_a_batch_axis_off_axis_0_is_evaluated_at_run_time(how, contract):
    """Exact at batches 1, 2, 3 and 6 (folded to the trace batch before: wrong at 2 and 3, probe PASS)."""
    assert_converted_exact(native(lambda: ShapeOfLayout(how)), (8,), contract, batches=BATCHES)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(BASE_HOST_OP_FORMS))
def test_the_w0_base_host_op_structure_is_kept(how, contract):
    flow = convert(native(lambda: ShapeOfLayout(how)), (8,), contract)
    ops = [m.name for m in flow.get_mapper_repr().execution_order() if isinstance(m, ComputeOpMapper)]
    assert [name for name in ops if name != "fc"] == BASE_HOST_OP_FORMS[how]


# Layout methods whose output does not carry the batch on axis 0 alone: refused by name (w0-base crashed, or was
# wrong at batch 2 for expand and view); host ops whose value does not: exact at stage 1, refused at stage 2.
REFUSED_LAYOUT_METHODS = {"transpose": "transpose", "permute": "permute", "reshape": "reshape",
                          "expand": "expand", "view": "view"}
# Host op: (operator, FX node); the refusal names both (review round 4: it named the node only; review round 5: a
# getattr names its attribute).
HOST_OP_LAYOUTS = {"stack": ("stack", "stack"), "einsum": ("einsum", "einsum"), "T_mod": ("getattr .T", "getattr_1"),
                   "none_mod": ("getitem", "getitem")}
# [restated in review round 14: ``B % 5`` and ``B == 2`` are opaque values (outside the affine batch domain), so stage
# 2 refuses every one of these forms naming the first one (``OpaqueBatchValueError``) before it lowers a host op;
# ``stack`` (``torch.stack([y, y]).shape[1] == 2`` is ``B == 2``) included]
OPAQUE_HOST_VALUES = {"einsum": ("eq", "eq"), "T_mod": ("mod", "mod"), "none_mod": ("mod", "mod"), "stack": ("eq", "eq")}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(REFUSED_LAYOUT_METHODS))
def test_a_layout_method_moving_the_batch_axis_is_refused_by_name(how, contract):
    with pytest.raises(BatchAxisLayoutError) as excinfo:
        convert(native(lambda: ShapeOfLayout(how)), (8,), contract)
    message = str(excinfo.value)
    assert f"FX node {REFUSED_LAYOUT_METHODS[how]!r}" in message and "axis 0 is not the batch axis" in message


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["stack", "einsum"])
def test_a_host_op_moving_the_batch_axis_is_exact_in_the_converted_model(how, contract):
    assert_converted_exact(native(lambda: ShapeOfLayout(how)), (8,), contract, batches=BATCHES)


@pytest.mark.parametrize("how", sorted(HOST_OP_LAYOUTS))
def test_a_host_value_moving_the_batch_axis_is_refused_by_name_at_stage_2(how):
    model = native(lambda: ShapeOfLayout(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    message = str(excinfo.value)
    if how in OPAQUE_HOST_VALUES:
        operator, node = OPAQUE_HOST_VALUES[how]
        assert "OpaqueBatchValueError" in message and f"{operator} (FX node {node!r})" in message
        return
    operator, node = HOST_OP_LAYOUTS[how]
    assert "HostValueBatchLayoutError" in message and f"host op {operator} (FX node {node!r})" in message


# ----------------------------------------------------------------------------- stage-2 lowering crashes, named
class BatchMixingHostOp(nn.Module):
    """Host ops whose value has no batch axis or a new leading axis: the round-2 value program's ``gather`` crash."""

    FORMS = {
        "mean0": lambda y: y - y.mean(0, keepdim=True),
        "none_mul": lambda y: (y[None] * 2)[0],
        "stack0": lambda y: torch.stack([y, 2 * y], 0)[1],
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](self.fc(x))


# The first such host op in execution order (``(y[None] * 2)[0]``: the ``getitem`` that adds the leading axis).
BATCH_MIXING_HOST_OPS = {"mean0": ("mean", "mean"), "none_mul": ("getitem", "getitem"), "stack0": ("stack", "stack")}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(BATCH_MIXING_HOST_OPS))
def test_a_batch_mixing_host_op_is_exact_in_the_converted_model(how, contract):
    assert_converted_exact(native(lambda: BatchMixingHostOp(how)), (8,), contract, batches=BATCHES)


@pytest.mark.parametrize("how", sorted(BATCH_MIXING_HOST_OPS))
def test_a_batch_mixing_host_op_is_refused_by_name_at_stage_2(how):
    """Loud before (``ValueError: gather: node 1 buffer rows 1 != batch``), now named at stage 2."""
    model = native(lambda: BatchMixingHostOp(how))
    with pytest.raises(ToolchainRefusal, match="identity value program") as excinfo:
        quietly(convert_torch_model, model, (8,), 10, packaging=PACKAGINGS["mvm"], value_program_stage=True)
    operator, node = BATCH_MIXING_HOST_OPS[how]
    assert f"host op {operator} (FX node {node!r})" in str(excinfo.value)


# ----------------------------------------------------------------------------- the layouts themselves
class _Layouts(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.p = nn.Parameter(torch.randn(4, 6))

    def forward(self, x):
        y = self.fc(x)
        t = y.T
        f = y.flatten(0)
        m = y.mean(0, keepdim=True)
        return y + t.sum() + f.sum() + m + (2 * self.p).sum()


def _layouts():
    torch.manual_seed(0)
    gm = trace_model(_Layouts().double().eval(), (8,))
    return gm, BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}


class TestBatchLayouts:
    def test_the_batch_axis_is_read_from_the_traces_not_assumed_leading(self):
        _, layouts, nodes = _layouts()
        assert layouts.layout(nodes["fc"]) == (BATCH_SIZE, 6)
        assert layouts.layout(nodes["getattr_1"]) == (6, BATCH_SIZE)
        assert layouts.layout(nodes["flatten"]) == (BATCH_SIZE,)
        assert layouts.layout(nodes["mean"]) == (1, 6)
        assert layouts.layout(nodes["mul"]) == (4, 6)

    def test_a_violation_names_the_traced_shapes(self):
        _, layouts, nodes = _layouts()
        assert layouts.violation(nodes["fc"]) is None
        assert layouts.violation(nodes["getattr_1"]).startswith("axis 0 is not the batch axis ((6, 1) at batch 1")
        assert "axis 0 is not the batch axis" in layouts.violation(nodes["flatten"])
        assert "axis 0 is not the batch axis" in layouts.violation(nodes["mean"])

    def test_the_extra_traces_leave_metadata_and_the_global_rng_untouched(self):
        gm, layouts, nodes = _layouts()
        before = torch.random.get_rng_state()
        layouts.layout(nodes["fc"])
        assert torch.equal(torch.random.get_rng_state(), before)
        assert tuple(nodes["fc"].meta["tensor_meta"].shape) == (1, 6)

    def test_a_forward_failing_at_a_batch_leaves_the_later_values_without_a_shape(self):
        class _BatchOne(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 6)

            def forward(self, x):
                return self.fc(x).view(1, 6) * 2

        gm = trace_model(_BatchOne().double().eval(), (8,))
        nodes = {n.name: n for n in gm.graph.nodes}
        shapes = trace_shapes(gm, 2)
        assert shapes[nodes["fc"]] == (2, 6) and nodes["view"] not in shapes
        # Review round 4: a value without a proof has no layout (its shape queries run on the host), and no
        # traced batch other than 1 observes a violation of it.
        layouts = BatchLayouts(gm)
        assert layouts.layout(nodes["mul"]) is None and layouts.violation(nodes["mul"]) is None


class _Flattened(nn.Module):
    """``flatten(0, 1)`` merges the batch with the tokens; ``squeeze()`` drops the batch axis at batch 1."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(6, 6)

    def forward(self, x):
        y = self.fc(x)
        if self.how == "flatten01":
            return y.flatten(0, 1)
        if self.how == "torch_flatten0":
            return torch.flatten(y)
        if self.how == "squeeze_all":
            return y[:, :1].squeeze() * 2
        if self.how == "unsqueeze0":
            return y.unsqueeze(0)
        raise ValueError(self.how)


@pytest.mark.parametrize("how,reason", [
    ("flatten01", "axis 0 is not the batch axis"), ("torch_flatten0", "axis 0 is not the batch axis"),
    ("squeeze_all", "the rank changes with the batch size"), ("unsqueeze0", "axis 0 is not the batch axis"),
])
def test_one_rule_refuses_every_layout_method_off_the_batch_axis(how, reason):
    with pytest.raises(BatchAxisLayoutError, match=reason):
        convert(native(lambda: _Flattened(how)), (3, 6), "spiking")


# ----------------------------------------------------------------------------- a host op reading a layout alias
class AliasedHostOperand(nn.Module):
    """The round-4 reviewer's ``w4_guard.py`` LSTM forms: ``y.unsqueeze(1)`` is carried as ``y``'s value (an alias of
    another shape), and the LSTM read it as a sequence of length B. Refused by name on the round-13 tree only because
    an LSTM guard left the program without a static proof (``h.dim()`` unknown); TypeError or wrong on w0-base."""

    FORMS = {
        "cat": lambda h: torch.cat([h, h], h.dim() - 1).flatten(1),
        "softmax": lambda h: F.softmax(h, dim=h.dim() - 1).flatten(1),
        "transpose": lambda h: h.transpose(h.dim() - 1, h.dim() - 2).flatten(1),
        "squeeze_lstm": lambda h: h.squeeze(1),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)
        self.lstm = nn.LSTM(6, 4, batch_first=True)

    def forward(self, x):
        y = self.fc(x)
        h = self.lstm(y.unsqueeze(1))[0]
        return self.FORMS[self.how](h)


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(AliasedHostOperand.FORMS))
def test_a_host_op_reads_a_layout_alias_as_its_native_value(how, contract):
    """[review round 14] A layout carried as an alias of another shape (``unsqueeze``, ``squeeze``, ``expand`` of an
    activation) is lowered as the host op of its native method before a host op reads it: the LSTM reads ``(B, 1, 6)``,
    and every form is exact at batches 1 to 13 (the ``cat`` form crashed in the converted model's forward)."""
    assert_converted_exact(native(lambda: AliasedHostOperand(how)), (8,), contract, batches=tuple(range(1, 14)),
                           kinds=("normal", "uniform"))
