"""A per-sample layout fixed at conversion must not be one a batch decision reaches (review round 6; round 7; restated
in review round 14).

A layout method (``view``, ``reshape``, ``flatten``, ``squeeze``, ``unsqueeze``, ``expand``, ``permute``,
``transpose``, ``torch.flatten``, ``nn.Flatten``) is lowered with the traced per-sample shape. When its size or axis
argument is computed from a run-time extent, the traced shape holds at the traced batch only. Such conversions were
certified and wrong at batch 13 (the probe runs batches 1 and 6), on w0-base too.

[restated in review round 14: the owner's shape-proof domain computes every batch-dependent size as an affine form
``a*B + c``. ``_a(y) = y[::7].shape[0]`` (``ceil(B / 7)``) and ``_g(x) = x.size(0) > 7`` are opaque values outside it
(the slice's length and the comparison, ``getitem`` and ``gt``), and every node that depends on one in a shape
position runs at run time: a layout an opaque value reaches is the host op of its native method on its run-time
operands, so each round-6 form is exact at every batch, unless a mapper that fixes a per-sample shape reads such a
value (refused by name). A host op reading an operand carried as a layout alias of another shape
(``torch.squeeze(y.unsqueeze(1), 2 - A)``: the alias handed it ``y``, refused by name in rounds 6 to 13) reads the
native value now: the alias is first lowered as the host op of its native method. The round-6 and round-7 guard
bookkeeping (a kernel's guard decided nothing, any guard left the program without
a static proof) is gone: no fold rests on a guard.]
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BATCH_SIZE, BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.refusals import BatchAxisLayoutError
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .harness import CONTRACTS, PACKAGINGS, assert_converted_exact, convert, native, quietly


def _a(y):
    """ceil(B / 7): 1 at batches 1 to 7, 2 at 8 to 14."""
    return y[::7].shape[0]


def _g(x):
    return x.size(0) > 7


class LayoutArgument(nn.Module):
    """A layout or reduction operator whose size or axis is computed from a run-time extent."""

    FORMS = {
        # value witnesses: same output shape as native, other values at batch 13
        "softmax_reshape": lambda x, y: F.softmax(y.reshape(-1, 6 // _a(y)), -1).reshape(y.shape[0], -1),
        "softmax_view": lambda x, y: F.softmax(y.view(-1, 6 // _a(y)), -1).view(y.size(0), 6),
        "softmax_view_gate": lambda x, y: F.softmax(y.view(-1, 6 // (1 + _g(x))), -1).view(y.size(0), 6),
        "cumsum_view": lambda x, y: torch.cumsum(y.view(-1, 6 // _a(y)), 1).view(y.size(0), 6),
        "softmax_flatten": lambda x, y: F.softmax(torch.flatten(y.view(y.size(0), 2, 3), _a(y)), -1).reshape(
            y.shape[0], -1),
        # layout-argument battery
        "flatten_fn": lambda x, y: torch.flatten(y.unsqueeze(1), 2 - _a(y)),
        "reshape": lambda x, y: y.reshape(-1, 6 // _a(y)),
        "flatten_fn_gate": lambda x, y: torch.flatten(y.unsqueeze(1), 1 + _g(x)),
        "flatten": lambda x, y: y.unsqueeze(1).flatten(2 - _a(y)),
        "squeeze_fn": lambda x, y: torch.squeeze(y.unsqueeze(1), 2 - _a(y)),
        "flatten_gate": lambda x, y: y.unsqueeze(1).flatten(1 + _g(x)),
        "view_axis": lambda x, y: y.view(y.size(0), _a(y), -1),
        "squeeze": lambda x, y: y.unsqueeze(1).squeeze(2 - _a(y)),
        "pool_flatten": lambda x, y: F.adaptive_avg_pool1d(y.unsqueeze(1), 6 // _a(y)).flatten(1),
        "view_gate": lambda x, y: y.view(-1, 6 // (1 + _g(x))),
        "view": lambda x, y: y.view(-1, 6 // _a(y)),
        "reshape_batch": lambda x, y: y.reshape(y.shape[0] * _a(y), -1),
        "movedim_flatten": lambda x, y: torch.movedim(y.unsqueeze(1), 1, 2 - _a(y)).flatten(1),
        "reshape_gate": lambda x, y: y.reshape(-1, 6 // (1 + _g(x))),
        # refused by the probe (unsqueeze, expand) or anonymously (transpose) before
        "expand": lambda x, y: y.unsqueeze(1).expand(-1, _a(y), -1),
        "unsqueeze": lambda x, y: y.unsqueeze(2 - _a(y)),
        "transpose": lambda x, y: y.transpose(1, 2 - _a(y)),
        # declared change: exact before at the batches where the native model runs (it raises from batch 8 on)
        "flatten_start": lambda x, y: y.flatten(_a(y) - 1 + 1),
        # a layout of a host op's value that such a guard decided: certified and crashing at batch 13 in the
        # ReshapeMapper before (chunk on w0-base too); the last three exact, w0-base refused them at its forward probe:
        # transpose_gate_flatten stays refused by name (a transpose is no pure reshape), and since round 8 a layout the
        # gate only scales (gate_view) or only squeezes (squeeze_gate_flatten) is exact (``test_shape_decisions.py``)
        "slice_gate_flatten": lambda x, y: y[:, : 6 // (1 + _g(x))].flatten(1),
        "slice_gate_reshape": lambda x, y: y[:, : 6 // (1 + _g(x))].reshape(y.size(0), -1),
        "chunk_flatten": lambda x, y: torch.chunk(y, _a(y), 1)[0].flatten(1),
        "pool_view_flatten": lambda x, y: F.adaptive_avg_pool1d(y.view(y.size(0), 1, 6), 6 // _a(y)).flatten(1),
        "transpose_gate_flatten": lambda x, y: torch.transpose(y.view(y.size(0), 2, 3), 1, 1 + _g(x)).flatten(1),
        "squeeze_gate_flatten": lambda x, y: torch.squeeze(y.view(y.size(0), 1, 6), 1 + _g(x)).flatten(1),
        "gate_view": lambda x, y: (y * (1 + _g(x))).view(y.size(0), 2, 3).flatten(1),
        # host ops: the native function at run time, exact
        "chunk": lambda x, y: torch.chunk(y, _a(y), 1)[0],
        "stride": lambda x, y: y[:, :: _a(y)],
        "mean_method": lambda x, y: y.mean(2 - _a(y)),
        "slice_gate": lambda x, y: y[:, : 6 // (1 + _g(x))],
        "cumsum": lambda x, y: torch.cumsum(y, dim=2 - _a(y)),
        "reshape_fn": lambda x, y: torch.reshape(y, (-1, 6 // _a(y))),
        "split": lambda x, y: torch.split(y, 6 // _a(y), 1)[0],
        "narrow": lambda x, y: torch.narrow(y, 1, 0, 6 // _a(y)),
        "mean_keepdim": lambda x, y: torch.mean(y, 2 - _a(y), keepdim=True),
        "mean_gate": lambda x, y: torch.mean(y, 1 - _g(x), keepdim=True),
        "roll": lambda x, y: torch.roll(y, shifts=_a(y) - 1, dims=1),
        "slice": lambda x, y: y[:, : 6 // _a(y)],
        "stack": lambda x, y: torch.stack([y, y], 2 - _a(y)),
        "pad": lambda x, y: F.pad(y, (0, _a(y) - 1)),
        "slice_ellipsis": lambda x, y: y[..., : 6 // _a(y)],
        "softmax_dim": lambda x, y: F.softmax(y, dim=-_a(y)),
        # controls: no guard, unchanged and byte-identical to w0-base
        "batch_view": lambda x, y: y.view(y.size(0), -1),
        "input_view": lambda x, y: y + x.view(x.size(0), -1)[:, :6],
        "view_flatten": lambda x, y: y.view(y.size(0), 2, 3).flatten(1),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](x, self.fc(x))


# [restated in review round 14: refused by name in rounds 6 to 13 (the host op named, and the opaque value): a host op
# reading an operand carried as a layout alias of another shape (``y.unsqueeze(1)`` handed on as ``y``); certified and
# wrong at batch 13 before round 6, on w0-base too (``pool_flatten`` refused by name there)] the alias is lowered as the
# host op of its native method before the host op reads it (``LayoutMixin._native_operands``), so each form is exact
ALIASED_HOST_OPERANDS = ["flatten_fn", "flatten_fn_gate", "flatten", "squeeze_fn", "flatten_gate", "squeeze",
                         "pool_flatten", "movedim_flatten", "expand"]
# [restated in review round 14: refused by name in rounds 6 to 13, naming the guard's origin] a layout an opaque value
# reaches, read by host ops or the output only: the host op of its native method, exact at every batch where the
# native model runs (``flatten_start``: up to batch 7; ``softmax_flatten``, ``softmax_dim``: up to batch 14)
NATIVE_LAYOUTS = ["softmax_reshape", "softmax_view", "softmax_view_gate", "cumsum_view", "softmax_flatten", "reshape",
                  "view_axis", "view_gate", "view", "reshape_batch", "reshape_gate", "unsqueeze", "transpose",
                  "flatten_start", "slice_gate_flatten", "slice_gate_reshape", "chunk_flatten", "pool_view_flatten",
                  "transpose_gate_flatten"]
NATIVE_BATCHES = {"flatten_start": tuple(range(1, 8)), "softmax_flatten": tuple(range(1, 15))}
# A gate that only scales the layout's input, or squeezes it by a pure reshape: refused by name in rounds 6 and 7 (the
# decision reached the layout), exact since round 8 (it reaches no per-sample shape the program reads).
RESTORED = ["gate_view", "squeeze_gate_flatten"]
HOST_OPS = ["chunk", "stride", "mean_method", "slice_gate", "cumsum", "reshape_fn", "split", "narrow",
            "mean_keepdim", "mean_gate", "roll", "slice", "stack", "pad", "slice_ellipsis", "softmax_dim"]
CONTROLS = ["batch_view", "input_view", "view_flatten"]
EVERY_BATCH = tuple(range(1, 15))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(ALIASED_HOST_OPERANDS))
def test_a_host_op_reading_a_layout_alias_an_opaque_value_reaches_reads_its_native_value(how, contract):
    """Certified and wrong at batch 13 before round 6 (on w0-base too; ``pool_flatten`` refused by name there). [restated
    in review round 14: refused by name in rounds 6 to 13, naming the opaque value; the alias is the host op of its
    native method now, so the host op reads ``(B, 1, 6)`` as natively: exact at every batch the native model runs at]"""
    flow = assert_converted_exact(native(lambda: LayoutArgument(how)), (8,), contract,
                                  batches=NATIVE_BATCHES.get(how, EVERY_BATCH), kinds=("normal", "uniform"))
    assert flow.shape_opaque is not None


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", NATIVE_LAYOUTS)
def test_a_layout_an_opaque_value_reaches_is_its_native_host_op_and_exact(how, contract):
    """Certified and wrong at batch 13 on the round-5 tree (w0-base too), refused by name in rounds 6 to 13: exact at
    every batch the native model runs at, with no static layout of the decided value and no literal."""
    flow = assert_converted_exact(native(lambda: LayoutArgument(how)), (8,), contract,
                                  batches=NATIVE_BATCHES.get(how, EVERY_BATCH), kinds=("normal", "uniform"))
    assert flow.shape_literals == {} and flow.shape_opaque is not None


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", RESTORED)
def test_a_layout_a_gate_only_scales_or_squeezes_is_exact_at_every_batch(how, contract):
    """Declared change (round 8): refused by name in rounds 6 and 7; refused at its forward probe on w0-base."""
    assert_converted_exact(native(lambda: LayoutArgument(how)), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", HOST_OPS)
def test_a_host_op_whose_guard_decides_its_shape_is_exact_at_every_batch(how, contract):
    """It runs the native function on the native operands at run time."""
    assert_converted_exact(native(lambda: LayoutArgument(how)), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


# The w0-base mapper structure of the controls (``out/structure_w0-base.txt``), type and name in execution order.
CONTROL_STRUCTURE = {
    "batch_view": ["InputMapper", "Ensure2DMapper", "fc", "ReshapeMapper"],
    "input_view": ["InputMapper", "Ensure2DMapper", "fc", "ReshapeMapper", "getitem", "add"],
    "view_flatten": ["InputMapper", "Ensure2DMapper", "fc", "ReshapeMapper", "ReshapeMapper"],
}


def _structure(flow) -> list:
    rows = []
    for mapper in flow.get_mapper_repr().execution_order():
        name = getattr(mapper, "name", None)
        rows.append(name if isinstance(mapper, ComputeOpMapper) else type(mapper).__name__)
    return ["fc" if row == "PerceptronMapper" else row for row in rows]


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", CONTROLS)
def test_a_layout_without_a_guard_is_unchanged(how, contract):
    flow = assert_converted_exact(native(lambda: LayoutArgument(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert _structure(flow) == CONTROL_STRUCTURE[how]


def test_the_declared_change_is_a_program_the_native_model_cannot_run_from_batch_8_on():
    """``y.flatten(A)``: start dim 1 up to batch 7, 2 (out of range for a 2-D value) from batch 8 on."""
    model = native(lambda: LayoutArgument("flatten_start"))
    with torch.no_grad():
        model(torch.zeros(7, 8, dtype=torch.float64))
        with pytest.raises(IndexError):
            model(torch.zeros(8, 8, dtype=torch.float64))


# ----------------------------------------------------------------------------- the opaque value's origin
def _layouts(factory):
    torch.manual_seed(0)
    gm = trace_model(factory().double().eval(), (8,))
    return BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}


class TestBatchDecision:
    """[restated in review round 14: the decisions were guard requests; they are the opaque values of the domain]"""

    def test_a_slice_length_outside_the_domain_is_the_layouts_decision(self):
        layouts, nodes = _layouts(lambda: LayoutArgument("view"))
        decision = layouts.batch_decision(nodes["view"])
        assert decision.node is nodes["getitem"] and decision.kind == "opaque" and decision.operator == "getitem"
        assert "floordiv by 7" in decision.detail and layouts.layout(nodes["view"]) is None
        assert layouts.layout(nodes["fc"]) == (BATCH_SIZE, 6)
        assert layouts.proof_statement().endswith("(first: getitem, FX node 'getitem')")

    def test_the_scalar_arithmetic_of_an_argument_can_be_the_decision(self):
        layouts, nodes = _layouts(lambda: LayoutArgument("view_gate"))
        decision = layouts.batch_decision(nodes["view"])
        assert decision.node is nodes["gt"] and decision.kind == "opaque"
        assert decision.detail.startswith("a value outside the affine batch domain (gt whose outcome changes")

    @pytest.mark.parametrize("how", CONTROLS)
    def test_no_opaque_value_no_decision_and_a_static_proof(self, how):
        layouts, nodes = _layouts(lambda: LayoutArgument(how))
        assert all(layouts.batch_decision(node) is None for node in nodes.values())
        assert layouts.opaque() is None and layouts.layout(nodes["fc"]) == (BATCH_SIZE, 6)
        assert layouts.proof_statement() == (
            "static shape proof: every batch-dependent size in the affine batch domain (a*B + c, B >= 1)")

    def test_a_value_after_a_decided_input_is_reached_by_the_decision(self):
        """``softmax_view``'s second view reads the decided first one: the decision reaches it through the data flow."""
        layouts, nodes = _layouts(lambda: LayoutArgument("softmax_view"))
        assert layouts.batch_decision(nodes["view"]).node is nodes["getitem"]
        assert layouts.batch_decision(nodes["view_1"]).node is nodes["getitem"]
        assert layouts.batch_decision(nodes["fc"]) is None

    def test_a_host_op_that_reads_an_opaque_value_decides_its_users(self):
        layouts, nodes = _layouts(lambda: LayoutArgument("chunk_flatten"))
        assert layouts.batch_decision(nodes["flatten"]).node is nodes["getitem"]


class UnrelatedGuard(nn.Module):
    """A layout of a value computed before an attention leaf's guard, in another branch."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 6)
        self.proj = nn.Linear(8, 8)
        self.attention = nn.MultiheadAttention(4, 1, batch_first=True)

    def forward(self, x):
        y = self.fc(x)
        tokens = self.proj(x).reshape(x.size(0), 2, 4)
        h = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        return y.view(y.size(0), 2, 3).flatten(1) + h.flatten(1)[:, :6]


def test_a_guard_in_another_branch_decides_nothing():
    """The attention leaf's own run requests ``Eq(B, 7)``; no fold rests on a guard now, so it decides nothing and
    leaves the static proof. [restated in review round 14: round 6 left the program without a static proof]"""
    layouts, nodes = _layouts(UnrelatedGuard)
    assert all(layouts.batch_decision(node) is None for node in nodes.values()) and layouts.opaque() is None
    assert layouts.layout(nodes["view"]) == (BATCH_SIZE, 2, 3) and layouts.layout(nodes["attention"]) is None
    assert layouts.proof_statement() == (
        "static shape proof: every batch-dependent size in the affine batch domain (a*B + c, B >= 1)")


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_guard_in_another_branch_leaves_the_conversion_exact(contract):
    assert_converted_exact(native(UnrelatedGuard), (8,), contract, batches=(1, 2, 3, 6, 7, 8),
                           kinds=("normal", "uniform"))


# ----------------------------------------------------------------------------- the pass does not depend on the dtype
class ConvChain(nn.Module):
    """A static shape chain after a convolution; an fp32 convolution's kernel heuristic guards ``B < 16``."""

    FORMS = {
        "scale": lambda h: h * h.shape[-1] ** -0.5,
        "layer_norm": lambda h: F.layer_norm(h, h.shape[-1:]),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.conv = nn.Conv2d(1, 2, 3)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.FORMS[self.how](self.conv(x).flatten(1)))


CONV_LITERALS = {
    "scale": {"getitem": ("getitem", "8"), "pow_1": ("pow", "0.3535533905932738")},
    "layer_norm": {"getitem": ("getitem", "torch.Size([8])")},
}


@pytest.mark.parametrize("how", sorted(CONV_LITERALS))
def test_an_fp32_model_folds_what_its_fp64_witness_folds(how):
    """Refused before by name (``WitnessProgramMismatchError``: no literal in the returned conversion, one in the
    witness); exact on w0-base. The fp32 conversion is certified and folds the fp64 literals."""
    torch.manual_seed(0)
    model = ConvChain(how).eval()
    flow = quietly(convert_torch_model, copy.deepcopy(model), (1, 4, 4), 4, packaging=PACKAGINGS["spiking"]).eval()
    assert flow.conversion_certificate.certified and flow.shape_literals == CONV_LITERALS[how]
    for batch in (1, 2, 5, 16, 17):
        x = torch.randn(batch, 1, 4, 4, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            assert torch.equal(flow(x), model(x)), batch


def test_the_symbolic_pass_of_an_fp32_convolution_proves_its_layout():
    torch.manual_seed(0)
    gm = trace_model(ConvChain("scale").eval(), (1, 4, 4))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    assert layouts.layout(nodes["conv"]) == (BATCH_SIZE, 2, 2, 2) and layouts.opaque() is None
