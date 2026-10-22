"""Every traced value has a symbolic value; a per-sample shape the pass cannot prove is never fixed (review round 8).

The symbolic-batch pass does not evaluate an opaque call, nor a declared call whose symbolic run raises, and a kernel
can make a batch extent concrete (``F.pad`` and a batched ``matmul`` give 7, the hint). Before, every node downstream
of such a value was skipped or saw the concrete 7, so a gate computed from it recorded no guard and no decision:
``g = 1 + (ident(x).size(0) > 7)`` through an identity helper, a helper's 0-dim count, a gate on ``F.pad(y, (0,
2)).size(0)``. Seven such programs were certified and wrong at batches 9 and 13 by rel 0.09 to 1.0 (on w0-base too).
Now such a value is taken from the seeded traces at batches 1, 2 and 3 (B on each batch axis, the traced extent on each
static one; a concrete batch axis gets B back), it and everything computed from it are unproved (never folded), and
every declared node downstream runs, so its guard requests and decisions are recorded. An opaque call returning a
tensor without a batch axis (a 0-dim count) from a run-time value is a batch decision.

The per-sample shape of an opaque call's tensor result is the traced one at the traced batches only. Before, a layout
of it was lowered with that shape: ``regroup(y)``, which gives ``(B, 2, 3)`` up to batch 8 and ``(B, 3, 2)`` above,
then ``reshape(h.size(0), h.size(2), h.size(1))``, was certified and wrong by rel 0.35 to 1.0 at batches 9 and 13 (on
w0-base too), under a certificate hypothesis. Now such a layout fixes no shape: it is the host op of its native method
on its run-time operands, exact at every batch, and a value program holding such a call is refused by name at stage 2
(the IR fixes every per-sample shape). The hypothesis is gone.

[restated in review round 9: a program with a call outside the declared set is impure (``symbolic/purity.py``): the
pass never runs for it, nothing is folded, and every layout is the host op of its native method. The gates read through
an identity helper or a helper's count, refused by name in round 8, are exact at every batch; the value programs are
refused by name as impure (``ImpureShapeProgramError``, naming the first impure node). The trace-taken values of the
pass remain for pure programs: a declared kernel that makes a batch extent concrete (``F.pad``) and a declared call
whose symbolic run raises.]
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mapping_utils import ComputeOpMapper
from mimarsinan.torch_mapping.converter_handlers.batch_layout import BatchLayouts
from mimarsinan.torch_mapping.converter_handlers.symbolic.batch_domain import BATCH
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity.certificate import CERTIFICATE_HYPOTHESES

from .harness import (
    CONTRACTS,
    assert_converted_exact,
    assert_impure_exact,
    assert_impure_stage_2_refused,
    convert,
    keep_opaque,
    native,
)

EVERY_BATCH = (1, 2, 3, 6, 7, 9, 13)


@torch.fx.wrap
def ident(t):
    """An identity helper: the same tensor, the same per-sample shape at every batch."""
    keep_opaque(t)  # a dataflow body: inlined otherwise (leaf-call inlining), and these witnesses are opaque helpers
    return t


@torch.fx.wrap
def pick(n):
    return 1 if n <= 7 else 2


@torch.fx.wrap
def count(t):
    """A scalar of a run-time value: 1 for a batch up to 7, else 2."""
    return 1 if t.shape[0] <= 7 else 2


@torch.fx.wrap
def count_t(t):
    """The same count as a 0-dim tensor."""
    return torch.tensor(1 if t.shape[0] <= 7 else 2)


@torch.fx.wrap
def regroup(t):
    """(B, 6) -> (B, 2, 3) up to batch 8, (B, 3, 2) above: a per-sample shape that follows the batch."""
    return t.view(t.shape[0], 2, 3) if t.shape[0] <= 8 else t.view(t.shape[0], 3, 2)


@torch.fx.wrap
def swish(t):
    keep_opaque(t)  # a dataflow body: inlined otherwise (leaf-call inlining), and these witnesses are opaque results
    return t * torch.sigmoid(t)


class Gated(nn.Module):
    """The reviewer's round-7 witnesses (``r7_mods.py``) and their controls."""

    FORMS = {
        "ident_gate_view": lambda m, x, y: F.softmax(y.view(x.size(0), 1 + (ident(x).size(0) > 7), -1), -1).flatten(1),
        "ident_pick_view": lambda m, x, y: F.softmax(y.view(x.size(0), pick(ident(x).size(0)), -1), -1).flatten(1),
        "ident_count_view": lambda m, x, y: F.softmax(y.view(x.size(0), count(ident(y)), -1), -1).flatten(1),
        "ident_gate_reshape": lambda m, x, y: _gate_reshape(ident(y)),
        "zero_dim_count_view": lambda m, x, y: F.softmax(y.view(x.size(0), count_t(x), -1), -1).flatten(1),
        "pad_gate_view": lambda m, x, y: F.softmax(
            y.view(x.size(0), 1 + (F.pad(y, (0, 2)).size(0) > 7), -1), -1).flatten(1),
        "direct_gate_view": lambda m, x, y: F.softmax(y.view(x.size(0), 1 + (x.size(0) > 7), -1), -1).flatten(1),
        "lstm_gate_view": lambda m, x, y: F.softmax(
            y.view(x.size(0), 1 + (m.lstm(y.unsqueeze(1))[0].size(0) > 7), -1), -1).flatten(1),
        "ident_gate_fold": lambda m, x, y: y * y.shape[1] + y * (ident(x).size(0) > 7),
        "direct_gate_fold": lambda m, x, y: y * y.shape[1] + y * (x.size(0) > 7),
        "swish_view_size": lambda m, x, y: swish(y).view(swish(y).size(0), -1),
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)
        self.lstm = nn.LSTM(6, 6, batch_first=True)

    def forward(self, x):
        return self.FORMS[self.how](self, x, self.fc(x))


def _gate_reshape(h):
    g = 1 + (h.size(0) > 7)
    return F.softmax(h.reshape(h.size(0), g, -1), -1).reshape(h.size(0), -1)


class GatedConv(nn.Module):
    """``IdentGateConv``: a convolution stem, the gate read through the identity helper of the input."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x):
        y = self.conv(x).flatten(1)
        g = 1 + (ident(x).size(0) > 7)
        return F.softmax(y.view(x.size(0), 2 * g, -1), -1).flatten(1)


# [restated in review round 14: refused by name in rounds 8 to 13, naming the guard ``add`` installs] a gate on the
# batch is an opaque value (``gt``), so the view it reaches is the host op of its native method: exact at every batch
GATED = ["pad_gate_view", "direct_gate_view", "lstm_gate_view"]
# [restated in review round 9] form -> the first impure node: refused by name in round 8 (naming ``add``, ``pick``,
# ``count`` or ``count_t``), exact at every batch now
IMPURE_GATED = {"ident_gate_view": "ident", "ident_pick_view": "ident", "ident_count_view": "ident",
                "ident_gate_reshape": "ident", "zero_dim_count_view": "count_t"}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", GATED)
def test_a_gate_on_the_batch_is_an_opaque_value_and_its_view_runs_on_the_host(how, contract):
    """Certified and wrong at batches 9 and 13 before round 8, by rel 0.38 to 1.0 (on w0-base too)."""
    flow = assert_converted_exact(native(lambda: Gated(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque[:2] == ("gt", "gt") and flow.shape_literals == {}


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(IMPURE_GATED))
def test_a_decision_downstream_of_a_call_outside_the_declared_set_is_exact(how, contract):
    """Certified and wrong at batches 9 and 13 by rel 0.38 to 1.0 in round 7 (on w0-base too), refused by name in
    round 8."""
    model = native(lambda: Gated(how))
    assert_impure_exact(model, (8,), contract, IMPURE_GATED[how], IMPURE_GATED[how])
    if contract == "mvm":
        assert_impure_stage_2_refused(model, (8,), IMPURE_GATED[how], IMPURE_GATED[how])


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_gate_through_an_identity_helper_of_a_conv_input_is_exact(contract):
    """``IdentGateConv``: certified and wrong at batches 9 and 13 by rel 0.09 to 1.0 in round 7 (on w0-base too),
    refused by name in round 8. [restated in review round 9: impure, exact at every batch]"""
    assert_impure_exact(native(GatedConv), (1, 4, 4), contract, "ident", "ident")


class SequenceFirstGate(nn.Module):
    """Round-7 open issue 4: a gate on the batch extent of a sequence-first LSTM's output (its kernel makes that
    extent the concrete 7): certified and wrong at batches 9 and 13 by rel 0.41 to 0.63 on the round-7 tree and on
    w0-base (``fix_r8/out/seqfirst_lstm_*.txt``)."""

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)
        self.lstm = nn.LSTM(6, 6)

    def forward(self, x):
        y = self.fc(x)
        h = self.lstm(y.view(x.size(0), 1, 6))[0]  # sequence-first: the batch extent is the sequence length
        gated = F.softmax(y.view(x.size(0), 1 + (h.size(0) > 7), -1), -1).flatten(1)
        return gated if self.how == "view" else h.flatten(1) + gated


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", ["view", "sum"])
def test_a_gate_on_the_batch_extent_of_a_sequence_first_lstm_is_exact(how, contract):
    """Refused by name in rounds 8 to 13. [restated in review round 14: the LSTM's rule states its output's leading
    axes, so ``h.size(0)`` is B and the gate on it an opaque value; the view runs on the host, exact]"""
    flow = assert_converted_exact(native(lambda: SequenceFirstGate(how)), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert flow.shape_opaque[:2] == ("gt", "gt")


def _pass(how: str):
    gm = trace_model(native(lambda: Gated(how)), (8,))
    layouts = BatchLayouts(gm)
    return layouts, layouts._pass(), {n.name: n for n in gm.graph.nodes}


def test_an_opaque_tensor_result_leaves_the_program_without_a_pass():
    """Round 8: the identity helper's value was the traced one, unproved, and the gate's decision was recorded.
    [restated in review round 9: the program is impure, so the pass never runs and no value has a layout]"""
    layouts, proof, nodes = _pass("ident_gate_view")
    assert layouts.impurity.node == "ident" and proof.batch is None and proof.shapes == {}
    assert layouts.layout(nodes["ident"]) is None and layouts.layout(nodes["fc"]) is None


def test_a_batch_axis_a_kernel_made_concrete_is_the_rules():
    """``F.pad`` gives (7, 8) in torch's run. [restated in review round 14: rounds 8 to 13 took the batch axis from
    the traces, unproved; the padding rule states it (``LEADING``: the leading axis passes through), and torch's 7 is
    the batch at the hint, so the cross-check agrees]"""
    layouts, proof, nodes = _pass("pad_gate_view")
    assert proof.shapes[nodes["pad"]].dims == (BATCH, 8) and nodes["pad"] not in proof.unproved
    assert layouts.purity.pure and proof.batch_decision(nodes["view"]).node is nodes["gt"]


def test_an_opaque_zero_dim_count_of_a_run_time_value_makes_the_program_impure():
    """Round 8: an ``opaque`` batch decision. [restated in review round 9: the helper makes the program impure]"""
    layouts, proof, _ = _pass("zero_dim_count_view")
    assert layouts.impurity.node == "count_t" and proof.batch is None


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_value_gate_through_an_identity_helper_is_exact_without_a_fold(contract):
    """``IdentGateFoldElsewhere``: the gate scales a value and requests no guard on either form; round 8 folded
    ``y.shape[1]`` in both under the same static proof. [restated in review round 9, declared change: the identity
    form is impure, so ``y.shape[1]`` is its run-time host op; the direct form folds as before; both exact]"""
    ident = assert_impure_exact(native(lambda: Gated("ident_gate_fold")), (8,), contract, "ident", "ident")
    direct = assert_converted_exact(native(lambda: Gated("direct_gate_fold")), (8,), contract, batches=EVERY_BATCH,
                                    kinds=("normal", "uniform"))
    assert ident.shape_literals == {} and direct.shape_literals == {"getitem": ("getitem", "6")}
    assert direct.shape_proof == ("static shape proof: every batch-dependent size in the affine batch domain (a*B + c, "
                                  "B >= 1); opaque batch values computed on the host at run time (first: gt, FX node "
                                  "'gt')")  # [restated in review round 14: the gate is an opaque value]


# ----------------------------------------------------------------------------- layouts of an opaque result
class Regrouped(nn.Module):
    FORMS = {
        "swap_reshape": lambda m, y: F.softmax(_swap(regroup(y)), -1).flatten(1),
        "flatten": lambda m, y: F.softmax(regroup(y), -1).flatten(1),
        "size_argument": lambda m, y: F.softmax(y.view(y.size(0), regroup(y).size(1), -1), -1).flatten(1),
        "unsqueeze": lambda m, y: F.softmax(regroup(y).unsqueeze(1), -1).flatten(1),
        "permute": lambda m, y: regroup(y).permute(0, 2, 1).flatten(1),
        "expand": lambda m, y: regroup(y).unsqueeze(1).expand(-1, 2, -1, -1).flatten(1),
        "concatenation": lambda m, y: torch.cat([regroup(y), regroup(y)], 2).flatten(1),
        "swish_flatten": lambda m, y: swish(y.view(y.size(0), 2, 3)).flatten(1),
        "swish_scale": lambda m, y: swish(y) * y.shape[1],
    }

    def __init__(self, how: str):
        super().__init__()
        self.how = how
        self.fc = nn.Linear(8, 6)

    def forward(self, x):
        return self.FORMS[self.how](self, self.fc(x))


def _swap(h):
    return h.reshape(h.size(0), h.size(2), h.size(1))


@pytest.mark.parametrize("contract", CONTRACTS)
@pytest.mark.parametrize("how", sorted(Regrouped.FORMS))
def test_a_layout_of_an_opaque_result_fixes_no_shape_and_is_exact_at_every_batch(how, contract):
    """``swap_reshape`` (``RegroupSwapReshape``) and ``size_argument`` were certified and wrong at batches 9 and 13
    by rel 0.35 to 1.0 before (on w0-base too); ``expand`` was refused anonymously at the probe (its
    ``ReshapeMapper``); the others were exact and stay exact."""
    assert_converted_exact(native(lambda: Regrouped(how)), (8,), contract, batches=EVERY_BATCH,
                           kinds=("normal", "uniform"))


def _structure(flow) -> list:
    return [m.name if isinstance(m, ComputeOpMapper) else type(m).__name__
            for m in flow.get_mapper_repr().execution_order()]


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_layout_of_an_opaque_result_is_its_native_host_op(contract):
    """Declared change: the layout is the host op of its native method (``reshape`` with its run-time ``size``
    operands), not a ``ReshapeMapper`` of the traced per-sample shape. [restated in review round 9: stage 2 refuses
    the impure program as a whole (``flow.shape_impurity``), where round 8 marked the host ops ``assumed_shape``]"""
    flow = convert(native(lambda: Regrouped("swap_reshape")), (8,), contract)
    names = _structure(flow)
    assert ["regroup", "size", "size_1", "size_2", "reshape", "softmax", "flatten"] == [
        name for name in names if name in {"regroup", "size", "size_1", "size_2", "reshape", "softmax", "flatten"}]
    assert "ReshapeMapper" not in names and flow.shape_impurity.node == "regroup"


@pytest.mark.parametrize("contract", CONTRACTS)
def test_a_view_of_an_opaque_result_by_its_batch_extent_is_exact(contract):
    """``swish(y).view(swish(y).size(0), -1)``: exact on w0-base with a ``ReshapeMapper``; the view is now the native
    host op on the run-time size (declared change, identical values)."""
    flow = assert_converted_exact(native(lambda: Gated("swish_view_size")), (8,), contract, batches=EVERY_BATCH,
                                  kinds=("normal", "uniform"))
    assert _structure(flow)[-1] == "view"


@pytest.mark.parametrize("how", ["swap_reshape", "flatten"])
def test_a_value_program_holding_an_opaque_tensor_result_is_refused_by_name_at_stage_2(how):
    """``RegroupFlatten``: its value program was wrong at batches 9 and 13 while stage 2 passed. [restated in review
    round 9: ``ImpureShapeProgramError`` naming ``regroup``; round 8: ``AssumedHostShapeError``]"""
    assert_impure_stage_2_refused(native(lambda: Regrouped(how)), (8,), "regroup", "regroup")


def test_the_certificate_holds_under_trace_fidelity_and_the_trust_base():
    """[restated in review round 12: trace fidelity alone before; the owner's trust base adds T1 and T2 beside it, and
    no hypothesis of per-sample shapes returns] [extended in review round 16: the owner's T3]"""
    assert [h.split(":")[0] for h in CERTIFICATE_HYPOTHESES] == [
        "trace fidelity", "T1 shipped runtime", "T2 nn.Module call protocol", "T3 opaque-helper effects"]


def test_a_shape_query_of_an_opaque_result_is_never_folded():
    """[restated in review round 9: no value of the impure program has a layout, ``fc`` included; round 8 proved
    ``fc``'s ``(B, 6)``]"""
    gm = trace_model(native(lambda: Regrouped("swap_reshape")), (8,))
    layouts, nodes = BatchLayouts(gm), {n.name: n for n in gm.graph.nodes}
    assert all(layouts.layout(nodes[name]) is None for name in ("fc", "regroup", "reshape", "softmax", "flatten"))
