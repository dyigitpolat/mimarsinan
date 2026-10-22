"""Leaf-call inlining, review round 2 of the Dynamo capture (fix round 2 of the coverage fixes): the five forms the
review certified while wrong, each a regression witness here.

- An operand's batch forms were fitted to the samples at batches 1, 2 and 3. ``h[:8]`` (``min(B, 8)``) fits ``B``, and
  ``W[: B // 7 + 3]`` is 3 rows at every sampled batch; a body reading such an extent only in a decision left the
  operand dead after the splice, so the converter's batch domain never saw it (wrong from batch 9, and from batch 7).
  The operands' extents are now proved in that domain (``leaf_operands``; witnesses ``h[:8] * 1.0``, a copy, since
  ``h[:8]`` itself is a view of the other operand's storage, which keeps the call opaque by itself).
- The batch-1 comparison ran at B = 1 only; an operand of ``B - 1`` rows is 1 row at batch 2, where Dynamo's 0/1
  assumption on its symbol fails without a guard (``squeeze``), wrong at batch 2 (witness ``h[1:] * 1.0``).
- A dead in-place dunder call (``t.__imul__(2.0)``) was dropped as an unread value: its write was lost.
- ``inspect.signature`` follows ``__wrapped__``: a ``functools.wraps`` wrapper was captured with the wrapped
  function's defaults.
- One traced value given as two operands got two stand-ins, so ``a is b`` was False in the capture; now the operands
  that are one tensor natively share a stand-in, and distinct tensors of one storage keep the call opaque.

Every witness ends certified and exact at every batch and input scale here (the converted model and its identity
value program), or refused by name with its converted model exact; a gated form is compared at inputs of scale 1e5 too.
"""

from __future__ import annotations

import copy
import functools

import pytest
import torch
import torch.fx
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.fx_rewrites.body_capture import capture_body
from mimarsinan.torch_mapping.fx_rewrites.operand_samples import TensorSample
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import ConversionValueParityError, relative_error
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program
from operand_completeness.harness import quietly

W = 4
SHAPE = (W,)
BATCHES = (1, 2, 3, 6, 7, 9, 13)
SCALES = (1.0, 1e5)
EXACT = 1e-12


# ----------------------------------------------------------------------------- (1) operand extents from the domain
@torch.fx.wrap
def same_rows(a, b):
    if a.shape[0] == b.shape[0]:
        return a * 2.0
    return a * 1.0


@torch.fx.wrap
def scale_by_rows(a, b):
    return a * float(b.shape[0])


@torch.fx.wrap
def add_rows(a, b):
    return a + b.shape[0]


@torch.fx.wrap
def gelu_scaled(t, k):
    return torch.nn.functional.gelu(t) * k


# ----------------------------------------------------------------------------- (2) a symbol 0 or 1 at another batch
@torch.fx.wrap
def sq_other(a, b):
    return a * (2.0 if b.squeeze(0).ndim == 1 else 1.0)


@torch.fx.wrap
def sq_all_other(a, b):
    return a * (2.0 if b.squeeze().dim() == 1 else 1.0)


def _nested_inner(b):
    return 2.0 if b.squeeze(0).ndim == 1 else 1.0


@torch.fx.wrap
def nested_outer(a, b):
    return a * _nested_inner(b)


# ----------------------------------------------------------------------------- (3) in-place dunder writes
@torch.fx.wrap
def d_imul(t):
    t.__imul__(2.0)
    return t * 1.0


@torch.fx.wrap
def d_iadd(t):
    t.__iadd__(1.0)
    return t * 1.0


@torch.fx.wrap
def d_isub(t):
    t.__isub__(1.0)
    return t * 1.0


@torch.fx.wrap
def d_itruediv(t):
    t.__itruediv__(2.0)
    return t * 1.0


@torch.fx.wrap
def d_ipow(t):
    t.__ipow__(2.0)
    return t * 1.0


@torch.fx.wrap
def d_imul_used(t):
    u = t.__imul__(2.0)
    return u * 1.0


@torch.fx.wrap
def big_imul(t):
    t.__imul__(torch.where(t > 1e4, 2.0, 1.0))
    return t * 1.0


# ----------------------------------------------------------------------------- (4) functools.wraps
def capped(f):
    @functools.wraps(f)
    def wrapper(t, cap=1e4):
        return f(t, cap)
    return wrapper


@torch.fx.wrap
@capped
def clamp_cap(t, cap=1e6):
    return t.clamp(max=cap)


def doubling(f):
    @functools.wraps(f)
    def wrapper(t, k=2.0):
        return f(t, k)
    return wrapper


@torch.fx.wrap
@doubling
def scale_default(t, k=1.0):
    return t * k


def passing(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@torch.fx.wrap
@passing
def doubled(t):
    return t * 2.0


# ----------------------------------------------------------------------------- (5) one value, two operands
@torch.fx.wrap
def same_object(a, b):
    if a is b:
        return (a + b).clamp(max=1e4)
    return a + b


@torch.fx.wrap
def pair_sum(a, b):
    return a * 2.0 + b.sum(1, keepdim=True)


UP_TO_8 = (1, 2, 3, 6, 7, 8)
"""The batches at which a call given ``h[:8]``, opaque, converts exactly: from batch 9 its operand no longer has the
batch's rows, and the host op of an opaque call takes its operands at the native per-sample shapes (a boundary of
host-op operand layouts the base converter has too, not one of inlining)."""

CASES = {
    # case: (the forward after h = fc1(x), the helper whose calls are decided, the outcome, a text every reason holds,
    #        the batches compared)
    "minK_relation": (lambda net, h: same_rows(h, h[:8]), same_rows, "refused", "one storage", UP_TO_8),
    "minK_relation_copy": (lambda net, h: same_rows(h, h[:8] * 1.0), same_rows, "refused", "its operand 'b'",
                           UP_TO_8),
    "minK12_relation_copy": (lambda net, h: same_rows(h, h[:12] * 1.0), same_rows, "refused", "its operand 'b'",
                             (1, 2, 3, 6, 7, 9)),
    "minK_dataflow_copy": (lambda net, h: add_rows(h, h[:8] * 1.0), add_rows, "refused", "its operand 'b'", UP_TO_8),
    "static_on_samples": (lambda net, h: scale_by_rows(h, net.fc1.weight.repeat(4, 1)[: h.shape[0] // 7 + 3]),
                          scale_by_rows, "refused", "its operand 'b'", BATCHES),
    "tail_sq_dead": (lambda net, h: sq_other(h, h[1:]), sq_other, "refused", "one storage", BATCHES),
    "tail_sq_dead_copy": (lambda net, h: sq_other(h, h[1:] * 1.0), sq_other, "refused", "at batch 2", BATCHES),
    "tail_sq_all_dead_copy": (lambda net, h: sq_all_other(h, h[1:] * 1.0), sq_all_other, "refused", "at batch 2",
                              BATCHES),
    "nested_tail_squeeze_copy": (lambda net, h: nested_outer(h, h[1:] * 1.0), nested_outer, "refused", "at batch 2",
                                 BATCHES),
    "helper_dunder_imul": (lambda net, h: h + d_imul(h), d_imul, "refused", "writes in place", BATCHES),
    "helper_dunder_iadd": (lambda net, h: h + d_iadd(h), d_iadd, "refused", "writes in place", BATCHES),
    "helper_dunder_isub": (lambda net, h: h + d_isub(h), d_isub, "refused", "writes in place", BATCHES),
    "helper_dunder_itruediv": (lambda net, h: h + d_itruediv(h), d_itruediv, "refused", "writes in place", BATCHES),
    "helper_dunder_ipow": (lambda net, h: h + d_ipow(h), d_ipow, "refused", "writes in place", BATCHES),
    "helper_dunder_used": (lambda net, h: h + d_imul_used(h), d_imul_used, "refused", "writes in place", BATCHES),
    "big_imul": (lambda net, h: h + big_imul(h), big_imul, "refused", "writes in place", BATCHES),
    "wraps_cap": (lambda net, h: clamp_cap(h), clamp_cap, "certified", None, BATCHES),
    "wraps_default": (lambda net, h: scale_default(h), scale_default, "certified", None, BATCHES),
    "wraps_variadic": (lambda net, h: doubled(h), doubled, "refused", "variadic", BATCHES),
    "identity_two_operands": (lambda net, h: same_object(h, h), same_object, "certified", None, BATCHES),
    "identity_contiguous": (lambda net, h: same_object(h, h.contiguous()), same_object, "certified", None, BATCHES),
    "identity_distinct": (lambda net, h: same_object(h, h * 1.0), same_object, "certified", None, BATCHES),
    "views_of_one_storage": (lambda net, h: pair_sum(h, h[:, :2]), pair_sum, "refused", "one storage", BATCHES),
}


class Net(nn.Module):
    def __init__(self, case: str):
        super().__init__()
        self.case = case
        self.fc1, self.act, self.fc2 = nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(self.act(CASES[self.case][0](self, h)))


class TwoCalls(Net):
    """A call whose operand the batch domain does not prove (``min(B, 8)`` rows) beside one it does: only the first
    stays."""

    def __init__(self):
        super().__init__("minK_relation_copy")

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(self.act(same_rows(h, h[:8] * 1.0) + gelu_scaled(h, 0.5)))


class Chained(Net):
    """The second call's operand is computed from the first, which stays opaque: the second stays too."""

    def __init__(self):
        super().__init__("minK_relation_copy")

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(self.act(gelu_scaled(same_rows(h, h[:8] * 1.0), 0.5)))


def native(factory, *args) -> nn.Module:
    torch.manual_seed(0)
    return factory(*args).double().eval()


def x_at(batch: int, scale: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(9100 + 37 * batch)
    return scale * torch.randn(batch, *SHAPE, generator=g, dtype=torch.float64)


def _worst(candidate, model, batches, flat: bool = False) -> float:
    worst = 0.0
    for scale in SCALES:
        for batch in batches:
            x = x_at(batch, scale)
            with torch.no_grad():
                want = model(x)
                got = candidate(x)
            got = got.reshape(want.shape) if flat else got
            assert got.shape == want.shape, (batch, scale, tuple(got.shape), tuple(want.shape))
            worst = max(worst, relative_error(got, want).relative_error)
    return worst


def _named(exc: BaseException) -> bool:
    if not isinstance(exc, ToolchainRefusal):
        return False
    if isinstance(exc, ConversionValueParityError):
        return isinstance(getattr(exc.result, "error", None), ToolchainRefusal)
    return True


def outcome(model: nn.Module, batches=BATCHES) -> str:
    """``certified`` (the flow and its identity value program exact at every batch and scale) or ``refused`` (by name,
    its converted model exact wherever it converts); fails on anything else."""
    try:
        flow = quietly(convert_torch_model, copy.deepcopy(model), SHAPE, 3, packaging=MVM_PACKAGING,
                       value_program_stage=True)
    except ToolchainRefusal as exc:
        assert _named(exc), f"refused, but not by name: {exc}"
        try:
            stage_1 = quietly(convert_torch_model, copy.deepcopy(model), SHAPE, 3, packaging=MVM_PACKAGING).eval()
        except ToolchainRefusal as first:
            assert _named(first), f"stage 1 refused, but not by name: {first}"
            return "refused"
        worst = _worst(stage_1, model, batches)
        assert worst <= EXACT, f"refused at stage 2, and its converted model is wrong ({worst:.3e})"
        return "refused"
    assert flow.conversion_certificate.certified, flow.conversion_certificate
    flow.eval()
    worst = _worst(flow, model, batches)
    assert worst <= EXACT, f"CERTIFIED AND WRONG: converted model {worst:.3e}"
    worst = _worst(identity_value_program(flow), model, batches, flat=True)
    assert worst <= EXACT, f"CERTIFIED AND WRONG: identity value program {worst:.3e}"
    return "certified"


def _reasons(model: nn.Module, helper):
    gm = trace_model(copy.deepcopy(model), SHAPE)
    return [n.meta.get(OPAQUE_BODY) for n in gm.graph.nodes if n.op == "call_function" and n.target is helper]


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_review_r2_witness_is_exact_or_refused_by_name(case):
    assert outcome(native(Net, case), CASES[case][4]) == CASES[case][2]


@pytest.mark.parametrize("case", sorted(c for c in CASES if CASES[c][2] == "refused"))
def test_a_review_r2_refusal_keeps_the_call_opaque_and_names_why(case):
    reasons = _reasons(native(Net, case), CASES[case][1])
    assert reasons and all(r and CASES[case][3] in r for r in reasons), reasons


@pytest.mark.parametrize("case", sorted(c for c in CASES if CASES[c][2] == "certified"))
def test_a_review_r2_certified_witness_is_inlined(case):
    assert _reasons(native(Net, case), CASES[case][1]) == []


def test_an_unproved_operand_keeps_only_its_own_call_opaque():
    model = native(TwoCalls)
    assert [bool(r) for r in _reasons(model, same_rows)] == [True]
    assert _reasons(model, gelu_scaled) == []
    assert outcome(model, UP_TO_8) == "refused"


def test_a_call_computed_from_an_unproved_call_stays_opaque_too():
    model = native(Chained)
    assert [bool(r) for r in _reasons(model, same_rows)] == [True]
    reasons = _reasons(model, gelu_scaled)
    assert len(reasons) == 1 and reasons[0] and "computed from" in reasons[0], reasons
    assert outcome(model, UP_TO_8) == "refused"


def _rows(rows_at, width=W):
    return tuple(TensorSample.of(torch.empty(rows_at(b), width, dtype=torch.float64)) for b in (1, 2, 3))


def test_a_capture_compares_every_batch_at_which_a_symbol_is_0_or_1():
    """``b`` has B - 1 rows: its symbol is 0 at batch 1 and 1 at batch 2, so a static capture runs at both."""

    def body(a, b):
        return a * (2.0 if b.squeeze(0).ndim == 1 else 1.0)

    found = capture_body(body, {"a": None, "b": None}, {"a": _rows(lambda b: b), "b": _rows(lambda b: b - 1)})
    assert isinstance(found, str) and "at batch 2" in found, found

    def benign(a, b):
        return a * 3.0 + b.sum()

    found = capture_body(benign, {"a": None, "b": None}, {"a": _rows(lambda b: b), "b": _rows(lambda b: b - 1)})
    assert not isinstance(found, str), found


def test_a_wrapper_is_bound_by_its_own_signature():
    """``inspect.signature`` follows ``__wrapped__``: the wrapper's own default (1e4) is the one natively used."""
    gm = trace_model(native(Net, "wraps_cap"), SHAPE)
    clamps = [n for n in gm.graph.nodes if n.op == "call_method" and n.target == "clamp"]
    assert len(clamps) == 1 and clamps[0].kwargs.get("max") == 1e4, gm.graph
