"""Leaf-call inlining from a Dynamo capture of the helper's body (fix round 2 of the coverage fixes): the regression
witnesses of review round 1 and of the Dynamo scout.

Round 1 traced a helper's body with FX on Proxies. A Python decision FX makes without recording a node (``isinstance``,
``type``, ``torch.is_tensor`` or ``hasattr`` of a Proxy, ``is_fx_tracing()``, an exception FX raises before recording,
``str`` or ``repr`` of a tensor, ``dtype is``) selected an arm at trace time; the arm a real tensor takes was never
traced. Thirteen helpers that double their operand from batch 9 on were certified at both stages and wrong at batches 9
and 13; ``ndim`` read with ``isinstance`` was a value mismatch of 0.477 where the base named a refusal; and the trial
trace stored a Proxy in a helper's module cache, so the native model raised ``NameError`` afterwards.

The capture never runs the helper: Dynamo interprets its bytecode on stand-in operands, every batch-carrying axis a
symbol, and the call is inlined only when the capture is one graph, no guard on the batch fails at some batch B >= 1, a
second capture at batch 1 is the same program, the body writes nothing in place, and every node is a declared
operator. Each witness here ends certified and exact at batches 1, 2, 6, 9 and 13 (the converted model and its
identity value program), or refused by name with its converted model exact at those batches; none is certified while
wrong, and none is a value mismatch.
"""

from __future__ import annotations

import collections
import copy

import pytest
import torch
import torch.fx
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401  (helpers below)
from torch.fx._symbolic_trace import is_fx_tracing
from torch.overrides import handle_torch_function, has_torch_function
from torchvision.ops import StochasticDepth
from torchvision.ops.stochastic_depth import stochastic_depth

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from mimarsinan.torch_mapping.value_parity import ConversionValueParityError, relative_error
from mimarsinan.torch_mapping.value_parity.value_program import identity_value_program
from operand_completeness.harness import quietly

W = 4
SHAPE = (W,)
BATCHES = (1, 2, 6, 9, 13)
EXACT = 1e-12


# ----------------------------------------------------------------------------- the 13 certified-wrong forms
@torch.fx.wrap
def arm_isinstance(t):
    if isinstance(t, torch.Tensor):
        k = 2.0 if t.shape[0] > 8 else 1.0
    else:
        k = 1.0
    return t * k


@torch.fx.wrap
def arm_is_tensor(t):
    k = (2.0 if t.size(0) > 8 else 1.0) if torch.is_tensor(t) else 1.0
    return t * k


@torch.fx.wrap
def arm_type_is(t):
    k = (2.0 if t.shape[0] > 8 else 1.0) if type(t) is torch.Tensor else 1.0
    return t * k


@torch.fx.wrap
def arm_hasattr(t):
    if hasattr(t, "_no_such_attribute_"):
        k = 1.0
    else:
        k = 2.0 if t.shape[0] > 8 else 1.0
    return t * k


@torch.fx.wrap
def arm_is_fx_tracing(t):
    if is_fx_tracing():
        return t * 1.0
    return t * (2.0 if t.shape[0] > 8 else 1.0)


@torch.fx.wrap
def try_len(t):
    try:
        k = 2.0 if len(t) > 8 else 1.0
    except Exception:  # noqa: BLE001
        k = 1.0
    return t * k


@torch.fx.wrap
def str_rows(t):
    return t * (2.0 if str(t).count("\n") >= 8 else 1.0)


@torch.fx.wrap
def repr_len(t):
    return t * (2.0 if len(repr(t)) > 400 else 1.0)


@torch.fx.wrap
def try_iter(t):
    try:
        rows = sum(1 for _ in t)
    except Exception:  # noqa: BLE001
        rows = 1
    return t * (2.0 if rows > 8 else 1.0)


@torch.fx.wrap
def dtype_is_arm(t):
    if t.dtype is torch.float64:
        return t * (2.0 if t.shape[0] > 8 else 1.0)
    return t * 1.0


@torch.fx.wrap
def dict_key_arm(t, table):
    return t * table[isinstance(t, torch.Tensor) and t.shape[0] > 8]


@torch.fx.wrap
def kw_scale(t, *, scale=1.0, big=2.0):
    return t * (big if isinstance(t, torch.Tensor) and t.shape[0] > 8 else scale)


@torch.fx.wrap
def inner_arm(t):
    return t * (2.0 if isinstance(t, torch.Tensor) and t.shape[0] > 8 else 1.0)


@torch.fx.wrap
def outer_calls_inner(t):
    return inner_arm(t) + 0.0


# ----------------------------------------------------------------------------- ndim and the module cache
@torch.fx.wrap
def ndim_isinstance(t):
    d = t.ndim
    return t * (2.0 if isinstance(d, int) else 1.0)


_PE_CACHE: dict = {}
_PE_ROWS_CACHE: dict = {}


@torch.fx.wrap
def add_pe(t, dim):
    """A lazily built table, cached per literal width, in the operand's dtype and device."""
    if dim not in _PE_CACHE:
        _PE_CACHE[dim] = torch.arange(dim, dtype=t.dtype, device=t.device) * 0.1
    return t + _PE_CACHE[dim]


@torch.fx.wrap
def add_pe_rows(t, dim):
    """The same cache, the body also reading the batch extent (dataflow on it)."""
    if dim not in _PE_ROWS_CACHE:
        _PE_ROWS_CACHE[dim] = torch.arange(dim, dtype=t.dtype, device=t.device) * 0.1
    n = t.shape[0]
    return t.reshape(n, dim) + _PE_ROWS_CACHE[dim]


# ----------------------------------------------------------------------------- batch-1 forms (0/1 specialization)
@torch.fx.wrap
def eq_one(t):
    return t * (2.0 if t.shape[0] == 1 else 1.0)


@torch.fx.wrap
def ne_one_len(t):
    return t * (1.0 if len(t) != 1 else 2.0)


@torch.fx.wrap
def squeeze_rank(t):
    return t * (2.0 if t.squeeze(0).ndim == 1 else 1.0)


@torch.fx.wrap
def squeeze_all_rank(t):
    return t * (2.0 if t.squeeze().dim() == 1 else 1.0)


@torch.fx.wrap
def transpose_contiguous(t):
    return t * (2.0 if t.t().is_contiguous() else 1.0)


# ----------------------------------------------------------------------------- benign forms
@torch.fx.wrap
def eq_zero(t):
    return t * (2.0 if t.size(0) == 0 else 1.0)


@torch.fx.wrap
def reshape_rows(t):
    n = t.shape[0]
    return t.reshape(n, 2, W // 2).flatten(1) * 3.0


@torch.fx.wrap
def dim_branch(t):
    return t * (2.0 if t.dim() == 2 else 1.0)


@torch.fx.wrap
def dtype_eq_branch(t):
    return t * (2.0 if t.dtype == torch.float64 else 1.0)


@torch.fx.wrap
def device_str(t):
    return t * (2.0 if "cpu" in str(t.device) else 1.0)


@torch.fx.wrap
def masked(t):
    return t.masked_fill(t == 0, 1.0) * (t != 0)


@torch.fx.wrap
def no_grad_body(t):
    with torch.no_grad():
        y = t * 2.0
    return y + 1.0


@torch.fx.wrap
def returns_none(t):
    t.mul(2.0)
    return None


@torch.fx.wrap
def stride_gate(c, t):
    """Doubles ``t`` from batch 3 on: ``c`` keeps its extent (4,) at every batch, its stride is the batch."""
    return t * (2.0 if c.stride(0) > 2 else 1.0)


# ----------------------------------------------------------------------------- container outputs
Pair = collections.namedtuple("Pair", "a b")


@torch.fx.wrap
def nested_out(t, ks):
    return (t * ks[0], [t + ks[1], {"z": t - ks[0]}])


@torch.fx.wrap
def literal_in_out(t, ks):
    return t * ks[0], ks[1]


@torch.fx.wrap
def empty_list(t, ks):
    return t * (1.0 + len(ks))


@torch.fx.wrap
def none_and_list(t, mask, ks):
    return t * ks[0] if mask is None else t


@torch.fx.wrap
def int_keyed(t, cfg):
    return {0: t * cfg[0], 1: t + cfg[1]}


@torch.fx.wrap
def single_list_out(t, ks):
    return [t * ks[0]]


@torch.fx.wrap
def operand_out(t, ks):
    """An operand returned as it is, beside a computed value."""
    return t, t * ks[0]


@torch.fx.wrap
def nt_out(t, ks):
    return Pair(t * ks[0], t + ks[1])


@torch.fx.wrap
def shape_out(t):
    """The operand's shape returned (the batch extent in it)."""
    return t.shape


# ----------------------------------------------------------------------------- nested and self-dispatching
@torch.fx.wrap
def inner_shape(t):
    n = t.shape[0]
    return t * (2.0 if isinstance(n, int) and n > 8 else 1.0)


@torch.fx.wrap
def outer_calls_opaque(t):
    return inner_shape(t) * 1.0


@torch.fx.wrap
def scaled(t, factor):
    return t * factor


@torch.fx.wrap
def scaled_twice(t, factor, twice):
    y = scaled(t, factor)
    return scaled(y, factor) if twice else y


@torch.fx.wrap
def self_dispatch(t):
    if has_torch_function((t,)):
        return handle_torch_function(self_dispatch, (t,), t)
    return t * 2.0


def selfrec(t):
    """Not wrapped: FX records it through ``__torch_function__``."""
    if has_torch_function((t,)):
        return handle_torch_function(selfrec, (t,), t)
    return t * 2.0


@torch.fx.wrap
def mutual_a(t, n):
    return t if n == 0 else mutual_b(t * 2.0, n - 1)


@torch.fx.wrap
def mutual_b(t, n):
    return mutual_a(t, n)


@torch.fx.wrap
def self_rec(t, n):
    return t if n == 0 else self_rec(t * 2.0, n - 1)


@torch.fx.wrap
def deep_rec(t, n):
    """A recursion on its literal depth, natively bounded, deeper than Dynamo's interpretation reaches."""
    return t if n == 0 else deep_rec(t * 1.0, n - 1)


@torch.fx.wrap
def ping(t, n):
    return t if n == 0 else pong(t + 0.0, n - 1)


@torch.fx.wrap
def pong(t, n):
    return ping(t, n)


# ----------------------------------------------------------------------------- the programs
CASES = {
    # case: (the helper call on y = fc1(x), the helper whose call is decided, the expected outcome)
    "isinstance_tensor_arm": (lambda net, y: arm_isinstance(y), arm_isinstance, "refused"),
    "torch_is_tensor_arm": (lambda net, y: arm_is_tensor(y), arm_is_tensor, "refused"),
    "type_is_tensor_arm": (lambda net, y: arm_type_is(y), arm_type_is, "refused"),
    "hasattr_arm": (lambda net, y: arm_hasattr(y), arm_hasattr, "refused"),
    "is_fx_tracing_arm": (lambda net, y: arm_is_fx_tracing(y), arm_is_fx_tracing, "refused"),
    "try_len_fallback": (lambda net, y: try_len(y), try_len, "refused"),
    "str_of_tensor_rows": (lambda net, y: str_rows(y), str_rows, "refused"),
    "repr_of_tensor_len": (lambda net, y: repr_len(y), repr_len, "refused"),
    "try_iter": (lambda net, y: try_iter(y), try_iter, "refused"),
    "dtype_is_arm": (lambda net, y: dtype_is_arm(y), dtype_is_arm, "refused"),
    "dict_key_arm": (lambda net, y: dict_key_arm(y, {False: 1.0, True: 2.0}), dict_key_arm, "refused"),
    "kw_scale": (lambda net, y: kw_scale(y, scale=1.0, big=2.0), kw_scale, "refused"),
    "outer_calls_inner": (lambda net, y: outer_calls_inner(y), outer_calls_inner, "refused"),
    "ndim_isinstance": (lambda net, y: ndim_isinstance(y), ndim_isinstance, "certified"),
    "eq_one": (lambda net, y: eq_one(y), eq_one, "refused"),
    "ne_one_len": (lambda net, y: ne_one_len(y), ne_one_len, "refused"),
    "squeeze_rank": (lambda net, y: squeeze_rank(y), squeeze_rank, "refused"),
    "squeeze_all_rank": (lambda net, y: squeeze_all_rank(y), squeeze_all_rank, "refused"),
    "transpose_contiguous": (lambda net, y: transpose_contiguous(y), transpose_contiguous, "refused"),
    "eq_zero": (lambda net, y: eq_zero(y), eq_zero, "certified"),
    "reshape_rows": (lambda net, y: reshape_rows(y), reshape_rows, "certified"),
    "dim_branch": (lambda net, y: dim_branch(y), dim_branch, "certified"),
    "dtype_eq_branch": (lambda net, y: dtype_eq_branch(y), dtype_eq_branch, "certified"),
    "device_str": (lambda net, y: device_str(y), device_str, "certified"),
    "masked": (lambda net, y: masked(y), masked, "certified"),
    "no_grad_body": (lambda net, y: no_grad_body(y), no_grad_body, "certified"),
    "returns_none": (lambda net, y: y if returns_none(y) is None else y, returns_none, "certified"),
    "stochastic_depth": (lambda net, y: y + net.drop(y), stochastic_depth, "certified"),
    "nested_out": (lambda net, y: (lambda r: r[0] + r[1][0] * r[1][1]["z"])(nested_out(y, [2.0, 0.5])), nested_out,
                   "certified"),
    "literal_in_out": (lambda net, y: (lambda r: r[0] * r[1])(literal_in_out(y, (2.0, 3.0))), literal_in_out,
                       "certified"),
    "empty_list": (lambda net, y: empty_list(y, []), empty_list, "certified"),
    "none_and_list": (lambda net, y: none_and_list(y, None, [3.0]), none_and_list, "certified"),
    "int_keyed": (lambda net, y: (lambda d: d[0] * d[1])(int_keyed(y, {0: 2.0, 1: -0.5})), int_keyed, "certified"),
    "single_list_out": (lambda net, y: single_list_out(y, [2.0])[0], single_list_out, "certified"),
    "operand_out": (lambda net, y: (lambda r: r[0] - r[1])(operand_out(y, [3.0])), operand_out, "certified"),
    "namedtuple_out": (lambda net, y: (lambda p: p.a * p.b)(nt_out(y, [2.0, -1.0])), nt_out, "refused"),
    "shape_out": (lambda net, y: y.reshape(shape_out(y)), shape_out, "refused"),
    "outer_calls_opaque": (lambda net, y: outer_calls_opaque(y), outer_calls_opaque, "refused"),
    "scaled_twice": (lambda net, y: scaled_twice(y, 0.5, True), scaled_twice, "certified"),
    "self_dispatch": (lambda net, y: self_dispatch(y), self_dispatch, "certified"),
    "selfrec": (lambda net, y: selfrec(y), selfrec, "certified"),
    "mutual_a": (lambda net, y: mutual_a(y, 3), mutual_a, "certified"),
    "self_rec": (lambda net, y: self_rec(y, 3), self_rec, "certified"),
    "deep_rec": (lambda net, y: deep_rec(y, 300), deep_rec, "refused"),
    "ping_pong": (lambda net, y: ping(y, 300), ping, "refused"),
}
"""Every witness: certified ones must be exact at every batch, refused ones refused by name."""

CERTIFIED_WRONG_IN_ROUND_1 = ("isinstance_tensor_arm", "torch_is_tensor_arm", "type_is_tensor_arm", "hasattr_arm",
                              "is_fx_tracing_arm", "try_len_fallback", "str_of_tensor_rows", "repr_of_tensor_len",
                              "try_iter", "dtype_is_arm", "dict_key_arm", "kw_scale", "outer_calls_inner")
BATCH_ONE_FORMS = ("eq_one", "ne_one_len", "squeeze_rank", "squeeze_all_rank", "transpose_contiguous")


class Net(nn.Module):
    def __init__(self, case: str):
        super().__init__()
        self.case = case
        self.fc1, self.act, self.fc2 = nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3)
        self.drop = StochasticDepth(0.2, "row")  # torchvision's module: its forward calls the leaf function

    def forward(self, x):
        return self.fc2(self.act(CASES[self.case][0](self, self.fc1(x))))


class PeNet(Net):
    def __init__(self, rows: bool):
        super().__init__("isinstance_tensor_arm")
        self.rows = rows

    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(add_pe_rows(y, W) if self.rows else add_pe(y, W)))


def native(factory, *args) -> nn.Module:
    torch.manual_seed(0)
    return factory(*args).double().eval()


def x_at(batch: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(4242 + 31 * batch + seed)
    return torch.randn(batch, *SHAPE, generator=g, dtype=torch.float64)


def _worst(candidate, model, flat: bool = False) -> float:
    worst = 0.0
    for batch in BATCHES:
        x = x_at(batch)
        with torch.no_grad():
            want = model(x)
            got = candidate(x)
        got = got.reshape(want.shape) if flat else got
        assert got.shape == want.shape, (batch, tuple(got.shape), tuple(want.shape))
        worst = max(worst, relative_error(got, want).relative_error)
    return worst


def _named(exc: BaseException) -> bool:
    """A refusal by name: a toolchain refusal that is not a measured value mismatch (a probe refusal whose stage
    raised a named refusal is named; one that measured an error above its bound is not)."""
    if not isinstance(exc, ToolchainRefusal):
        return False
    if isinstance(exc, ConversionValueParityError):
        error = getattr(exc.result, "error", None)
        return isinstance(error, ToolchainRefusal)
    return True


def outcome(model: nn.Module) -> str:
    """``certified`` (both stages, the flow and its identity value program exact at every batch) or ``refused`` (by
    name, its converted model exact at every batch where it converts); fails on anything else."""
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
        worst = _worst(stage_1, model)
        assert worst <= EXACT, f"refused at stage 2, and its converted model is wrong ({worst:.3e})"
        return "refused"
    certificate = flow.conversion_certificate
    assert certificate.certified and certificate.missing_stages == (), certificate
    flow.eval()
    worst = _worst(flow, model)
    assert worst <= EXACT, f"CERTIFIED AND WRONG: converted model {worst:.3e}"
    worst = _worst(identity_value_program(flow), model, flat=True)
    assert worst <= EXACT, f"CERTIFIED AND WRONG: identity value program {worst:.3e}"
    return "certified"


def _decided(model: nn.Module, helper):
    """The traced program and the reasons its calls of ``helper`` stayed opaque (empty when inlined)."""
    gm = trace_model(copy.deepcopy(model), SHAPE)
    calls = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is helper]
    return gm, [n.meta.get(OPAQUE_BODY) for n in calls]


# ----------------------------------------------------------------------------- tests
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_witness_is_exact_or_refused_by_name(case):
    assert outcome(native(Net, case)) == CASES[case][2]


@pytest.mark.parametrize("case", CERTIFIED_WRONG_IN_ROUND_1)
def test_a_round_1_certified_wrong_helper_stays_opaque_with_a_named_reason(case):
    """[review round 1 of the coverage fixes: inlined from FX's Proxy path, certified at both stages and wrong by
    0.438 and 0.497 at batches 9 and 13]"""
    _, reasons = _decided(native(Net, case), CASES[case][1])
    assert reasons and all(reasons), reasons
    assert all("decides on the batch" in r or "Dynamo does not capture its body" in r for r in reasons), reasons


@pytest.mark.parametrize("case", BATCH_ONE_FORMS)
def test_a_body_that_differs_at_batch_1_stays_opaque(case):
    """Dynamo assumes a batch symbol is not 0 or 1; the squeezes escape every guard, the other forms guard
    ``Ne(B, 1)``; the call stays opaque either way, named."""
    _, reasons = _decided(native(Net, case), CASES[case][1])
    assert reasons and all(reasons), reasons
    assert all("at batch 1" in r or "decides on the batch" in r for r in reasons), reasons


@pytest.mark.parametrize("case", ["squeeze_rank", "squeeze_all_rank"])
def test_the_squeezes_are_caught_by_the_batch_1_capture(case):
    _, reasons = _decided(native(Net, case), CASES[case][1])
    assert reasons and all("at batch 1" in r for r in reasons), reasons


def test_ndim_read_with_isinstance_is_inlined_and_exact():
    """[review round 1: a lazy Attribute proxy of ``t.ndim``, never materialized; value mismatch 0.477 at every batch
    where the base named a refusal]"""
    gm, reasons = _decided(native(Net, "ndim_isinstance"), ndim_isinstance)
    assert reasons == [], reasons
    assert outcome(native(Net, "ndim_isinstance")) == "certified"


@pytest.mark.parametrize("rows", [False, True], ids=["inlined_body", "body_reading_the_batch_extent"])
@pytest.mark.parametrize("warm", [False, True], ids=["cold_cache", "warm_cache"])
def test_a_module_cache_holds_real_tensors_and_the_native_model_runs_after(rows, warm):
    """[review round 1: the trial trace stored a Proxy in the cache, and the native model raised ``NameError``, which
    the census recorded as the native model's rejection]"""
    cache = _PE_ROWS_CACHE if rows else _PE_CACHE
    cache.clear()
    model = native(PeNet, rows)
    if warm:
        with torch.no_grad():
            model(x_at(1))
    try:
        assert outcome(model) == "certified"
        assert cache and all(type(v) is torch.Tensor for v in cache.values()), cache
        with torch.no_grad():
            assert model(x_at(5)).shape == (5, 3)
    finally:
        cache.clear()


def test_a_shape_returned_by_a_body_is_not_held_constant():
    """``t.shape`` returned as it is: Dynamo's code rebuilds it from the operand at run time; spliced as a constant it
    would be the stand-in's shape."""
    _, reasons = _decided(native(Net, "shape_out"), shape_out)
    assert reasons and all(reasons), reasons


def test_a_namedtuple_output_stays_opaque_named():
    _, reasons = _decided(native(Net, "namedtuple_out"), nt_out)
    assert reasons and all(reasons), reasons


@pytest.mark.parametrize("case", ["deep_rec", "ping_pong"])
def test_a_recursion_dynamo_cannot_unroll_stays_opaque_and_its_reason_names_it(case):
    """[review round 1: a cut chain named "the result of an opaque call", not the recursion]"""
    _, reasons = _decided(native(Net, case), CASES[case][1])
    assert reasons and all("calls itself" in r for r in reasons), reasons


@pytest.mark.parametrize("case", ["self_dispatch", "selfrec", "mutual_a", "self_rec", "scaled_twice",
                                  "outer_calls_inner"])
def test_nested_and_self_dispatching_helpers_are_decided_once(case):
    """Dynamo interprets nested calls itself: a self-dispatching helper on plain tensors takes its tensor path, a
    recursion on a literal unrolls, and an inner helper's batch decision keeps the outer call opaque (one call node,
    never a chain of trial traces)."""
    gm, reasons = _decided(native(Net, case), CASES[case][1])
    names = {getattr(n.target, "__name__", "") for n in gm.graph.nodes if n.op == "call_function"}
    if CASES[case][2] == "certified":
        assert reasons == [] and not names & {"self_dispatch", "selfrec", "mutual_a", "mutual_b", "self_rec",
                                              "scaled", "scaled_twice"}, (reasons, names)
    else:
        assert len(reasons) == 1 and reasons[0] and "inner_arm" not in names, (reasons, names)


class StrideNet(Net):
    def __init__(self):
        super().__init__("reshape_rows")

    def forward(self, x):
        y = self.fc1(x)
        return self.fc2(self.act(stride_gate(y.t().contiguous()[:, 0], y)))


def test_an_operand_whose_strides_change_with_the_batch_at_fixed_extents_stays_opaque():
    """Dynamo holds an operand without a batch-carrying axis static, its strides included; one whose stride is the
    batch would be captured at the stand-in's. (Its opaque call's converted model differs from the native one: the
    host op reads its operand's stride, which the converted program does not keep, a boundary of host-op operand
    layouts that fix round 1 had too.)"""
    _, reasons = _decided(native(StrideNet), stride_gate)
    assert reasons and "changes its strides or offset with the batch" in reasons[0], reasons


def test_nothing_is_inlined_while_torchs_declared_code_is_rebound(monkeypatch):
    from mimarsinan.torch_mapping.fx_rewrites import leaf_call_inlining

    monkeypatch.setattr(leaf_call_inlining, "declared_code_change", lambda: "torch.sigmoid is rebound since import")
    _, reasons = _decided(native(Net, "reshape_rows"), reshape_rows)
    assert reasons == ["the declared operators are not torch's own code as imported: torch.sigmoid is rebound since "
                       "import"], reasons
