"""Leaf-call inlining binds a call by the callee's code (fix round 3 of the coverage fixes, verify finding R1).

Python binds a call by the function's code, ``__code__``, ``__defaults__`` and ``__kwdefaults__``. A reported signature
can differ from it: ``__signature__`` (a signature-preserving decorator copies the wrapped function's onto the wrapper),
``__wrapped__`` and ``__text_signature__``, each of which ``inspect.signature`` reads. Round 2 bound with
``inspect.signature(follow_wrapped=False)``, which still reads ``__signature__``: ``sig_clamp(t, cap=1e4)`` reporting
``(t, cap=1e6)`` was inlined as ``clamp(max=1e6)``, certified and wrong at inputs of scale 1e5.

Every binding and rebuild of an inlined call now uses the signature of a fresh function made from the callee's code,
globals, name, defaults and closure, its keyword defaults copied (``operand_samples.code_call``); a bound method is
its ``__func__`` with ``__self__`` bound first; any other callable (a ``functools.partial``, a callable object) stays
opaque by name. Each witness here ends certified and exact at every batch and at inputs of scale 1 and 1e5 (the
converted model and its identity value program), or refused by name with its converted model exact.
"""

from __future__ import annotations

import functools
import inspect
import re
import types
from pathlib import Path

import pytest
import torch
import torch.fx
import torch.nn as nn

from mimarsinan.torch_mapping.fx_rewrites import operand_samples
from mimarsinan.torch_mapping.fx_rewrites.body_capture import capture_body
from mimarsinan.torch_mapping.fx_rewrites.body_output import OperandOutput
from mimarsinan.torch_mapping.fx_rewrites.operand_samples import TensorSample, sample_operands
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model

from .test_body_capture_review_r2 import _reasons, native, outcome


def code_call(target):
    return operand_samples.code_call(target)  # read at the call: the tests run (red) on a tree without it

W = 4
SHAPE = (W,)
BATCHES = (1, 2, 3, 9, 13)
"""The batches compared (each at input scales 1 and 1e5, ``outcome``)."""
NATIVE_CAP, REPORTED_CAP = 1e4, 1e6


def _reported(function, signature):
    function.__signature__ = signature
    return function


def _clamp_reported(t, cap=REPORTED_CAP):
    return t.clamp(max=cap)


# ----------------------------------------------------------------------------- reported signatures that lie
def sig_clamp(t, cap=NATIVE_CAP):
    """Natively clamps at 1e4; reports ``(t, cap=1e6)`` by ``__signature__`` (no ``__wrapped__``): the verify probe's
    ``b4v_signature_attr``."""
    return _clamp_reported(t, cap)


sig_clamp.__signature__ = inspect.signature(_clamp_reported)
torch.fx.wrap("sig_clamp")


def sub_pair(a, b):
    """Reports its parameters in the other order, ``(b, a)``."""
    return a - 2.0 * b


sub_pair.__signature__ = inspect.Signature([inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                            for n in ("b", "a")])
torch.fx.wrap("sub_pair")


def var_scale(*args):
    """Variadic code reporting a plain ``(t, k=2.0)``: natively ``var_scale(h)`` scales by 3."""
    return args[0] * (args[1] if len(args) > 1 else 3.0)


var_scale.__signature__ = inspect.Signature([
    inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("k", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=2.0)])
torch.fx.wrap("var_scale")


def kw_kind(t, cap=NATIVE_CAP):
    """Reports ``cap`` as keyword-only with the other default."""
    return t.clamp(max=cap)


kw_kind.__signature__ = inspect.Signature([
    inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("cap", inspect.Parameter.KEYWORD_ONLY, default=REPORTED_CAP)])
torch.fx.wrap("kw_kind")


def _signature_preserving(f):
    """A decorator that keeps the wrapped function's signature: ``functools.wraps`` and ``__signature__`` copied."""

    @functools.wraps(f)
    def wrapper(t, cap=NATIVE_CAP):
        return f(t, cap)

    wrapper.__signature__ = inspect.signature(f)
    return wrapper


@torch.fx.wrap
@_signature_preserving
def wraps_sig_clamp(t, cap=REPORTED_CAP):
    return t.clamp(max=cap)


def text_clamp(t, cap=NATIVE_CAP):
    """Reports ``(t, cap=1000000.0)`` by ``__text_signature__``, which ``inspect.signature`` reads first."""
    return t.clamp(max=cap)


text_clamp.__text_signature__ = f"(t, cap={REPORTED_CAP!r})"
torch.fx.wrap("text_clamp")


# ----------------------------------------------------------------------------- defaults reassigned at run time
def redefaulted(t, cap=REPORTED_CAP):
    return t.clamp(max=cap)


redefaulted.__defaults__ = (NATIVE_CAP,)
torch.fx.wrap("redefaulted")


def rekwdefaulted(t, *, cap=REPORTED_CAP):
    return t.clamp(max=cap)


rekwdefaulted.__kwdefaults__ = {"cap": NATIVE_CAP}
torch.fx.wrap("rekwdefaulted")


def stale_defaults(t, cap=REPORTED_CAP):
    """Its ``__signature__`` taken before its ``__defaults__`` were reassigned: the reported default is stale."""
    return t.clamp(max=cap)


_reported(stale_defaults, inspect.signature(stale_defaults)).__defaults__ = (NATIVE_CAP,)
torch.fx.wrap("stale_defaults")


def stale_kwdefaults(t, *, cap=REPORTED_CAP):
    return t.clamp(max=cap)


_reported(stale_kwdefaults, inspect.signature(stale_kwdefaults)).__kwdefaults__ = {"cap": NATIVE_CAP}
torch.fx.wrap("stale_kwdefaults")


# ----------------------------------------------------------------------------- callables other than a function
class Scaler:
    def __init__(self, k: float):
        self.k = k

    def scale(self, t, k=None):
        return t * (self.k if k is None else k)


method_object = Scaler(3.0).scale
torch.fx.wrap("method_object")


def _times(k, t):
    return t * k


method_literal = types.MethodType(_times, 2.0)
torch.fx.wrap("method_literal")


def _times_cap(k, t, cap=NATIVE_CAP):
    return (t * k).clamp(max=cap)


_times_cap.__signature__ = inspect.Signature([
    inspect.Parameter("k", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("t", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    inspect.Parameter("cap", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=REPORTED_CAP)])
method_literal_sig = types.MethodType(_times_cap, 2.0)
torch.fx.wrap("method_literal_sig")


def _partial_body(t, cap=REPORTED_CAP):
    return t.clamp(max=cap)


partial_clamp = functools.partial(_partial_body, cap=NATIVE_CAP)
partial_clamp.__name__ = "partial_clamp"  # FX names a call_function node by its target's __name__
torch.fx.wrap("partial_clamp")


class _Clamp:
    def __call__(self, t):
        return t.clamp(max=NATIVE_CAP)


object_clamp = _Clamp()
object_clamp.__name__ = "object_clamp"
torch.fx.wrap("object_clamp")


CASES = {
    # case: (the forward after h = fc1(x), the helper, the outcome, a text every reason holds, the spliced clamp bound)
    "signature_attr": (lambda h: sig_clamp(h), sig_clamp, "certified", None, NATIVE_CAP),
    "signature_order": (lambda h: sub_pair(h, h * 0.25), sub_pair, "certified", None, None),
    "signature_kind_variadic": (lambda h: var_scale(h), var_scale, "refused", "variadic", None),
    "signature_kind_keyword": (lambda h: kw_kind(h), kw_kind, "certified", None, NATIVE_CAP),
    "signature_on_wraps_wrapper": (lambda h: wraps_sig_clamp(h), wraps_sig_clamp, "certified", None, NATIVE_CAP),
    "text_signature": (lambda h: text_clamp(h), text_clamp, "certified", None, NATIVE_CAP),
    "defaults_reassigned": (lambda h: redefaulted(h), redefaulted, "certified", None, NATIVE_CAP),
    "kwdefaults_reassigned": (lambda h: rekwdefaulted(h), rekwdefaulted, "certified", None, NATIVE_CAP),
    "defaults_after_signature": (lambda h: stale_defaults(h), stale_defaults, "certified", None, NATIVE_CAP),
    "kwdefaults_after_signature": (lambda h: stale_kwdefaults(h), stale_kwdefaults, "certified", None, NATIVE_CAP),
    "bound_method_object": (lambda h: method_object(h), method_object, "refused", "its operand 'self' is", None),
    "bound_method_literal": (lambda h: method_literal(h), method_literal, "certified", None, None),
    "bound_method_reported": (lambda h: method_literal_sig(h), method_literal_sig, "certified", None, NATIVE_CAP),
    "functools_partial": (lambda h: partial_clamp(h), partial_clamp, "refused", "functools.partial", None),
    "callable_object": (lambda h: object_clamp(h), object_clamp, "refused", "_Clamp", None),
}


class Net(nn.Module):
    def __init__(self, case: str):
        super().__init__()
        self.case = case
        self.fc1, self.act, self.fc2 = nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3)

    def forward(self, x):
        return self.fc2(self.act(CASES[self.case][0](self.fc1(x))))


# ----------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("case", sorted(CASES))
def test_every_reported_signature_witness_is_exact_or_refused_by_name(case):
    assert outcome(native(Net, case), BATCHES) == CASES[case][2]


@pytest.mark.parametrize("case", sorted(c for c in CASES if CASES[c][2] == "refused"))
def test_a_call_whose_binding_its_code_does_not_give_stays_opaque_and_names_why(case):
    reasons = _reasons(native(Net, case), CASES[case][1])
    assert reasons and all(r and CASES[case][3] in r for r in reasons), reasons


@pytest.mark.parametrize("case", sorted(c for c in CASES if CASES[c][2] == "certified"))
def test_a_call_is_inlined_with_the_values_its_code_binds(case):
    gm = trace_model(native(Net, case), SHAPE)
    assert not [n for n in gm.graph.nodes if n.op == "call_function" and n.target is CASES[case][1]], gm.graph
    want = CASES[case][4]
    if want is not None:
        clamps = [n for n in gm.graph.nodes if n.op == "call_method" and n.target == "clamp"]
        assert [n.kwargs.get("max") for n in clamps] == [want], gm.graph


# ----------------------------------------------------------------------------- the binding sites, directly
def _swapped(function):
    """``function`` reporting its two parameters in the other order."""
    names = list(inspect.signature(function).parameters)[::-1]
    return _reported(function, inspect.Signature([
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names]))


def _rows(rows_at, width):
    return tuple(TensorSample.of(torch.empty(rows_at(b), width, dtype=torch.float64)) for b in (1, 2, 3))


def test_the_capture_passes_each_stand_in_as_the_code_names_it():
    """``_run_capture`` built the call from the reported order: ``a``'s stand-in was passed as ``b``."""

    def body(a, b):
        return a * 2.0 + b.sum(1, keepdim=True)

    found = capture_body(_swapped(body), {"a": None, "b": None},
                         {"a": _rows(lambda b: b, W), "b": _rows(lambda b: b, 1)})
    assert not isinstance(found, str), found
    operands = sorted(binding for binding in found.bindings if binding[0] == "operand")
    assert operands == [("operand", "a"), ("operand", "b")], found.bindings


def test_the_output_is_rebuilt_with_the_operands_the_code_names():
    """``_rebuild`` ran the residual code with the reported order: a returned operand was named as the other."""

    def body(a, b):
        return b, a * 2.0

    found = capture_body(_swapped(body), {"a": None, "b": None},
                         {"a": _rows(lambda b: b, W), "b": _rows(lambda b: b, W)})
    assert not isinstance(found, str), found
    assert found.output[0] == OperandOutput("b"), found.output


@torch.fx.wrap
def rows_pair(a, b):
    return a * 2.0 + b.sum(1, keepdim=True)


_swapped(rows_pair)


class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(W, W), nn.Linear(W, 3)

    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(rows_pair(h, h[:, :1] * 1.0))


def test_the_seeded_runs_record_each_operand_under_the_name_its_code_gives():
    model = native(PairNet)
    gm = torch.fx.symbolic_trace(model)
    calls = [n for n in gm.graph.nodes if n.op == "call_function" and n.target is rows_pair]
    rows = sample_operands(gm, calls, SHAPE, "cpu").operands[calls[0]]
    assert [(row["a"].shape[1], row["b"].shape[1]) for row in rows] == [(W, 1)] * 3, rows


def test_code_call_reads_the_code_never_a_reported_signature():
    for function, reported in ((sig_clamp, REPORTED_CAP), (text_clamp, REPORTED_CAP),
                               (stale_defaults, REPORTED_CAP), (wraps_sig_clamp, REPORTED_CAP)):
        assert inspect.signature(function, follow_wrapped=False).parameters["cap"].default == reported
        found = code_call(function)
        assert not isinstance(found, str) and found.function is function and found.leading == ()
        assert found.signature.parameters["cap"].default == NATIVE_CAP, found.signature
    assert list(code_call(sub_pair).signature.parameters) == ["a", "b"]
    assert [p.kind for p in code_call(var_scale).signature.parameters.values()] == [inspect.Parameter.VAR_POSITIONAL]
    assert code_call(stale_kwdefaults).signature.parameters["cap"].default == NATIVE_CAP


def test_code_call_binds_a_bound_method_with_its_self_first():
    found = code_call(method_literal_sig)
    assert not isinstance(found, str) and found.function is _times_cap and found.leading == (2.0,)
    bound = found.bind(("t",), {})
    bound.apply_defaults()
    assert dict(bound.arguments) == {"k": 2.0, "t": "t", "cap": NATIVE_CAP}


@pytest.mark.parametrize("target, named", [(partial_clamp, "functools.partial"), (object_clamp, "_Clamp"),
                                           (len, "builtin_function_or_method"),
                                           (types.MethodType(partial_clamp, 1.0), "functools.partial")])
def test_code_call_names_why_a_callable_that_is_not_a_function_has_no_code_binding(target, named):
    found = code_call(target)
    assert isinstance(found, str) and named in found, found


def test_no_binding_site_reads_a_reported_signature():
    """Every ``inspect.signature`` of leaf-call inlining is the one of ``code_signature`` (a ratchet)."""
    root = Path(operand_samples.__file__).resolve().parent
    found = {path.name: len(re.findall(r"inspect\.signature\(", path.read_text())) for path in root.glob("*.py")}
    assert {name: n for name, n in found.items() if n} == {"operand_samples.py": 1}, found
    for path in root.glob("*.py"):
        assert "follow_wrapped" not in path.read_text(), path.name
