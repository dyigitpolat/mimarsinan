"""The body of a Python helper as TorchDynamo captures it, for leaf-call inlining (fix round 2 of the coverage fixes).

Dynamo's full-graph frontend (``torch._dynamo.convert_frame.fullgraph_capture``, private in torch 2.11) interprets the
helper's bytecode on FakeTensors into one torch-level FX graph, or raises; it never runs the helper. A Python decision
on a tensor (``isinstance``, ``type``, ``hasattr``, ``is_fx_tracing()``, ``len``, iteration, ``dtype is``) is
evaluated as it is for a real tensor, a read of the batch extent is a symbol, ``str`` or ``repr`` of a tensor and a
read of its data are errors, and a write to state outside the frame is an error (``CAPTURE_CONFIG``). The capture runs
on stand-in operands (empty tensors with each operand's metadata, so no user tensor is tagged), through a copy of the
helper's globals (Dynamo installs names into the frame's globals), and never calls ``torch._dynamo.reset()``.

The operands' metadata come from the seeded native runs at batches 1, 2 and 3 (``operand_samples``; one stand-in per
tensor object, and a call whose distinct operands share a storage stays opaque). An axis whose extent differs between
them carries the batch and is marked dynamic, and every symbol of the capture (an extent, a stride, a storage offset of
an operand) must be affine in the batch across them; the symbol then stands for that form ``a*B + c``, which the
converter's batch domain proves for every extent before the call is inlined (``leaf_operands``). ``capture_body``
returns the body (``CapturedBody``) only when:

- (a) the capture is one graph;
- (b) every guard, deferred runtime assert, replacement, range, symbolic operation, static evaluation, ``guard_or``
  decision and Dynamo fold of the capture is proved in the affine batch domain, each symbol at its form
  (``body_rules``, coverage W8), and every operand's captured sizes, strides and offset reproduce every sample;
- (c) at batch 1 and at every batch B >= 1 where a symbol's form is 0 or 1, a capture with every extent static is the
  same program as the first at that B (Dynamo assumes a batch symbol is not 0 or 1, and a squeeze escapes every
  guard; ``body_comparison``; review round 2: ``B - 1`` is 1 at batch 2);
- (d) nothing is written in place (a write torch declares or names, ``__imul__``) once every assignment of a literal
  to a tensor the body allocates is computed out of place (``body_rules.local_writes``, coverage W8): any other write
  stays the opaque call's effect, which the probe observes (T3), never one of the program's own writes;
- (e) after the normalization every node is a declared operator (``declared_operators.opaque_reason``);

else the reason the call stays opaque. Normalization (``body_output``): a dead call is dropped only when it has no
node operand (trace-time Python) or a schema with no write or alias; a dunder method is its ``operator`` function.
The output structure is rebuilt by Dynamo's own residual code, run on inert operands under a profile hook that refuses
any call it makes other than Dynamo's helpers and the graph's stand-in, so a user class is never constructed
(``body_output``). Any failure of the machinery, a missing private API included, leaves the call opaque with its
reason; nothing is inlined on a failure.
"""

from __future__ import annotations

import contextlib
import inspect
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import sympy
import torch
from torch._dynamo import config as dynamo_config
from torch._dynamo import convert_frame, source as dynamo_source
from torch._dynamo import utils as dynamo_utils
from torch.fx.experimental import _config as fx_config
from torch.utils._sympy.numbers import int_oo

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.fx_rewrites.body_comparison import differs_at_batch
from mimarsinan.torch_mapping.fx_rewrites.body_output import BodyRefusal as _Opaque
from mimarsinan.torch_mapping.fx_rewrites.body_output import CapturedBody, extract, first_line
from mimarsinan.torch_mapping.fx_rewrites.body_rules import prove, recorded, rewrite_assignments, unproved_record
from mimarsinan.torch_mapping.fx_rewrites.operand_samples import (
    TRACE_BATCHES, TensorSample, affine_fit, code_function, code_signature, identity_reason, sample_at, stand_ins,
)

__all__ = ["CAPTURE_CONFIG", "BATCH", "capture_body", "capture_unavailable"]

CAPTURE_CONFIG = dict(
    side_effect_replay_policy="error",  # a write to state outside the frame is an error
    replay_side_effects=False,
    assume_static_by_default=True,  # an axis not marked is static
    automatic_dynamic_shapes=False,
    specialize_int=True,  # an int or float literal is held constant
    specialize_float=True,
    capture_scalar_outputs=False,
    capture_dynamic_output_shape_ops=False,
)
"""Dynamo's configuration for a capture (``torch._dynamo.config``)."""

SHAPE_CONFIG = dict(backed_size_oblivious=True, use_duck_shape=False)
"""``torch.fx.experimental._config``: a batch symbol ranges over [0, inf) and torch's own size checks are oblivious to
0 and 1; two extents equal at the hint get two symbols, so a body relating them records the relation as a guard."""

BATCH = sympy.Symbol("B", integer=True, positive=True)
"""The batch in a symbol's form ``a*B + c``."""

_PRIVATE_API = (
    (convert_frame, ("fullgraph_capture", "CaptureOutput")),
    (dynamo_utils, ("get_metrics_context", "call_size", "call_stride", "call_storage_offset")),
    (dynamo_source, ("LocalSource", "GlobalSource", "TensorPropertySource", "TensorProperty", "GetItemSource",
                     "DictGetItemSource", "AttrSource")),
    (fx_config, tuple(SHAPE_CONFIG)),
    (dynamo_config, tuple(CAPTURE_CONFIG)),
)
"""The private names the capture reads (torch 2.11), each read at the call: one missing leaves every call opaque."""


def capture_unavailable() -> Optional[str]:
    """Why the private capture API is not available in this torch; ``None`` when it is."""
    for module, names in _PRIVATE_API:
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            return f"the body capture is not available in this torch ({module.__name__}.{missing[0]} is missing)"
    return None


def _globals_copy(function: types.FunctionType) -> types.FunctionType:
    """``function``'s code function (``code_function``) with a copy of its globals, where Dynamo installs its names."""
    copy = code_function(function, dict(function.__globals__))
    copy.__qualname__, copy.__module__ = function.__qualname__, function.__module__
    return copy


def _run_capture(function: types.FunctionType, arguments: Dict[str, Any], operands: Dict[str, torch.Tensor],
                 marks: Dict[str, Tuple[int, ...]]):
    """One capture with the operands' stand-ins ``operands``, ``marks`` (operand -> axes) dynamic, by its code."""
    target = _globals_copy(function)
    for name, axis in [(name, axis) for name, axes in marks.items() for axis in axes]:
        torch._dynamo.mark_dynamic(operands[name], axis)
    bound = inspect.BoundArguments(code_signature(function), {  # type: ignore[arg-type]
        name: operands.get(name, value) for name, value in arguments.items()})
    with contextlib.ExitStack() as stack:
        stack.enter_context(cast(Any, dynamo_config).patch(**CAPTURE_CONFIG))
        stack.enter_context(cast(Any, fx_config).patch(**SHAPE_CONFIG))
        stack.enter_context(dynamo_utils.get_metrics_context())
        records = stack.enter_context(recorded())  # torch's symbolic folds, re-proved in ``_judge``
        captured, exc = guarded(lambda: convert_frame.fullgraph_capture(target, tuple(bound.args), dict(bound.kwargs)))
    if isinstance(exc, RecursionError):
        raise _Opaque(f"its body calls itself deeper than Dynamo's interpretation reaches ({first_line(exc)})")
    if exc is not None or captured is None:  # Dynamo's refusal of the body, whatever its class
        raise _Opaque(f"Dynamo does not capture its body as one graph ({first_line(exc) if exc else 'no capture'})")
    return captured, target, records


def _form(values: Sequence[int]) -> Optional[sympy.Expr]:
    """The affine form ``a*B + c`` through the samples at ``TRACE_BATCHES``, ``None`` when they are not affine."""
    fit = affine_fit(values)
    return None if fit is None else fit[0] * BATCH + fit[1]


def _comparison_batches(forms: Dict[sympy.Symbol, sympy.Expr]) -> List[int]:
    """Batch 1 and every batch B >= 1 at which a batch symbol's form is 0 or 1 (Dynamo assumes it is neither)."""
    found = set()
    for form in forms.values():
        if not form.free_symbols:
            continue
        a, c = int(cast(Any, form.coeff(BATCH, 1))), int(cast(Any, form.subs(BATCH, 0)))
        found |= {1} | {(v - c) // a for v in (0, 1) if a and (v - c) % a == 0 and (v - c) // a >= 1}
    return sorted(found)


def _property(source, samples: Dict[str, Tuple[TensorSample, ...]]) -> Optional[Tuple[int, ...]]:
    """The sampled values of a ``TensorPropertySource`` of an operand, ``None`` for any other source."""
    base = getattr(source, "base", None)
    if not isinstance(source, dynamo_source.TensorPropertySource) or not isinstance(
            base, dynamo_source.LocalSource) or base.local_name not in samples:
        return None
    rows, axis = samples[base.local_name], source.idx if source.idx is not None else 0
    if source.prop is dynamo_source.TensorProperty.SIZE:
        return tuple(row.shape[axis] for row in rows)
    if source.prop is dynamo_source.TensorProperty.STRIDE:
        return tuple(row.stride[axis] for row in rows)
    return tuple(row.offset for row in rows)


def _forms(shape_env, samples) -> Dict[sympy.Symbol, sympy.Expr]:
    """Every symbol of the capture with a source: its affine form; ``_Opaque`` when one has none."""
    forms: Dict[sympy.Symbol, sympy.Expr] = {}
    for symbol, sources in shape_env.var_to_sources.items():
        for source in sources:
            values = _property(source, samples)
            form = None if values is None else _form(values)
            if form is None:
                where = source.name if values is None else f"{source.name}: {', '.join(map(str, values))} at batches " \
                    f"{', '.join(map(str, TRACE_BATCHES))}"
                raise _Opaque(f"its capture holds a symbol the batch does not give an affine form ({where})")
            if forms.setdefault(symbol, form) != form:
                raise _Opaque(f"its capture holds one symbol for two forms ({forms[symbol]}, {form})")
    return forms


def _judge(shape_env, forms: Dict[sympy.Symbol, sympy.Expr], records: list) -> None:
    """(b): every guard, runtime assert, replacement, range, symbolic operation and static evaluation (``records``)
    of the capture is proved for every batch B >= 1 in the affine batch domain (``body_rules``); ``_Opaque`` else."""
    statements: List[Tuple[str, sympy.Basic]] = [("guard", g.expr) for g in shape_env.guards]
    statements += [("runtime assert", ra.expr) for ras in shape_env.deferred_runtime_asserts.values() for ra in ras]
    statements += [("replacement", sympy.Eq(sym, rep)) for sym, rep in shape_env.replacements.items()]
    for symbol, form in forms.items():
        found = shape_env.var_to_range.get(symbol)
        if found is not None:
            statements.append(("range", sympy.And(sympy.Ge(form, found.lower) if found.lower != -int_oo else True,
                                                  sympy.Le(form, found.upper) if found.upper != int_oo else True)))
    for kind, statement in statements:
        free = getattr(statement, "free_symbols", set()) - {BATCH}
        unknown = free - set(forms)
        if unknown:
            raise _Opaque(f"its capture states {kind} {statement} on a value outside the batch domain "
                          f"({', '.join(sorted(map(str, unknown)))})")
        if not prove(statement, forms, BATCH):
            shown = ", ".join(f"{s} = {forms[s]}" for s in sorted(free, key=str))
            raise _Opaque(f"its body decides on the batch: Dynamo's {kind} {statement} ({shown}) is not proved for "
                          "every batch B >= 1 in the affine batch domain")
    unproved = unproved_record(records, forms, BATCH, _comparison_batches(forms))
    if unproved is not None:
        raise _Opaque(unproved)


def _value_at(value: Any, forms: Dict[sympy.Symbol, sympy.Expr], batch: int) -> Any:
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        found = value.node.expr.xreplace({s: f.subs(BATCH, batch) for s, f in forms.items()})
        return int(found) if found.is_Integer else found
    return value


def _reproduces(body: CapturedBody, forms, samples) -> None:
    """(b): each operand's captured sizes, strides and offset are its sampled ones at every sampled batch."""
    for node, binding in zip(body.placeholders(), body.bindings):
        fake = node.meta.get("example_value")
        if binding[0] != "operand" or not isinstance(fake, torch.Tensor):
            continue
        for batch, row in zip(TRACE_BATCHES, samples[binding[1]]):
            sizes = [_value_at(n, forms, batch) for n in fake.shape]
            strides = [_value_at(s, forms, batch) if n != 1 else 1 for s, n in zip(fake.stride(), row.shape)]
            captured = (sizes, strides, _value_at(fake.storage_offset(), forms, batch))
            native = list(_layout(row)[0])
            if captured != (list(row.shape), native, row.offset):
                raise _Opaque(f"its operand {binding[1]!r} is captured as sizes {captured[0]}, strides {captured[1]}, "
                              f"offset {captured[2]} at batch {batch}, natively {list(row.shape)}, {native}, "
                              f"{row.offset}")


def _layout(row: TensorSample) -> tuple:
    """A sample's strides (an extent-1 axis's as 1: torch ignores it) and offset."""
    return tuple(s if n != 1 else 1 for s, n in zip(row.stride, row.shape)), row.offset


def _marks(samples: Dict[str, Tuple[TensorSample, ...]]) -> Dict[str, Tuple[int, ...]]:
    """Per operand, the axes whose extent differs between the samples (each a batch symbol of the capture). An
    operand with none is static for Dynamo, its strides and offset included, so they must not change either."""
    marks = {}
    for name, rows in samples.items():
        axes = tuple(axis for axis in range(len(rows[0].shape)) if len({row.shape[axis] for row in rows}) > 1)
        if not axes and len({_layout(row) for row in rows}) > 1:
            raise _Opaque(f"its operand {name!r} keeps its extents and changes its strides or offset with the batch")
        if axes:
            small = [axis for axis in axes if rows[-1].shape[axis] < 2]
            if small:
                raise _Opaque(f"its operand {name!r} has a batch-carrying axis {small[0]} of extent "
                              f"{rows[-1].shape[small[0]]} at batch {TRACE_BATCHES[-1]}")
            marks[name] = axes
    return marks


def capture_body(function: types.FunctionType, arguments: Dict[str, Any],
                 samples: Dict[str, Tuple[TensorSample, ...]]) -> Union[CapturedBody, str]:
    """``function``'s body for the call bound to ``arguments`` (every tensor operand by name in ``samples``, its
    metadata at ``TRACE_BATCHES``; every other argument a literal), or the reason the call stays opaque."""
    unavailable = capture_unavailable()
    if unavailable is not None:
        return unavailable
    found, exc = guarded(lambda: _capture(function, arguments, samples))
    if isinstance(exc, _Opaque):
        return str(exc)
    if exc is not None or found is None:  # a failure of the capture machinery: the call stays opaque
        return f"its body capture failed ({first_line(exc) if exc is not None else 'no body'})"
    return found


def _extract(*capture: Any) -> CapturedBody:
    """``extract``, then rule (d)'s rewrite of the body's assignments (``body_rules.local_writes``)."""
    body = extract(*capture)
    reason = rewrite_assignments(body.gm)
    if reason is not None:
        raise _Opaque(reason)
    return body


def _capture(function: types.FunctionType, arguments: Dict[str, Any],
             samples: Dict[str, Tuple[TensorSample, ...]]) -> CapturedBody:
    marks = _marks(samples)
    reason = identity_reason(samples)
    if reason is not None:
        raise _Opaque(reason)
    captured, target, records = _run_capture(function, arguments, stand_ins(samples, len(TRACE_BATCHES) - 1), marks)
    body = _extract(captured, target, function, arguments, samples)
    shape_env = captured.graph_capture_output.output_graph.shape_env
    forms = _forms(shape_env, samples)
    _judge(shape_env, forms, records)
    _reproduces(body, forms, samples)
    body.check_declared()
    for batch in _comparison_batches(forms):
        at = {name: (sample_at(rows, batch),) for name, rows in samples.items()}
        missing = next((name for name, row in at.items() if row[0] is None), None)
        if missing is not None:
            raise _Opaque(f"its operand {missing!r} has no affine metadata at batch {batch}, where a batch symbol of "
                          "its capture is 0 or 1")
        one, one_target, folds = _run_capture(function, arguments, stand_ins(cast(Any, at), 0), {})
        other = _extract(one, one_target, function, arguments, samples)
        other.check_declared()
        found = unproved_record(folds, {}, BATCH, []) or differs_at_batch(  # its folds the helper's own answers too
            body, other, {s: f.subs(BATCH, batch) for s, f in forms.items()})
        if found is not None:
            raise _Opaque(f"its body takes another path at batch {batch} ({found})")
    return body
