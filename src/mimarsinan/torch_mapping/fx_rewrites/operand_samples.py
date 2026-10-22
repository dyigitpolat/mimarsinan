"""Seeded native runs of a traced program at the converter's trace batches (``trace_values.TRACE_BATCHES``: 1, 2 and
3) that record, for each leaf call to be decided, the metadata of every operand it is called with, and the run-time
type of every value at batch 1.

The runs are the converter's own kind of sample (``trace_values``): the program runs natively on seeded inputs, every
call outside the declared set on real tensors of real batches, as the shape propagation and the probe run it; no
random state is read and no gradient is recorded. A run that fails leaves the calls after the failure without samples.
Only metadata is kept (``TensorSample``: extents, strides, storage offset, dtype, device, ``requires_grad``, exact type,
and the call's first operand that is the same tensor object, and the first whose storage it shares), never a value.
The capture's stand-ins are built from it (``stand_ins``: one per object, so ``a is b`` answers as natively), at a
trace batch or at another one by the samples' affine continuation (``sample_at``).

Every binding and rebuild of a leaf call (these runs, the decision, the capture, the return value, the splice) binds
it as Python does, by its target's code (``code_call``): the signature of a fresh function of the code, globals, name,
defaults and closure, its keyword defaults copied, never a signature the target reports (``__signature__``,
``__wrapped__``, ``__text_signature__``; fix round 3 of the coverage fixes: round 2 still read ``__signature__``); a
bound method is its ``__func__`` with ``__self__`` bound first; any other callable has no binding from code.
"""

from __future__ import annotations

import dataclasses
import inspect
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
import torch.fx as fx

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.trace_values import TRACE_BATCHES, rng_neutral

__all__ = ["CodeCall", "OperandSamples", "TRACE_BATCHES", "TensorSample", "affine_fit", "code_call", "code_function",
           "code_signature", "example_dtype", "identity_reason", "sample_at", "sample_operands", "stand_ins"]


def code_function(function: types.FunctionType, globals_: Optional[Dict[str, Any]] = None) -> types.FunctionType:
    """A fresh function of ``function``'s code, globals (``globals_`` when given), name, defaults and closure, its
    keyword defaults copied: what Python binds a call of ``function`` by, and nothing it reports (an empty
    ``__dict__``: no ``__signature__``, ``__wrapped__`` or ``__text_signature__``)."""
    fresh = types.FunctionType(function.__code__, function.__globals__ if globals_ is None else globals_,
                               function.__name__, function.__defaults__, function.__closure__)
    fresh.__kwdefaults__ = dict(function.__kwdefaults__) if function.__kwdefaults__ else None
    return fresh


def code_signature(function: types.FunctionType) -> inspect.Signature:
    """The signature Python binds a call of ``function`` by: its code's, never one ``function`` reports."""
    return inspect.signature(code_function(function))


@dataclass(frozen=True)
class CodeCall:
    """A call target's binding from its code: ``function`` (a Python function) and its ``code_signature``, with
    ``leading`` (a bound method's ``__self__``) bound before the call's own operands."""

    function: types.FunctionType
    signature: inspect.Signature
    leading: Tuple[Any, ...] = ()

    def bind(self, args: Sequence[Any], kwargs: Mapping[str, Any]) -> inspect.BoundArguments:
        return self.signature.bind(*self.leading, *args, **kwargs)


def code_call(target: Any) -> Union[CodeCall, str]:
    """``target``'s binding from its code (a bound method: its ``__func__``'s, ``__self__`` first); else why it has
    none (a ``functools.partial``, a callable object, a builtin: its binding is not a Python function's code)."""
    method = type(target) is types.MethodType
    function, leading = (target.__func__, (target.__self__,)) if method else (target, ())
    if type(function) is not types.FunctionType:
        kind = type(function)
        return (f"its target is {'a method of ' if method else ''}a {kind.__module__}.{kind.__qualname__}, not a "
                "Python function whose code gives its binding")
    return CodeCall(function, code_signature(function), leading)


@dataclass(frozen=True)
class TensorSample:
    """A strided tensor's metadata at one batch."""

    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    offset: int
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool
    kind: type
    same_object: str = ""  # the call's first operand that is this very tensor ("": not recorded)
    same_storage: str = ""  # the call's first operand whose storage this tensor's is

    @classmethod
    def of(cls, value: torch.Tensor, same_object: str = "", same_storage: str = "") -> "TensorSample":
        return cls(tuple(int(n) for n in value.shape), tuple(int(s) for s in value.stride()),
                   int(value.storage_offset()), value.dtype, value.device, bool(value.requires_grad), type(value),
                   same_object, same_storage)


def _storage(value: torch.Tensor) -> Optional[Tuple[int, int]]:
    storage = value.untyped_storage()
    return (storage.data_ptr(), storage.nbytes()) if storage.data_ptr() else None  # an empty storage is no one's


def _row(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """A call's bound operands: a ``TensorSample`` per strided tensor (with the first operand that is the same object,
    and the first of the same storage), the value's type for anything else."""
    tensors = {name: value for name, value in arguments.items()
               if isinstance(value, torch.Tensor) and value.layout == torch.strided}
    storages = {name: _storage(value) for name, value in tensors.items()}
    return {name: TensorSample.of(value, next(n for n, v in tensors.items() if v is value), next(
        n for n, v in tensors.items() if v is value or storages[name] is not None and storages[n] == storages[name]))
        if name in tensors else type(value) for name, value in arguments.items()}


@dataclass
class OperandSamples:
    """``operands``: per leaf call, per trace batch reached, its bound operands (a ``TensorSample`` for a strided
    tensor, the value's type for anything else); ``types``: every node's value type at batch 1."""

    operands: Dict[fx.Node, List[Dict[str, Any]]] = field(default_factory=dict)
    types: Dict[fx.Node, type] = field(default_factory=dict)


class _Sampler(fx.Interpreter):
    def __init__(self, gm: fx.GraphModule, calls: Set[fx.Node], samples: OperandSamples, first: bool):
        super().__init__(gm)
        self.calls, self.samples, self.first = calls, samples, first

    def run_node(self, n: fx.Node) -> Any:
        if n in self.calls:
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            code = code_call(n.target)
            try:
                bound = None if isinstance(code, str) else code.bind(args, kwargs)
            except TypeError:
                bound = None
            if bound is not None:
                self.samples.operands.setdefault(n, []).append(_row(bound.arguments))
        value = super().run_node(n)
        if self.first:
            self.samples.types[n] = type(value)
        return value


def sample_operands(gm: fx.GraphModule, calls: Sequence[fx.Node], input_shape: Tuple[int, ...],
                    device: Any) -> OperandSamples:
    """The operands of ``calls`` when ``gm`` runs at each trace batch on seeded inputs of ``input_shape`` (the
    model's parameter dtype, as the tracer's shape propagation takes it)."""
    samples, dtype = OperandSamples(), example_dtype(gm)
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    rest = [node.args[0] if node.args else None for node in placeholders[1:]]
    for batch in TRACE_BATCHES:
        generator = torch.Generator().manual_seed(batch)
        x = torch.randn((batch, *input_shape), generator=generator, dtype=dtype).to(device)
        sampler = _Sampler(gm, set(calls), samples, batch == TRACE_BATCHES[0])
        with rng_neutral(gm), torch.no_grad():
            guarded(lambda: sampler.run(x, *rest))
    return samples


def example_dtype(gm: fx.GraphModule) -> torch.dtype:
    """The dtype of a seeded input: the model's parameter dtype, as the tracer's shape propagation takes it."""
    parameter = next(gm.parameters(), None)
    return parameter.dtype if parameter is not None else torch.get_default_dtype()


def affine_fit(values: Sequence[int]) -> Optional[Tuple[int, int]]:
    """``(a, c)`` with ``a*B + c`` the value at each of ``TRACE_BATCHES``; ``None`` when they are not affine."""
    (b0, v0), (b1, v1) = list(zip(TRACE_BATCHES, values))[:2]
    if (v1 - v0) % (b1 - b0):
        return None
    slope = (v1 - v0) // (b1 - b0)
    found = (slope, v0 - slope * b0)
    return found if all(v == found[0] * b + found[1] for b, v in zip(TRACE_BATCHES, values)) else None


def sample_at(rows: Sequence[TensorSample], batch: int) -> Optional[TensorSample]:
    """An operand's metadata at ``batch``: its sample at a trace batch, else every extent, stride and its offset
    continued affinely through the samples (``None`` when one is not affine there, or is negative at ``batch``)."""
    if batch in TRACE_BATCHES:
        return rows[TRACE_BATCHES.index(batch)]
    fits = [affine_fit(column) for column in (*zip(*(r.shape for r in rows)), *zip(*(r.stride for r in rows)),
                                              tuple(r.offset for r in rows))]
    if any(fit is None for fit in fits):
        return None
    values = [fit[0] * batch + fit[1] for fit in fits if fit is not None]
    rank = len(rows[0].shape)
    if min(values, default=0) < 0:
        return None
    return dataclasses.replace(rows[0], shape=tuple(values[:rank]), stride=tuple(values[rank:2 * rank]),
                               offset=values[-1])


def _stand_in(sample: TensorSample) -> torch.Tensor:
    """An empty tensor with ``sample``'s metadata (a fresh storage: the capture reads no data)."""
    if sample.offset == 0:
        value = torch.empty_strided(sample.shape, sample.stride, dtype=sample.dtype, device=sample.device)
    else:
        span = sample.offset + sum((n - 1) * s for n, s in zip(sample.shape, sample.stride)) + 1
        value = torch.empty(span, dtype=sample.dtype, device=sample.device).as_strided(
            sample.shape, sample.stride, sample.offset)
    return value.requires_grad_(sample.requires_grad) if sample.requires_grad else value


def stand_ins(samples: Mapping[str, Sequence[TensorSample]], at: int) -> Dict[str, torch.Tensor]:
    """A stand-in per operand at sample ``at``, one object for the operands that are one tensor natively (its
    ``same_object``): ``a is b`` in the capture answers as for the native call (review round 2: two stand-ins for
    ``f(h, h)`` answered it False)."""
    found: Dict[str, torch.Tensor] = {}
    for name, rows in samples.items():
        first = rows[at].same_object or name
        found[name] = found[first] if first in found else _stand_in(rows[at])
    return found


def identity_reason(samples: Mapping[str, Sequence[TensorSample]]) -> Optional[str]:
    """Why the operands' identities keep a capture from answering as natively: an operand that is another's tensor or
    storage at some batches only, or a distinct tensor of another's storage (stand-ins share no storage)."""
    for name, rows in samples.items():
        found = {(row.same_object or name, row.same_storage or name) for row in rows}
        if len(found) > 1:
            return f"its operand {name!r} is another operand's tensor or storage at some batches only"
        (same, storage), = found
        if storage != same:
            return (f"its operands {storage!r} and {name!r} are distinct tensors of one storage, which the capture's "
                    "stand-ins do not share")
    return None
