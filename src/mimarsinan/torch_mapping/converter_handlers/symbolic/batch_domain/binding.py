"""A kernel call's operands in its signature's order (review round 16).

The declared batch rules read a kernel's tensor operands by position (``kernel_rules._tensors``: the attention rule's
query and key, a matrix product's two factors, a recurrent module's input). A call may pass them by keyword in any
order (``mha(key=kv, query=q, value=kv)``, ``torch.matmul(other=W, input=y)``, ``lstm(hx=(h0, c0), input=q)``), so
before a rule reads them the call is bound to the signature the call runs (``bound``): every operand passed by keyword
whose parameter can be positional moves into the positional operands, in the signature's order, up to the first
parameter the call leaves out; the remaining keyword operands follow in the signature's order. A call passed
positionally is unchanged.

The signature is the one the call runs: a module's ``forward`` (read as ``nn.Module``'s call reads it), a Python
function's own, and torch's schemas of a builtin function or a tensor method (``self`` named ``input``, as torch's FX
names it); the first that binds the call is used. A call no signature binds keeps its operands as passed, and when one
of its keyword operands holds a tensor, no rule reads it (``None``): its sizes are then opaque.

Torch's Python argument parser reads, for every builtin function and tensor method, numpy's names of four parameters
(``numpy_compatibility_arg_names``): ``axis`` for ``dim``, ``keepdims`` for ``keepdim``, ``x``, ``a`` and ``x1`` for
``input``, ``x2`` for ``other``, where the call does not pass the parameter's own name. ``canonical_keywords`` reads
them so, once for every node of the traced graph (``canonical_graph_keywords``, the tracer's) and in ``bound``, so
every rule and every handler of the converter reads the call with torch's own names (review round 16, targeted pass:
``torch.sum(y, axis=1)`` was read as a full reduction, ``torch.cat([y, y], axis=1)`` as one along axis 0).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.fx as fx
from torch.fx.operator_schemas import _torchscript_schema_to_signature, get_signature_for_torch_op

from mimarsinan.torch_mapping.conversion_probe import guarded
from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import DECLARED_TENSOR_TYPES

__all__ = ["bound", "canonical_graph_keywords", "canonical_keywords"]

_NUMPY_NAMES = {"axis": "dim", "keepdims": "keepdim", "x": "input", "a": "input", "x1": "input", "x2": "other"}
"""Torch's numpy-compatibility parameter names (``python_arg_parser``'s ``numpy_compatibility_arg_names``), each the
name of the schema parameter it stands for."""

_POSITIONAL = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)


@functools.lru_cache(maxsize=None)
def _method_signatures(name: str) -> Tuple[inspect.Signature, ...]:
    schemas, _ = guarded(lambda: torch._C._jit_get_schemas_for_operator(f"aten::{name}"))
    found = [guarded(lambda schema=schema: _torchscript_schema_to_signature(schema))[0] for schema in schemas or ()]
    return tuple(signature for signature in found if signature is not None)


def _schema_signatures(op: str, target: Any) -> List[inspect.Signature]:
    """Torch's schemas of a builtin function or a tensor method (none for a Python function or a module)."""
    if op == "call_method":
        return list(_method_signatures(str(target)))
    if op != "call_function" or guarded(lambda: inspect.signature(target))[0] is not None:
        return []
    schemas, _ = guarded(lambda: get_signature_for_torch_op(target))
    return [found for found in (schemas if isinstance(schemas, list) else []) if isinstance(found, inspect.Signature)]


def _signatures(op: str, target: Any, module: Any) -> List[inspect.Signature]:
    if op == "call_module":
        found, _ = guarded(lambda: inspect.signature(module.forward)) if module is not None else (None, None)
        return [found] if found is not None else []
    found, _ = guarded(lambda: inspect.signature(target)) if op == "call_function" else (None, None)
    return [found] if found is not None else _schema_signatures(op, target)


def canonical_keywords(op: str, target: Any, module: Any, args: tuple, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """``kwargs`` of a builtin function or a tensor method with torch's numpy-compatibility names read as the
    parameters they stand for (``_NUMPY_NAMES``), as the first of its schemas that binds the call so renamed reads
    them; ``kwargs`` as passed otherwise (a Python function, a module, a call no schema binds)."""
    if not any(name in _NUMPY_NAMES for name in kwargs):
        return dict(kwargs)
    for signature in _schema_signatures(op, target):
        names = signature.parameters
        renamed = {(_NUMPY_NAMES[name] if name in _NUMPY_NAMES and name not in names and _NUMPY_NAMES[name] in names
                    and _NUMPY_NAMES[name] not in kwargs else name): value for name, value in kwargs.items()}
        if len(renamed) == len(kwargs) and guarded(lambda: signature.bind(*args, **renamed))[0] is not None:
            return renamed
    return dict(kwargs)


def canonical_graph_keywords(gm: fx.GraphModule) -> None:
    """Every call of a builtin function, and every method call on a value whose run-time type (the tracer's shape
    propagation records it) is exactly a declared tensor type, reads its keyword operands with torch's own names."""
    for node in gm.graph.nodes:
        receiver = node.args[0] if node.args and isinstance(node.args[0], fx.Node) else None
        if node.op == "call_method" and (receiver is None or receiver.meta.get("type") not in DECLARED_TENSOR_TYPES):
            continue
        if node.op in ("call_function", "call_method") and node.kwargs:
            renamed = canonical_keywords(node.op, node.target, None, tuple(node.args), node.kwargs)
            if list(renamed) != list(node.kwargs):
                node.kwargs = renamed
    gm.recompile()


def _ordered(signature: inspect.Signature, arguments: Mapping[str, Any]) -> Tuple[tuple, Dict[str, Any]]:
    args: list = []
    kwargs: Dict[str, Any] = {}
    gap = False
    for name, parameter in signature.parameters.items():
        if name not in arguments:
            gap = gap or parameter.kind in _POSITIONAL
            continue
        value = arguments[name]
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(value)
        elif parameter.kind in _POSITIONAL and not gap:
            args.append(value)
        else:
            kwargs[name] = value
    return tuple(args), kwargs


def bound(op: str, target: Any, module: Any, args: tuple, kwargs: Mapping[str, Any],
          holds_tensor: Any) -> Optional[Tuple[tuple, Dict[str, Any]]]:
    """``(args, kwargs)`` of the call in its signature's order (the module docstring); the operands as passed when
    the call passes none by keyword; ``None`` when no signature binds a call with a tensor keyword operand
    (``holds_tensor``: the predicate of a domain value holding a tensor)."""
    if not kwargs:
        return tuple(args), {}
    kwargs = canonical_keywords(op, target, module, tuple(args), kwargs)
    for signature in _signatures(op, target, module):
        arguments, _ = guarded(lambda signature=signature: signature.bind(*args, **kwargs).arguments)
        if arguments is not None:
            return _ordered(signature, arguments)
    return None if any(holds_tensor(value) for value in kwargs.values()) else (tuple(args), dict(kwargs))
