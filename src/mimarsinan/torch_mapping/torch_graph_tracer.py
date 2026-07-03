"""FX-based graph extraction with shape propagation for native PyTorch models."""

from __future__ import annotations

import threading
from typing import Tuple

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


_LEAF_MODULES: Tuple[type, ...] = (
    nn.MultiheadAttention,
    nn.LSTM,
    nn.GRU,
    nn.RNN,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
)


class _MimarsinanTracer(fx.Tracer):
    """Custom tracer that treats certain modules as leaves."""

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, _LEAF_MODULES):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class TracingError(Exception):
    """Raised when a model cannot be symbolically traced."""


# FX patches nn.Module.__call__/__getattr__ non-atomically; capture the true originals at import time and serialize tracing so a stale cross-thread wrapper cannot leak.
_TRACE_LOCK = threading.Lock()
_ORIG_MODULE_CALL = nn.Module.__call__
_ORIG_MODULE_GETATTR = nn.Module.__getattr__


def _ensure_module_call_unpatched() -> None:
    """If a stale FX wrapper is still installed on nn.Module, restore originals."""
    current_call = nn.Module.__call__
    if getattr(current_call, "__fx_already_patched", False):
        nn.Module.__call__ = _ORIG_MODULE_CALL
    current_getattr = nn.Module.__getattr__
    if getattr(current_getattr, "__fx_already_patched", False):
        nn.Module.__getattr__ = _ORIG_MODULE_GETATTR


def trace_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device | str = "cpu",
) -> fx.GraphModule:
    """Trace a native PyTorch model and annotate node shapes via ShapeProp.

    ``input_shape`` excludes the batch dim, e.g. ``(3, 32, 32)``.
    Raises ``TracingError`` if the model is not symbolically traceable.
    """
    model = model.eval().to(device)

    with _TRACE_LOCK:
        _ensure_module_call_unpatched()
        try:
            try:
                tracer = _MimarsinanTracer()
                graph = tracer.trace(model)
                graph_module = fx.GraphModule(model, graph)
            except Exception as exc:
                raise TracingError(
                    f"Failed to symbolically trace the model "
                    f"({type(model).__name__}): {exc}\n"
                    f"Models with data-dependent control flow (if/for on tensor "
                    f"values) cannot be traced.  Consider refactoring or wrapping "
                    f"dynamic sections in leaf modules."
                ) from exc

            example_input = torch.randn(1, *input_shape, device=device)
            try:
                ShapeProp(graph_module).propagate(example_input)
            except Exception as exc:
                raise TracingError(
                    f"Shape propagation failed for {type(model).__name__}: {exc}\n"
                    f"Ensure the model accepts input of shape (1, {', '.join(str(d) for d in input_shape)})."
                ) from exc
        finally:
            _ensure_module_call_unpatched()

    return graph_module
