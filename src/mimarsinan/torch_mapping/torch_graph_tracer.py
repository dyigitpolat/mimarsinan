"""
FX-based graph extraction with shape propagation for native PyTorch models.

Uses ``torch.fx.symbolic_trace`` to produce a flat computational graph and
``ShapeProp`` to annotate every node with concrete tensor shapes.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp


# Modules that should never be traced into -- they are kept as opaque
# call_module nodes so the analyzer / converter can handle them explicitly.
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


def trace_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device | str = "cpu",
) -> fx.GraphModule:
    """Trace a native PyTorch model and annotate shapes.

    Args:
        model: The model to trace.  Must be symbolically traceable
            (no data-dependent control flow).
        input_shape: Shape of a single input sample *without* batch dim,
            e.g. ``(3, 32, 32)`` for CIFAR images.
        device: Device used for the shape-propagation forward pass.

    Returns:
        A ``torch.fx.GraphModule`` whose nodes carry ``meta['tensor_meta']``
        with shape / dtype information from ShapeProp.

    Raises:
        TracingError: If the model cannot be symbolically traced.
    """
    model = model.eval().to(device)

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

    return graph_module
