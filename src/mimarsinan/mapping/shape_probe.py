"""Single-source shape inference for host-side ``nn.Module`` ComputeOps.

The mapper graph emits one ``ComputeOp`` per non-neural op, each carrying a
batch-stripped ``input_shape`` / ``output_shape``.  Historically every
specialized mapper recomputed those shapes from constructor params
(stride/padding arithmetic, weight introspection, sequence-length math).
This module replaces that with a single zeros-tensor forward pass through
the wrapped module — the same trick the chip simulator's
``_get_compute_op_output_size`` and ``torch.fx.passes.shape_prop.ShapeProp``
already use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ProbedShapes:
    """Result of a single ``probe_module_io_shapes`` call.

    All shapes are batch-stripped.  ``input_shapes`` is always a list; the
    convenience property ``input_shape`` returns the sole entry when the
    probe was run against a unary module.
    """
    input_shapes: Tuple[Tuple[int, ...], ...]
    output_shape: Tuple[int, ...]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if len(self.input_shapes) != 1:
            raise ValueError(
                "ProbedShapes.input_shape is only defined for unary modules; "
                f"got {len(self.input_shapes)} inputs.  Use .input_shapes."
            )
        return self.input_shapes[0]


def _normalize_input_shapes(
    input_shape,
) -> Tuple[Tuple[int, ...], ...]:
    """Accept either a single shape tuple or a sequence of shapes."""
    if input_shape is None:
        raise ValueError("probe_module_io_shapes: input_shape must not be None")
    if len(input_shape) == 0:
        raise ValueError("probe_module_io_shapes: input_shape must not be empty")
    first = input_shape[0]
    if isinstance(first, (tuple, list)):
        return tuple(tuple(int(d) for d in s) for s in input_shape)
    return (tuple(int(d) for d in input_shape),)


def probe_module_io_shapes(
    module: nn.Module,
    input_shape,
    *,
    module_kwargs: Mapping[str, Any] | None = None,
    output_index: int | None = None,
) -> ProbedShapes:
    """Run a single zeros-tensor forward pass to infer ``module``'s output shape.

    ``input_shape`` is batch-stripped — the probe injects batch=1.  Pass a
    list/tuple of shape tuples for multi-tensor modules (e.g.
    ``nn.MultiheadAttention``).  Returns a frozen :class:`ProbedShapes`
    with all shapes batch-stripped.

    The module is run in ``.eval()`` under ``torch.no_grad()`` on the
    same device as its first parameter (CPU if none), with training
    state restored on exit.  Zero input is deliberate: cheap, deterministic,
    and matches the convention used by ``ShapeProp`` and the existing
    chip-simulation shape probe.
    """
    input_shapes = _normalize_input_shapes(input_shape)

    try:
        device = next(module.parameters()).device
    except StopIteration:
        try:
            device = next(module.buffers()).device
        except StopIteration:
            device = torch.device("cpu")

    dummy_inputs = tuple(
        torch.zeros((1, *shape), dtype=torch.float32, device=device)
        for shape in input_shapes
    )

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            out = module(*dummy_inputs, **(module_kwargs or {}))
            if output_index is not None:
                out = out[output_index]
    finally:
        if was_training:
            module.train()

    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"probe_module_io_shapes: module {type(module).__name__} returned "
            f"{type(out).__name__}, expected torch.Tensor.  Pass output_index "
            "if the module returns a tuple."
        )

    output_shape = tuple(int(d) for d in out.shape[1:])  # strip batch
    return ProbedShapes(input_shapes=input_shapes, output_shape=output_shape)
