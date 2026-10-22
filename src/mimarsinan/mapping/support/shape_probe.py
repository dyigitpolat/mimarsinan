"""Zeros-tensor shape inference for host-side ``nn.Module`` ComputeOps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

import torch
import torch.nn as nn

from mimarsinan.common.refusal import ToolchainRefusal


class NonTensorHostValueError(ToolchainRefusal, TypeError):
    """A host op whose value is not a tensor (a run-time shape value): the IR carries tensor values only."""

    def __init__(self, owner: str | None, module: nn.Module, value: object, operator: str | None = None):
        self.owner = owner
        self.operator = operator
        self.value_type = type(value).__name__
        if operator:
            where = f"host op {operator} (FX node {owner!r})"
        else:
            where = f"host op {owner!r} ({type(module).__name__})" if owner else f"module {type(module).__name__}"
        super().__init__(
            f"probe_module_io_shapes: {where} returned {self.value_type}, expected torch.Tensor; the IR carries "
            "tensor values only, so a value computed at run time from the batch size has no IR form and the "
            "lowering is refused. Pass output_index if the module returns a tuple."
        )


class HostValueBatchLayoutError(ToolchainRefusal, ValueError):
    """A host op whose value does not carry the batch on axis 0 alone: an IR buffer holds one row per sample."""

    def __init__(self, owner: str | None, violation: str, operator: str | None = None):
        self.owner = owner
        self.operator = operator
        self.violation = violation
        named = f"{operator} " if operator else ""
        super().__init__(
            f"host op {named}(FX node {owner!r}): in its value, {violation}; a value program's buffer holds one row "
            "per sample, so the value has no buffer form and the lowering is refused"
        )


@dataclass(frozen=True)
class ProbedShapes:
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


def _normalize_input_shapes(input_shape) -> Tuple[Tuple[int, ...], ...]:
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
    owner: str | None = None,
    operator: str | None = None,
) -> ProbedShapes:
    """Run one zeros-tensor forward pass through ``module``; return batch-stripped IO shapes (``owner`` names its FX
    node and ``operator`` the operator it applies, in a refusal)."""
    input_shapes = _normalize_input_shapes(input_shape)

    try:
        ref = next(module.parameters())
    except StopIteration:
        try:
            ref = next(module.buffers())
        except StopIteration:
            ref = None
    device = ref.device if ref is not None else torch.device("cpu")
    dtype = ref.dtype if ref is not None and ref.dtype.is_floating_point else torch.float32

    dummy_inputs = tuple(
        torch.zeros((1, *shape), dtype=dtype, device=device)
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
        raise NonTensorHostValueError(owner, module, out, operator)

    output_shape = tuple(int(d) for d in out.shape[1:])
    return ProbedShapes(input_shapes=input_shapes, output_shape=output_shape)
