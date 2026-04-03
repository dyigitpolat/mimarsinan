"""Mark segment-boundary perceptrons for host-side encoding (ComputeOp) mapping."""

from __future__ import annotations

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.mappers.conv import (
    Conv2DPerceptronMapper,
    Conv1DPerceptronMapper,
    Conv2DMapper,
    Conv1DMapper,
)
from mimarsinan.mapping.mappers.pooling import (
    MaxPool2DMapper,
    AvgPool2DMapper,
    AdaptiveAvgPool2DMapper,
)
from mimarsinan.mapping.mappers.structural import InputMapper


_PERCEPTRON_MAPPER_TYPES = (PerceptronMapper, Conv2DPerceptronMapper, Conv1DPerceptronMapper)

# Bare conv mappers (chip-style conv without Perceptron wrapper) — upstream neural, not raw input.
_BARE_CONV_MAPPER_TYPES = (Conv2DMapper, Conv1DMapper)

# Pooling maps to ComputeOp in IR — next conv/FC perceptron starts a new segment.
# (Do not list ModuleComputeMapper: it is also used for patch Linear / token-mix Linears.)
_BOUNDARY_MAPPER_TYPES = (
    MaxPool2DMapper,
    AvgPool2DMapper,
    AdaptiveAvgPool2DMapper,
)


def _is_perceptron_holder(node) -> bool:
    return isinstance(node, _PERCEPTRON_MAPPER_TYPES)


def _is_encoding_segment_start(node) -> bool:
    """True if this perceptron starts a segment (raw input or after a ComputeOp boundary).

    Walks ``source_mapper`` upward: structural mappers (Einops, reshape, etc.) are
    transparent. Stops at:

    * Another perceptron mapper → not an encoding start.
    * ``InputMapper`` → encoding (first neural op from raw input).
    * Pooling / ``ModuleComputeMapper`` → encoding (after host-side ComputeOp).
    """
    src = node.source_mapper
    while src is not None:
        if isinstance(src, _PERCEPTRON_MAPPER_TYPES):
            return False
        if isinstance(src, _BARE_CONV_MAPPER_TYPES):
            return False
        if isinstance(src, InputMapper):
            return True
        if isinstance(src, _BOUNDARY_MAPPER_TYPES):
            return True
        src = src.source_mapper
    return False


def mark_encoding_layers(model_repr: ModelRepresentation) -> None:
    """Set ``perceptron.is_encoding_layer`` on perceptrons that start a neural segment."""
    model_repr._ensure_exec_graph()
    for node in model_repr._exec_order:
        if not _is_perceptron_holder(node):
            continue
        if _is_encoding_segment_start(node):
            node.perceptron.is_encoding_layer = True
