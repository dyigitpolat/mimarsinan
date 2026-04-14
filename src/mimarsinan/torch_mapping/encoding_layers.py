"""Mark segment-boundary perceptrons for host-side encoding (ComputeOp) mapping."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper, ModuleComputeMapper
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
_BOUNDARY_MAPPER_TYPES = (
    MaxPool2DMapper,
    AvgPool2DMapper,
    AdaptiveAvgPool2DMapper,
)


def _is_perceptron_holder(node) -> bool:
    return isinstance(node, _PERCEPTRON_MAPPER_TYPES)


def _wraps_unbounded_raw_linear_or_conv(mapper) -> bool:
    """True if a ``ModuleComputeMapper`` wraps a bare ``nn.Linear``/``Conv`` (or
    ``nn.Sequential`` starting with one) — i.e. its output is raw, signed,
    unbounded values that need a trailing activation before spike encoding.

    Bounded-output modules (pool, layernorm, function wrappers) return False —
    the walk continues past them as before.
    """
    module = getattr(mapper, "module", None)
    if module is None:
        return False
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return True
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return isinstance(module[0], (nn.Linear, nn.Conv1d, nn.Conv2d))
    return False


def _is_encoding_segment_start(node) -> bool:
    """True if this perceptron starts a segment — its input is not a clean
    spike stream and so its forward must run host-side as a ComputeOp.

    Walks ``source_mapper`` upward: structural mappers (Einops, reshape, etc.)
    are transparent. Stops at:

    * Another perceptron mapper → not an encoding start (upstream is on-chip
      spike output).
    * ``InputMapper`` → encoding (first neural op from raw input).
    * Pooling boundary → encoding (after host-side ComputeOp).
    * ``ModuleComputeMapper`` wrapping a bare Linear/Conv → encoding.
      That upstream ComputeOp produces unbounded raw values, so this
      Perceptron cannot consume them on-chip (spike encoding undefined);
      it must run host-side where the activation closes the segment.
    * ``ModuleComputeMapper`` wrapping a bounded op (pool, layernorm,
      generic function wrapper) → transparent; keep walking.
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
        if isinstance(src, ModuleComputeMapper) and _wraps_unbounded_raw_linear_or_conv(src):
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
