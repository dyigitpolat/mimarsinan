"""Mark segment-boundary perceptrons for host-side encoding (ComputeOp) mapping."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper, ComputeOpMapper
from mimarsinan.mapping.mappers.conv import (
    Conv2DPerceptronMapper,
    Conv1DPerceptronMapper,
)
from mimarsinan.mapping.mappers.structural import InputMapper


_PERCEPTRON_MAPPER_TYPES = (PerceptronMapper, Conv2DPerceptronMapper, Conv1DPerceptronMapper)


def _is_perceptron_holder(node) -> bool:
    return isinstance(node, _PERCEPTRON_MAPPER_TYPES)


def _wraps_unbounded_raw_linear_or_conv(mapper) -> bool:
    """True if a ``ComputeOpMapper`` wraps a bare Linear/Conv (signed, unbounded output)."""
    module = getattr(mapper, "module", None)
    if module is None:
        return False
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return True
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return isinstance(module[0], (nn.Linear, nn.Conv1d, nn.Conv2d))
    return False


def _is_encoding_segment_start(node) -> bool:
    """True iff the upstream chain starts at raw input or unbounded host output.

    A perceptron whose source produces signed / unbounded values (raw Linear /
    Conv ComputeOp, or the raw network input) cannot be fed spikes directly —
    its forward must run host-side as a ComputeOp.  Structural mappers and
    bounded-output ComputeOps (LayerNorm, pool, GELU, ...) are transparent;
    upstream perceptrons stop the walk (on-chip spike output is fine).
    """
    src = node.source_mapper
    while src is not None:
        if isinstance(src, _PERCEPTRON_MAPPER_TYPES):
            return False
        if isinstance(src, InputMapper):
            return True
        if isinstance(src, ComputeOpMapper) and _wraps_unbounded_raw_linear_or_conv(src):
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
