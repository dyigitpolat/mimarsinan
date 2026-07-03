"""Mark segment-boundary perceptrons for host-side encoding (ComputeOp) mapping."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
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

    Such a perceptron's source produces signed/unbounded values that cannot be fed
    as spikes, so its forward must run host-side as a ComputeOp.
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


_VALID_PLACEMENTS = ("subsume", "offload")


def mark_encoding_layers(
    model_repr: ModelRepresentation, *, placement: str = "subsume",
) -> None:
    """Set ``perceptron.is_encoding_layer`` on perceptrons that start a neural segment.

    ``placement="subsume"`` marks segment-start perceptrons as host ComputeOps that
    generate spike trains; ``"offload"`` clears the mark so they map on-chip as NeuralCores.
    """
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"mark_encoding_layers placement must be one of {_VALID_PLACEMENTS!r}; "
            f"got {placement!r}"
        )
    model_repr._ensure_exec_graph()
    exec_order = model_repr._exec_order
    assert exec_order is not None  # populated by _ensure_exec_graph
    for node in exec_order:
        if not _is_perceptron_holder(node):
            continue
        # Idempotent per placement: offload clears any prior subsume marking so the perceptron maps on-chip.
        if placement == "offload":
            node.perceptron.is_encoding_layer = False
        elif _is_encoding_segment_start(node):
            node.perceptron.is_encoding_layer = True


def segment_entry_perceptrons(model_repr: ModelRepresentation) -> list:
    """Perceptrons that are the FIRST on-chip core of a neural segment.

    These read a freshly assembled hybrid stage input — the seam the synchronized
    TTFS wire contract grid-quantizes. Structural mappers are transparent in the walk.
    """
    model_repr._ensure_exec_graph()
    exec_order = model_repr._exec_order
    assert exec_order is not None  # populated by _ensure_exec_graph
    entries = []
    for node in exec_order:
        if not _is_perceptron_holder(node):
            continue
        if getattr(node.perceptron, "is_encoding_layer", False):
            continue
        src = node.source_mapper
        while src is not None:
            if _is_perceptron_holder(src):
                if getattr(src.perceptron, "is_encoding_layer", False):
                    entries.append(node.perceptron)
                break
            if isinstance(src, (InputMapper, ComputeOpMapper)):
                entries.append(node.perceptron)
                break
            src = src.source_mapper
    return entries
