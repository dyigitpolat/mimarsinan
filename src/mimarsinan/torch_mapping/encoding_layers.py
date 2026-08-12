"""Mark segment-boundary perceptrons for host-side encoding (ComputeOp) mapping."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.platform.packaging_contract import (
    SPIKING_PACKAGING,
    PackagingContract,
)
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

#: The stamp a graph carries when ``encoding_layer_placement`` has no meaning
#: for its target family. A value-domain (MVM) core consumes VALUES, so there
#: is no spike-train encoder to place anywhere — the registry even makes the
#: key unauthorable there (``domain="event"``). It is its own value, never a
#: bare ``None``: ``None`` means "nobody applied the configured placement",
#: which is the silent no-op this module exists to make loud.
PLACEMENT_NOT_APPLICABLE = "not_applicable"


def encoder_deploys_as_staircase_hop(placement: str) -> bool:
    """True when marked encoders deploy as host ops running their own staircased module.

    ``subsume`` runs the encoder perceptron itself host-side — installed ceil
    staircase included — a floor-quantizer hop like every on-chip core;
    ``offload`` host-encodes raw input with the mid-tread round (``ttfs_spike_time``).
    """
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"encoder_deploys_as_staircase_hop placement must be one of "
            f"{_VALID_PLACEMENTS!r}; got {placement!r}"
        )
    return placement == "subsume"


class UnresolvedEncodingPlacementError(ValueError):
    """A flow reached a placement-sensitive consumer without a resolved placement.

    Either nothing ever applied ``encoding_layer_placement`` to it (the marking
    answers no configured question), or it was resolved under a DIFFERENT
    placement than the caller is asking about (so the answer would describe a
    mapping that will not deploy).
    """


def mark_encoding_layers(
    model_repr: ModelRepresentation,
    *,
    placement: str = "subsume",
    packaging: PackagingContract = SPIKING_PACKAGING,
) -> None:
    """Set ``perceptron.is_encoding_layer`` on perceptrons that start a neural segment.

    ``placement="subsume"`` marks segment-start perceptrons as host ComputeOps that
    generate spike trains; ``"offload"`` clears the mark so they map on-chip as NeuralCores.
    ``packaging`` is the target family's contract: a VALUE-domain target has no
    spike-train encoder at all, so nothing is marked and the graph is stamped
    :data:`PLACEMENT_NOT_APPLICABLE` — an explicit answer, not an absent one.

    THE one writer of the placement decision, and it records the decision on the
    graph (:func:`resolved_encoding_placement`) so a consumer can tell a resolved
    marking from an unresolved one. Call it once, at flow birth — ``build_model``
    for a builder that returns a flow, ``convert_torch_model`` for a torch module.
    Consumers read, never re-mark: the negative-boundary subsume-forward policy
    adds host placements AFTER birth, and only the perceptrons placement OWNS
    (the encoding-segment starts) are written here, so anything this walk does
    not touch survives.
    """
    if placement not in _VALID_PLACEMENTS:
        raise ValueError(
            f"mark_encoding_layers placement must be one of {_VALID_PLACEMENTS!r}; "
            f"got {placement!r}"
        )
    if packaging.is_value_domain:
        model_repr.encoding_placement = PLACEMENT_NOT_APPLICABLE
        return
    model_repr._ensure_exec_graph()
    exec_order = model_repr._exec_order
    assert exec_order is not None  # populated by _ensure_exec_graph
    host_side = placement == "subsume"
    for node in exec_order:
        # Scoped to what placement owns: a perceptron that does NOT start an
        # encoding segment is never placement's to move, so a host mark another
        # writer put there (negative-boundary subsume-forward) is left alone.
        if not _is_perceptron_holder(node) or not _is_encoding_segment_start(node):
            continue
        node.perceptron.is_encoding_layer = host_side
    model_repr.encoding_placement = placement


def resolved_encoding_placement(model_repr: ModelRepresentation) -> str | None:
    """The placement this graph's encoder marking was resolved under, or ``None``."""
    return getattr(model_repr, "encoding_placement", None)


def require_resolved_encoding_placement(
    model_repr: ModelRepresentation, placement: str, *, context: str
) -> None:
    """Raise unless this graph's marking already answers ``placement``.

    The guard that keeps a placement no-op loud: a flow whose encoders were never
    placed (or were placed differently) cannot be measured, mapped or deployed as
    if it honored the configured knob. A value-domain graph answers BOTH
    placements — it has no encoder, so neither value would change its mapping.
    """
    resolved = resolved_encoding_placement(model_repr)
    if resolved == placement or resolved == PLACEMENT_NOT_APPLICABLE:
        return
    if resolved is None:
        raise UnresolvedEncodingPlacementError(
            f"{context}: this flow's encoding_layer_placement was never resolved, "
            f"so its encoder marking answers no configured question and cannot be "
            f"read as the {placement!r} deployment. Build it through "
            f"models.builders.build_model (or convert_torch_model) so the "
            f"placement is applied once, at flow birth."
        )
    raise UnresolvedEncodingPlacementError(
        f"{context}: this flow was built for encoding_layer_placement {resolved!r} "
        f"but is being read as {placement!r}. The marking on the graph is the "
        f"{resolved!r} one, so the answer would describe a mapping that will not "
        f"deploy — build a fresh flow for {placement!r} instead."
    )


def resolve_unstamped_encoding_placement(
    model_repr: ModelRepresentation,
    *,
    placement: str,
    packaging: PackagingContract = SPIKING_PACKAGING,
) -> bool:
    """Apply the configured placement to a flow that carries NO stamp; else leave it.

    The resume path. A cached flow written before the stamp existed
    deserializes unstamped, and an unstamped flow is exactly what the guard
    refuses — so a pre-change run directory could not be resumed at all. The
    fix is to APPLY the placement (not to wave the flow through): a legacy
    native flow really never had one applied.

    Safe here and nowhere later: this runs at cache load, before any step of
    this run executes, so no host placement from the negative-boundary
    subsume-forward policy exists yet in this process — and the marking is
    scoped to the encoding-segment starts anyway, so a subsume-forward mark an
    earlier process left on a perceptron placement does not own survives.
    Returns whether it applied anything.
    """
    if resolved_encoding_placement(model_repr) is not None:
        return False
    mark_encoding_layers(model_repr, placement=placement, packaging=packaging)
    return True


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
