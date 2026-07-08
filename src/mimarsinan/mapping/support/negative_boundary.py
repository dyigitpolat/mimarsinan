"""Negative value-boundary policy: calibrated shift (on) or subsume-forward (off)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.bias_compensation import apply_negative_value_shifts
from mimarsinan.mapping.support.value_domain import node_absorbs_negative_values

__all__ = [
    "NegativeBoundaryResult",
    "apply_negative_boundary_policy",
    "boundary_consumers",
    "calibrated_compute_op_minima",
    "lossy_negative_boundaries",
    "subsume_forward_negative_boundaries",
]

# The [0,1] spike-encode clamp only loses information below this floor.
NEGATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class NegativeBoundaryResult:
    """What the chosen mechanism did, and the minima it was decided from."""

    minima: Dict[Any, torch.Tensor] = field(default_factory=dict)
    shifts: Dict[Any, Any] = field(default_factory=dict)
    subsumed: List[Any] = field(default_factory=list)


def _perceptron_of(node):
    return getattr(node, "perceptron", None)


def _is_host_node(node) -> bool:
    """A node whose output is produced on the host, in the value domain."""
    if isinstance(node, ComputeOpMapper):
        return True
    perceptron = _perceptron_of(node)
    return perceptron is not None and bool(
        getattr(perceptron, "is_encoding_layer", False)
    )


def _consumer_map(mapper_repr) -> Dict[int, list]:
    mapper_repr._ensure_exec_graph()
    consumers: Dict[int, list] = {}
    for node in mapper_repr._exec_order:
        for dep in mapper_repr._deps.get(node, []):
            consumers.setdefault(id(dep), []).append(node)
    return consumers


def boundary_consumers(node, consumers: Dict[int, list]) -> list:
    """The first non-structural consumers of ``node``.

    Structural nodes (reshape/permute/…) neither change the sign nor encode, so
    the boundary a value crosses is defined by the first perceptron or host
    ComputeOp downstream of them.
    """
    found, frontier = [], list(consumers.get(id(node), []))
    seen: set[int] = set()
    while frontier:
        candidate = frontier.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, ComputeOpMapper) or _perceptron_of(candidate):
            found.append(candidate)
        else:
            frontier.extend(consumers.get(id(candidate), []))
    return found


def calibrated_compute_op_minima(
    model, calibration_x: torch.Tensor, T: int, *, forward_fn,
) -> Dict[Any, torch.Tensor]:
    """Per-ComputeOp minima of the NF boundary values on the calibration set.

    A ComputeOp-free graph has no value boundary at all, so it skips the
    calibration forward entirely — a structural no-op for both mechanisms.
    """
    mapper_repr = model.get_mapper_repr()
    mapper_repr._ensure_exec_graph()
    if not any(isinstance(n, ComputeOpMapper) for n in mapper_repr._exec_order):
        return {}
    recorder: Dict[Any, torch.Tensor] = {}
    with torch.no_grad():
        forward_fn(model, calibration_x, T, compute_min_recorder=recorder)
    return recorder


def _effective_minimum(op, mins: torch.Tensor) -> torch.Tensor:
    """The boundary minimum an encoder actually sees: the calibrated minimum
    plus any shift the ON mechanism baked onto this op."""
    shift = getattr(op, "_negative_shift", None)
    if shift is None:
        return mins
    return mins + torch.as_tensor(shift, dtype=mins.dtype, device=mins.device)


def lossy_negative_boundaries(model, minima: Dict[Any, torch.Tensor]) -> list:
    """ComputeOps whose EFFECTIVE boundary value goes negative while an on-chip
    segment encodes it — the exact precondition of silent clamp corruption.

    Host-only consumers (another ComputeOp, a subsumed perceptron) run in the
    value domain and clamp nothing, so they are never a lossy boundary.
    """
    consumers = _consumer_map(model.get_mapper_repr())
    lossy = []
    for op, mins in minima.items():
        if float(_effective_minimum(op, mins).min()) >= -NEGATIVE_TOLERANCE:
            continue
        if any(
            not _is_host_node(consumer)
            for consumer in boundary_consumers(op, consumers)
        ):
            lossy.append(op)
    return lossy


def subsume_forward_negative_boundaries(
    model, minima: Dict[Any, torch.Tensor],
) -> list:
    """Move consuming perceptrons onto the host until a non-negative-value-
    generating node absorbs the signed range (``negative_value_shift=off``).

    A subsumed perceptron runs host-side in the value domain — exact float
    math, no encode — so its own output becomes the new boundary. The walk
    stops at the first node that structurally cannot emit a negative value
    (ReLU, LIF, a non-negative clamp); a host ComputeOp that absorbs nothing
    is crossed, because it clamps nothing either. Returns the newly hosted
    perceptrons, in exec order. Idempotent.
    """
    mapper_repr = model.get_mapper_repr()
    consumers = _consumer_map(mapper_repr)
    order = {id(n): i for i, n in enumerate(mapper_repr._exec_order)}

    subsumed = []
    for op in lossy_negative_boundaries(model, minima):
        frontier = boundary_consumers(op, consumers)
        seen: set[int] = set()
        while frontier:
            node = frontier.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            perceptron = _perceptron_of(node)
            if perceptron is not None and not perceptron.is_encoding_layer:
                perceptron.is_encoding_layer = True
                subsumed.append((order[id(node)], perceptron))
            if node_absorbs_negative_values(node):
                continue
            frontier.extend(boundary_consumers(node, consumers))

    if subsumed and not any(
        not p.is_encoding_layer for p in model.get_perceptrons()
    ):
        raise NotImplementedError(
            "negative-boundary subsume-forward left no on-chip segment: every "
            "perceptron had to move to the host because nothing downstream of "
            "the negative ComputeOp boundary absorbs a signed range (no ReLU / "
            "LIF / non-negative clamp). This topology is not deployable with "
            "negative_value_shift=off — enable the calibrated shift, or give "
            "the boundary a non-negative-value-generating consumer."
        )
    return [p for _, p in sorted(subsumed, key=lambda pair: pair[0])]


def apply_negative_boundary_policy(
    model, calibration_x: torch.Tensor, T: int, *, shift_enabled: bool, forward_fn,
) -> NegativeBoundaryResult:
    """Make every negative ComputeOp→neural boundary lossless, then PROVE it.

    ``shift_enabled`` picks the mechanism; the post-condition is the same for
    both and is re-checked from the calibrated minima: no negative boundary may
    remain on-chip-encoded. A silently corrupting deployment is therefore not
    authorable — an unfixable topology raises here instead.
    """
    minima = calibrated_compute_op_minima(
        model, calibration_x, T, forward_fn=forward_fn,
    )
    shifts: Dict[Any, Any] = {}
    subsumed: List[Any] = []
    if shift_enabled:
        shifts = apply_negative_value_shifts(model, minima)
    else:
        subsumed = subsume_forward_negative_boundaries(model, minima)

    remaining = lossy_negative_boundaries(model, minima)
    if remaining:
        names = [getattr(op, "name", None) or repr(op) for op in remaining]
        mechanism = "calibrated shift" if shift_enabled else "subsume-forward"
        raise NotImplementedError(
            f"negative-boundary policy ({mechanism}) left {len(remaining)} "
            f"boundary/boundaries negative while an on-chip segment encodes "
            f"them: {names}. The [0,1] spike-encode clamp would silently drop "
            f"the range."
        )
    return NegativeBoundaryResult(minima=minima, shifts=shifts, subsumed=subsumed)
