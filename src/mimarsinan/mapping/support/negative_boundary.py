"""Negative value-boundary policy: calibrated shift (on) or subsume-forward (off)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.bias_compensation import (
    apply_negative_value_shifts,
    calibration_forward_for_mode,
)
from mimarsinan.mapping.support.value_domain import node_absorbs_negative_values
from mimarsinan.models.nn.activations.autograd import ChipInputQuantizer

__all__ = [
    "NegativeBoundaryResult",
    "apply_negative_boundary_policy",
    "boundary_consumers",
    "calibrated_compute_op_minima",
    "ensure_negative_boundary_policy",
    "lossy_negative_boundaries",
    "subsume_forward_negative_boundaries",
    "trained_entry_boundary",
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


def _carries_entry_quantizer(perceptron) -> bool:
    """Whether the perceptron's input wire stack holds a trained entry
    quantizer (installs may nest it in an ``nn.Sequential``)."""
    input_activation = getattr(perceptron, "input_activation", None)
    if input_activation is None:
        return False
    return any(
        isinstance(m, ChipInputQuantizer) for m in input_activation.modules()
    )


def trained_entry_boundary(op, consumers) -> bool:
    """[sigma-scope law] Whether ``op``'s boundary is TRAINED-lossy.

    ALL non-host boundary consumers carry a trained entry quantizer -> the
    exact-QAT trained through the deployed clamp (I1 holds with sigma = 0):
    stamping sigma afterwards would CHANGE the trained function. NONE -> the
    legacy sigma path. MIXED -> the entry-install and consumer-walk universes
    disagree — fail loud (the drift alarm)."""
    entries = [
        _perceptron_of(consumer)
        for consumer in boundary_consumers(op, consumers)
        if not _is_host_node(consumer)
    ]
    entries = [p for p in entries if p is not None]
    if not entries:
        return False
    quantized = [_carries_entry_quantizer(p) for p in entries]
    if all(quantized):
        return True
    if any(quantized):
        raise NotImplementedError(
            "negative-boundary: a boundary with MIXED consumers (some entries "
            "carry the trained quantizer, some do not) — the entry-install and "
            "consumer-walk universes disagree; install the quantizer on every "
            "entry of this boundary or on none."
        )
    return False


def calibrated_compute_op_minima(
    model, calibration_x: torch.Tensor, T: int, *, forward_fn,
) -> Dict[Any, torch.Tensor]:
    """Per-ComputeOp minima of the NF boundary values on the calibration set.

    A ComputeOp-free graph has no value boundary at all, so it skips the
    calibration forward entirely — a structural no-op for both mechanisms.
    """
    mapper_repr = model.get_mapper_repr()
    if not any(
        isinstance(n, ComputeOpMapper) for n in mapper_repr.execution_order()
    ):
        return {}
    recorder: Dict[Any, torch.Tensor] = {}
    with torch.no_grad():
        forward_fn(model, calibration_x, T, compute_min_recorder=recorder)
    return recorder


def _effective_minimum(op, mins: torch.Tensor) -> torch.Tensor:
    """The boundary minimum an encoder actually sees: the calibrated RAW
    minimum plus any shift the ON mechanism baked onto this op."""
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
    consumers = model.get_mapper_repr().consumer_map()
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
    consumers = mapper_repr.consumer_map()
    order = {id(n): i for i, n in enumerate(mapper_repr.execution_order())}

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
    # [sigma-scope law] Trained-clamp boundaries are the QAT's own function:
    # ONE filter feeds the stamp path, the subsume path, AND the recheck the
    # same universe (universe drift between them is how silent-skip bugs
    # slip in).
    consumers = model.get_mapper_repr().consumer_map()
    minima = {
        op: mins for op, mins in minima.items()
        if not trained_entry_boundary(op, consumers)
    }
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


def ensure_negative_boundary_policy(
    model,
    trainer,
    *,
    spiking_mode: str,
    simulation_steps: int,
    device,
    shift_enabled: bool,
    n_batches: int = 2,
) -> NegativeBoundaryResult | None:
    """Calibrate + apply the policy from the trainer's validation cache.

    Callable at the AQ install seam (so the exact-QAT trains through the
    shifted boundary) AND at SCM: a later call walks the already-shifted NF,
    records non-negative minima, and stamps nothing new — the SCM invocation
    degrades to the drift verifier.
    """
    batches = [x for x, _ in trainer.iter_validation_batches(n_batches)]
    if not batches:
        return None
    calibration_x = torch.cat(batches, dim=0).to(device)
    return apply_negative_boundary_policy(
        model,
        calibration_x,
        int(simulation_steps),
        shift_enabled=shift_enabled,
        forward_fn=calibration_forward_for_mode(spiking_mode),
    )
