"""The ODIN feasibility gates: typed, keyed, and each naming its remediation."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from mimarsinan.chip_simulation.soma_axes import PER_AXON_SIGN, PER_SYNAPSE_SIGN
from mimarsinan.transformations.quantization_bounds import quantization_bounds

KEY_THETA_CEILING = "odin.theta_ceiling"
KEY_WEIGHT_MAGNITUDE_RANGE = "odin.weight_magnitude_range"
KEY_FAN_IN = "odin.fan_in"
KEY_EMISSION_BOUND = "odin.emission_bound"
KEY_MEMBRANE_INIT = "odin.membrane_init"

#: The count currency a segment boundary carries (plan Sec.2.2): a per-window
#: count above this cannot be transported, so the deployment refuses at export
#: time rather than overflowing one of the four implementations.
EMISSION_CEILING = 127


class OdinFeasibilityError(ValueError):
    """A mapping is outside the declared target's feasible scope, by KEY."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key


def check_theta_ceiling(theta: Any, *, membrane_bits: int, core_index: int) -> int:
    """theta must land in ``[1, 2**membrane_bits - 1]``; theta IS the folded scale."""
    if membrane_bits <= 0:
        raise OdinFeasibilityError(
            KEY_THETA_CEILING,
            f"core {core_index}: the target declares no fixed membrane width "
            f"(membrane_bits={membrane_bits}), so no threshold ceiling exists to "
            f"check. Declare membrane_bits on the platform before exporting.")
    ceiling = (1 << int(membrane_bits)) - 1
    snapped = _as_integer(theta)
    if snapped is None:
        raise OdinFeasibilityError(
            KEY_THETA_CEILING,
            f"core {core_index}: threshold {theta!r} is not integral; the "
            f"threshold register is a {membrane_bits}-bit integer and a silent "
            f"truncation would change the deployed physics.")
    if snapped < 1:
        raise OdinFeasibilityError(
            KEY_THETA_CEILING,
            f"core {core_index}: threshold {snapped} is below 1. The compare is "
            f"'membrane >= theta', so theta=0 fires on every event including a "
            f"zero-magnitude one, which is not the mapped program.")
    if snapped > ceiling:
        raise OdinFeasibilityError(
            KEY_THETA_CEILING,
            f"core {core_index}: threshold {snapped} exceeds the "
            f"{membrane_bits}-bit membrane ceiling {ceiling}. theta IS the folded "
            f"quantization scale, so this is a real scope limit: re-run the "
            f"scale/threshold adaptation ladder targeting the ceiling, or deploy "
            f"a core variant with a wider membrane. Do not clip the threshold.")
    return snapped


def check_weight_magnitudes(
    matrix: Any, *, weight_bits: int, weight_sign_granularity: str, core_index: int
) -> None:
    """The representable weight set, which ``per_axon`` signs make SYMMETRIC."""
    if weight_sign_granularity == PER_AXON_SIGN:
        limit = (1 << (int(weight_bits) - 1)) - 1
        low = -limit
    elif weight_sign_granularity == PER_SYNAPSE_SIGN:
        low, limit = quantization_bounds(int(weight_bits))
    else:
        raise OdinFeasibilityError(
            KEY_WEIGHT_MAGNITUDE_RANGE,
            f"core {core_index}: unknown weight_sign_granularity "
            f"{weight_sign_granularity!r}; declare {PER_SYNAPSE_SIGN!r} or "
            f"{PER_AXON_SIGN!r} — the representable weight set differs between "
            f"them and cannot be guessed.")
    values = np.asarray(matrix)
    offenders = values[(values < low) | (values > limit)]
    if offenders.size:
        offender = offenders.flat[0]
        detail = ""
        if weight_sign_granularity == PER_AXON_SIGN:
            detail = (
                f" A {PER_AXON_SIGN} substrate stores the sign once per physical "
                f"row and the cell holds an UNSIGNED magnitude, so the range is "
                f"symmetric and q_min={quantization_bounds(int(weight_bits))[0]} "
                f"is not representable.")
        raise OdinFeasibilityError(
            KEY_WEIGHT_MAGNITUDE_RANGE,
            f"core {core_index}: weight {offender} is outside the representable "
            f"range [{low}, {limit}] at weight_bits={weight_bits} under "
            f"weight_sign_granularity={weight_sign_granularity!r}.{detail} "
            f"Re-quantize against the declared symmetric range; never saturate "
            f"silently.")


def check_fan_in(used_axons: int, *, effective_max_axons: int, core_index: int) -> None:
    """The exporter RE-CHECKS what the mapper should already have refused."""
    if int(used_axons) > int(effective_max_axons):
        raise OdinFeasibilityError(
            KEY_FAN_IN,
            f"core {core_index}: {used_axons} logical axons exceed the effective "
            f"limit {effective_max_axons}. The mapper's WideFanInUnsupportedError "
            f"should have refused this first; reaching the exporter means the "
            f"declared geometry and the packed mapping disagree. Re-map against "
            f"the declared platform, or declare a wider core.")


def check_membrane_init(value: Any, *, theta: int, core_index: int) -> int:
    """``V0`` is programmed into the neuron word's state field: ``0 <= V0 < theta``."""
    snapped = _as_integer(value)
    if snapped is None or snapped < 0 or snapped >= theta:
        raise OdinFeasibilityError(
            KEY_MEMBRANE_INIT,
            f"core {core_index}: membrane_init {value!r} must be an integer in "
            f"[0, {theta}) — the row-pair lemma needs the membrane BELOW theta at "
            f"window start, or a zero-magnitude event would fire.")
    return snapped


def emission_bound_of(
    contributions: Iterable[Tuple[int, int]], *, theta: int
) -> int:
    """``ceil((sum_a max(w,0) e_in(a) + (theta - 1)) / theta)`` — plan Sec.2.2."""
    total = sum(max(int(weight), 0) * int(events) for weight, events in contributions)
    return int(math.ceil((total + theta - 1) / theta))


def propagate_emission_bounds(
    mapping: Any, *, ceiling: int
) -> Dict[Tuple[int, int], int]:
    """Per-(core, neuron) emission bounds, propagated topologically within the segment.

    Seeded ``1`` at every segment entry (the encode emits at most one event per
    cycle) and at every always-on row; refuses above ``ceiling``, and refuses a
    cycle rather than recursing forever.
    """
    bounds: Dict[Tuple[int, int], int] = {}
    visiting: set = set()

    def source_events(source: Any) -> int:
        entry = entry_event_bound(source)
        if entry is not None:
            return entry
        return bound_for(int(source.core_), int(source.neuron_))

    def bound_for(core_index: int, neuron_index: int) -> int:
        key = (core_index, neuron_index)
        if key in bounds:
            return bounds[key]
        if key in visiting:
            raise OdinFeasibilityError(
                KEY_EMISSION_BOUND,
                f"core {core_index} neuron {neuron_index} sits on a cycle in the "
                f"segment graph; the emission bound is only defined on a DAG.")
        visiting.add(key)
        core = mapping.cores[core_index]
        theta = _core_theta(core, core_index)
        column = np.asarray(core.get_core_matrix())[:, neuron_index]
        contributions = [
            (int(weight), source_events(core.axon_sources[slot]))
            for slot, weight in enumerate(column)
            if weight > 0
        ]
        value = emission_bound_of(contributions, theta=theta)
        visiting.discard(key)
        if value > ceiling:
            raise OdinFeasibilityError(
                KEY_EMISSION_BOUND,
                f"core {core_index} neuron {neuron_index} can emit up to {value} "
                f"events per window, above the count-currency ceiling {ceiling}. "
                f"The remedy is scale/threshold adaptation that lowers the bound "
                f"— never a wider count clamp, which would silently change what "
                f"the wire carries.")
        bounds[key] = value
        return value

    for source in mapping.output_sources:
        source_events(source)
    for core_index, core in enumerate(mapping.cores):
        for neuron_index in range(int(core.neurons_per_core)):
            bound_for(core_index, neuron_index)
    return bounds


def entry_event_bound(source: Any):
    """The seed a segment ENTRY carries, or ``None`` for another core's neuron.

    The encode emits at most one event per cycle and an always-on row fires
    exactly once, so both seed 1; an off axon delivers nothing.
    """
    if getattr(source, "is_off_", False):
        return 0
    if getattr(source, "is_input_", False) or getattr(source, "is_always_on_", False):
        return 1
    return None if int(getattr(source, "core_", -1)) >= 0 else 0


def _core_theta(core: Any, core_index: int) -> int:
    theta = _as_integer(getattr(core, "threshold", None))
    if theta is None or theta < 1:
        raise OdinFeasibilityError(
            KEY_EMISSION_BOUND,
            f"core {core_index} carries threshold {getattr(core, 'threshold', None)!r}; "
            f"the emission bound divides by theta, so an unset or non-integral "
            f"threshold has no bound rather than a default one.")
    return theta


def _as_integer(value: Any):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    snapped = round(number)
    return snapped if abs(number - snapped) <= 1e-9 else None
