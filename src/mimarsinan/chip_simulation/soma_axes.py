"""The soma-law axes: firing granularity and membrane arithmetic (vocabulary + legality)."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from mimarsinan.chip_simulation.activation_axes import LIF_FAMILY
from mimarsinan.chip_simulation.activation_semantics import (
    is_streamed_lif,
    resolve_activation_semantics,
)
from mimarsinan.chip_simulation.core_semantics import (
    is_mvm_core_semantics,
    resolve_core_semantics,
)
from mimarsinan.chip_simulation.spiking_semantics import NOVENA_FIRING_MODE

FIRING_GRANULARITY_KEY = "firing_granularity"
MEMBRANE_ARITHMETIC_KEY = "membrane_arithmetic"
MEMBRANE_BITS_KEY = "membrane_bits"
WEIGHT_SIGN_GRANULARITY_KEY = "weight_sign_granularity"

# WHEN the threshold is evaluated. ``per_cycle`` is today's law: one compare
# per cycle on the reduced contribution. ``per_event`` compares after every
# arriving event occurrence in canonical order (>=0 spikes per neuron per
# cycle). Deliberately not "windowed": that word already means
# ``spiking_variant == synchronized``.
PER_CYCLE_FIRING = "per_cycle"
PER_EVENT_FIRING = "per_event"
FIRING_GRANULARITIES: Tuple[str, ...] = (PER_CYCLE_FIRING, PER_EVENT_FIRING)

# The membrane's ARITHMETIC. ``unbounded`` is today's signed accumulator (the
# default is not "exact": it is lattice-exact only under armed conditions).
UNBOUNDED_MEMBRANE = "unbounded"
SATURATING_UNSIGNED_MEMBRANE = "saturating_unsigned"
MEMBRANE_ARITHMETICS: Tuple[str, ...] = (
    UNBOUNDED_MEMBRANE, SATURATING_UNSIGNED_MEMBRANE,
)

# WHERE the weight sign physically lives (a platform width, not a soma law).
PER_SYNAPSE_SIGN = "per_synapse"
PER_AXON_SIGN = "per_axon"
WEIGHT_SIGN_GRANULARITIES: Tuple[str, ...] = (PER_SYNAPSE_SIGN, PER_AXON_SIGN)


def resolved_membrane_bits(cfg: Mapping[str, Any]) -> int:
    """The declared fixed membrane width; 0 = not fixed-width.

    TOTAL over every config shape: a malformed declaration reads as undeclared
    here because the registry's own bounds/type error is the single truth.
    """
    value = cfg.get(MEMBRANE_BITS_KEY)
    if value is None or isinstance(value, bool):
        return 0
    try:
        bits = int(value)
    except (TypeError, ValueError):
        return 0
    return bits if bits > 0 else 0


def legal_firing_granularities(cfg: Mapping[str, Any]) -> Tuple[str, ...]:
    """``per_event`` is declarable ONLY at the (lif, streamed) point.

    A windowed hop collapses counts at every boundary and would destroy the
    multiplicity the per-event law produces, so the point would be legal but
    refused — exactly what the legality harness exists to prevent. TOTAL over
    partial configs, with the mvm short-circuit ``is_streamed_lif`` owns
    (the ``t0_44`` scar: dormant axes must never leak legality), and an
    unresolvable axes point ruling NOTHING out — its own keyed error is the
    single truth.
    """
    if is_mvm_core_semantics(resolve_core_semantics(cfg)):
        return (PER_CYCLE_FIRING,)
    try:
        resolve_activation_semantics(cfg)
    except ValueError:
        return FIRING_GRANULARITIES
    return FIRING_GRANULARITIES if is_streamed_lif(cfg) else (PER_CYCLE_FIRING,)


def derived_firing_granularity(cfg: Mapping[str, Any]) -> str:
    """What an absent granularity resolves to: today's per-cycle law, always.

    Reaching ``per_event`` is an explicit declaration; no config changes
    meaning because the axis appeared.
    """
    del cfg
    return PER_CYCLE_FIRING


def legal_membrane_arithmetics(cfg: Mapping[str, Any]) -> Tuple[str, ...]:
    """A saturating unsigned accumulator is a LIF-family law only.

    TTFS latches a single spike and never re-accumulates, so a saturating
    register has nothing to mean there; an unknown family rules nothing out
    (its own keyed error is the single truth).
    """
    if is_mvm_core_semantics(resolve_core_semantics(cfg)):
        return (UNBOUNDED_MEMBRANE,)
    try:
        family = resolve_activation_semantics(cfg).family
    except ValueError:
        return MEMBRANE_ARITHMETICS
    return MEMBRANE_ARITHMETICS if family == LIF_FAMILY else (UNBOUNDED_MEMBRANE,)


def derived_membrane_arithmetic(cfg: Mapping[str, Any]) -> str:
    """BITS-DRIVEN, exactly like weight quantization: a declared
    ``membrane_bits`` width IS the declaration of a saturating register.

    The derived value never leaves the legal set — a width declared against a
    family that has no accumulator is reported by the soma-law contract, not
    by a silently illegal derivation.
    """
    saturating = (
        resolved_membrane_bits(cfg) > 0
        and SATURATING_UNSIGNED_MEMBRANE in legal_membrane_arithmetics(cfg)
    )
    return SATURATING_UNSIGNED_MEMBRANE if saturating else UNBOUNDED_MEMBRANE


def resolved_firing_granularity(cfg: Mapping[str, Any]) -> str:
    """The effective granularity: the declaration, else the derived default."""
    declared = cfg.get(FIRING_GRANULARITY_KEY)
    return derived_firing_granularity(cfg) if declared is None else str(declared)


def resolved_membrane_arithmetic(cfg: Mapping[str, Any]) -> str:
    """The effective membrane arithmetic: the declaration, else bits-driven."""
    declared = cfg.get(MEMBRANE_ARITHMETIC_KEY)
    return derived_membrane_arithmetic(cfg) if declared is None else str(declared)


def firing_mode_for_granularity(firing_granularity: str, otherwise: str) -> str:
    """The reset law a granularity REQUIRES; ``otherwise`` at every other point.

    ``per_event`` admits ONLY the hard-zero reset. Its physical realization
    folds a zero-magnitude row beside every event row, and such a row is a
    no-op only while a fire leaves ``m < theta`` — true of the zero reset, and
    FALSE of the subtractive one the moment one event carries ``>= 2*theta``
    (the row-pair lemma). The subtractive reset is therefore not a slower
    per-event law, it is an unrealizable one.
    """
    if firing_granularity == PER_EVENT_FIRING:
        return NOVENA_FIRING_MODE
    return otherwise
