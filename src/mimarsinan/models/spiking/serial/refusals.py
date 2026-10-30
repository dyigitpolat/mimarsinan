"""Typed refusals: every cycle-atomic theorem a per-event soma law denies.

A mechanism whose correctness argument assumes "one compare per cycle on the
reduced contribution" or "an unbounded lossless accumulator" is not slow under
the ODIN point — it is WRONG. Each such mechanism raises from here, naming the
axis value that denies it, so a refusal reads as a physics statement instead
of a missing feature.
"""

from __future__ import annotations

from typing import Any, Mapping

from mimarsinan.chip_simulation.soma_axes import MEMBRANE_BITS_KEY
from mimarsinan.models.nn.lif_kernels import MembraneRailTouchedError

__all__ = [
    "COUNT_CURRENCY_FLOOR_BITS",
    "COUNT_CURRENCY_LIMIT",
    "COUNT_CURRENCY_WORD_BITS",
    "CycleAtomicRefusalError",
    "EMISSION_COUNT_CEILING",
    "EmissionBoundExceededError",
    "MappingTransformRefusalError",
    "MembraneRailTouchedError",
    "SaturatingMembraneRefusalError",
    "SerialDecompositionMismatchError",
    "SerialFoldUnsupportedError",
    "SerialMembraneInitError",
    "SerialResetLawError",
    "SomaLawRefusalError",
    "count_ceiling",
    "refuse_cycle_atomic",
    "refuse_cycle_atomic_walk",
    "refuse_saturating_membrane",
]

COUNT_CURRENCY_FLOOR_BITS = 8
"""A chip narrower than a byte still gets the byte's currency: no
implementation of the fold carries counts in a sub-byte word."""

COUNT_CURRENCY_WORD_BITS = 16
"""The width of the widest count EVERY implementation carries, whatever chip is
running: nevresim's ``spike_t`` (``std::int16_t``), the counted raster it
prints, the ``int16`` arrays the host re-packs it into, and the torch fold's
own tensor. A ceiling above this would name a number some implementation cannot
hold, so ``count_ceiling`` floors every chip's claim at it."""

COUNT_CURRENCY_LIMIT = (1 << (COUNT_CURRENCY_WORD_BITS - 1)) - 1


def count_ceiling(chip_claims: Any) -> int:
    """THE count currency of ONE chip: what a per-window count may reach on it.

    The currency is a REPRESENTATION, not a geometry: the wire carries no count
    field at all (multiplicity is k adjacent AER events), so no crossbar width
    moves this number. What DOES move it is the width the chip declares its
    registers at, because that is the width at which every implementation
    instantiates the chip's law — nevresim's ``EventSerialIntegrate<bits>``, the
    generated core's accumulator, and this fold's own assertion. So the ceiling
    is the largest positive value a SIGNED word of the chip's declared
    arithmetic width holds, floored at a byte and capped by what the narrowest
    implementation carries.

    ``chip_claims`` is any surface that names ``membrane_bits`` — a resolved
    ``SomaLaw``, a ``CoreSpec``, a ``ChipConfig``, or a sealed bundle's
    ``chip_config`` mapping. TOTAL over every shape, like
    ``soma_axes.resolved_membrane_bits``: an undeclared or malformed width reads
    as undeclared and takes the byte's currency, because the registry's own
    bounds error is the single truth about a bad declaration.
    """
    bits = max(_claimed_membrane_bits(chip_claims), COUNT_CURRENCY_FLOOR_BITS)
    return min((1 << (bits - 1)) - 1, COUNT_CURRENCY_LIMIT)


def _claimed_membrane_bits(chip_claims: Any) -> int:
    """``membrane_bits`` off any claims surface; 0 = the chip declares none."""
    if isinstance(chip_claims, Mapping):
        value = chip_claims.get(MEMBRANE_BITS_KEY)
        if value is None:
            nested = chip_claims.get("soma_law")
            value = (nested.get(MEMBRANE_BITS_KEY)
                     if isinstance(nested, Mapping) else None)
    else:
        value = getattr(chip_claims, MEMBRANE_BITS_KEY, None)
    if value is None or isinstance(value, bool):
        return 0
    try:
        bits = int(value)
    except (TypeError, ValueError):
        return 0
    return bits if bits > 0 else 0


EMISSION_COUNT_CEILING = count_ceiling(None)
"""The count currency of a chip that declares NO register width — the byte's
127. It is the default ceiling every consumer falls back to and the STOCK
fabric's own currency (it declares an 8-bit membrane); a chip that declares a
wider register gets ``count_ceiling``'s answer instead. It is asserted, never
clamped: a silent saturation here is the exact failure the count currency
exists to make impossible."""


class SomaLawRefusalError(NotImplementedError):
    """Base: the resolved soma point denies this mechanism's theorem."""


class CycleAtomicRefusalError(SomaLawRefusalError):
    """A cycle-atomic mechanism met ``firing_granularity='per_event'``."""


class SaturatingMembraneRefusalError(SomaLawRefusalError):
    """A lossless-accumulator identity met ``membrane_arithmetic``
    ``'saturating_unsigned'``."""


class MappingTransformRefusalError(SomaLawRefusalError):
    """A mapping transform that re-thresholds a partial sum met the point."""


class SerialFoldUnsupportedError(SomaLawRefusalError):
    """The serial fold was handed a shape or law it does not implement."""


class SerialResetLawError(SomaLawRefusalError):
    """A per-event point declared a reset its row-pair realization denies."""


class SerialDecompositionMismatchError(SomaLawRefusalError):
    """The NF twin's event order is not the mapper's canonical slot order."""


class EmissionBoundExceededError(ValueError):
    """A neuron emitted more spikes in one cycle than the count currency holds."""


class SerialMembraneInitError(ValueError):
    """``lif_membrane_init`` violates the per-event window-start contract."""


def refuse_cycle_atomic(soma_law: Any, *, mechanism: str, theorem: str) -> None:
    """Refuse ``mechanism`` when the point evaluates the threshold per event."""
    if soma_law is None or not soma_law.is_per_event:
        return
    raise CycleAtomicRefusalError(
        f"{mechanism} is refused under firing_granularity='per_event': "
        f"{theorem} The per-event law evaluates the threshold after every "
        f"arriving event occurrence, so that hypothesis is false by "
        f"construction — running it would report a DIFFERENT physics as the "
        f"deployed number. Deploy firing_granularity='per_cycle', or use the "
        f"event-serial executor."
    )


def refuse_cycle_atomic_walk(
    soma_law: Any, *, retime: bool, synchronized: bool
) -> None:
    """Refuse the NF walk disciplines whose theorem the per-event law denies."""
    if synchronized:
        refuse_cycle_atomic(
            soma_law,
            mechanism="the synchronized two-window NF walk",
            theorem=(
                "[calculus §16] it evaluates one hop's emission from its input "
                "COUNTS alone, one staircase eval per window."
            ),
        )
    if retime:
        refuse_cycle_atomic(
            soma_law,
            mechanism="the per-hop retimed NF walk",
            theorem=(
                "[C3/R5] it replaces each hop's emitted train by the uniform "
                "re-encode of its window count, which carries at most one "
                "spike per cycle."
            ),
        )


def refuse_saturating_membrane(
    soma_law: Any, *, mechanism: str, identity: str
) -> None:
    """Refuse ``mechanism`` when the membrane saturates instead of accumulating."""
    if soma_law is None:
        return
    bounds = soma_law.membrane_bounds
    if bounds is None:
        return
    low, high = bounds
    raise SaturatingMembraneRefusalError(
        f"{mechanism} is refused under "
        f"membrane_arithmetic={soma_law.membrane_arithmetic!r} "
        f"(membrane_bits={soma_law.membrane_bits}): {identity} A register "
        f"confined to [{low:.0f}, {high:.0f}] cannot hold the residual charge "
        f"in general, so the identity is false and the reported number would "
        f"be a fiction. Deploy an unbounded membrane, or read counts only."
    )
