"""Typed refusals: every cycle-atomic theorem a per-event soma law denies.

A mechanism whose correctness argument assumes "one compare per cycle on the
reduced contribution" or "an unbounded lossless accumulator" is not slow under
the ODIN point — it is WRONG. Each such mechanism raises from here, naming the
axis value that denies it, so a refusal reads as a physics statement instead
of a missing feature.
"""

from __future__ import annotations

from typing import Any

from mimarsinan.models.nn.lif_kernels import MembraneRailTouchedError

__all__ = [
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
    "refuse_cycle_atomic",
    "refuse_cycle_atomic_walk",
    "refuse_saturating_membrane",
]

EMISSION_COUNT_CEILING = 127
"""The count currency's ceiling. A window/cycle count travels as one signed
8-bit event count (nevresim's ``spike_t``), the counted raster prints it and
the exporter prices the wire from the same number, so 127 is the loudest bound
EVERY implementation shares (plan §2.2). It is asserted, never clamped: a
silent saturation here is the exact failure the count currency exists to make
impossible."""


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
