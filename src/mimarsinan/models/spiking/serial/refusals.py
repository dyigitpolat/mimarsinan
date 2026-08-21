"""Typed refusals: every cycle-atomic theorem a per-event soma law denies.

A mechanism whose correctness argument assumes "one compare per cycle on the
reduced contribution" or "an unbounded lossless accumulator" is not slow under
the ODIN point — it is WRONG. Each such mechanism raises from here, naming the
axis value that denies it, so a refusal reads as a physics statement instead
of a missing feature.
"""

from __future__ import annotations

from typing import Any


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


def refuse_saturating_membrane(
    soma_law: Any, *, mechanism: str, identity: str
) -> None:
    """Refuse ``mechanism`` when the membrane saturates instead of accumulating."""
    if soma_law is None or not soma_law.saturates:
        return
    raise SaturatingMembraneRefusalError(
        f"{mechanism} is refused under "
        f"membrane_arithmetic='saturating_unsigned' "
        f"(membrane_bits={soma_law.membrane_bits}): {identity} A register "
        f"that clamps to [0, 2**bits - 1] on every update DESTROYS charge, so "
        f"the identity is false and the reported number would be a fiction. "
        f"Deploy an unbounded membrane, or read counts only."
    )
