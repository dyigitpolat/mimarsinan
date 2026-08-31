"""The event-serial soma law: one fold kernel, its policy, and its refusals."""

from mimarsinan.models.spiking.serial.fold import (
    lif_serial_fold,
    require_event_counts,
    require_serial_law,
)
from mimarsinan.models.spiking.serial.guards import (
    require_serial_deployment_admissible,
)
from mimarsinan.models.spiking.serial.policy import SerialLIFCyclePolicy
from mimarsinan.models.spiking.serial.refusals import (
    COUNT_CURRENCY_LIMIT,
    COUNT_CURRENCY_WORD_BITS,
    EMISSION_COUNT_CEILING,
    CycleAtomicRefusalError,
    EmissionBoundExceededError,
    count_ceiling,
    MappingTransformRefusalError,
    SaturatingMembraneRefusalError,
    SerialDecompositionMismatchError,
    SerialFoldUnsupportedError,
    SerialMembraneInitError,
    SerialResetLawError,
    SomaLawRefusalError,
    refuse_cycle_atomic,
    refuse_cycle_atomic_walk,
    refuse_saturating_membrane,
)

__all__ = [
    "COUNT_CURRENCY_LIMIT",
    "COUNT_CURRENCY_WORD_BITS",
    "CycleAtomicRefusalError",
    "EMISSION_COUNT_CEILING",
    "EmissionBoundExceededError",
    "MappingTransformRefusalError",
    "SaturatingMembraneRefusalError",
    "SerialDecompositionMismatchError",
    "SerialFoldUnsupportedError",
    "SerialLIFCyclePolicy",
    "SerialMembraneInitError",
    "SerialResetLawError",
    "SomaLawRefusalError",
    "count_ceiling",
    "lif_serial_fold",
    "refuse_cycle_atomic",
    "refuse_cycle_atomic_walk",
    "refuse_saturating_membrane",
    "require_event_counts",
    "require_serial_deployment_admissible",
    "require_serial_law",
]
