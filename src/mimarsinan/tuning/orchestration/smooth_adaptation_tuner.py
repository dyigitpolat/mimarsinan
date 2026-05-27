"""TunerBase and SmoothAdaptationTuner — shared tuning orchestration."""

from mimarsinan.tuning.orchestration.tuner_base import (
    TunerBase,
    CATASTROPHIC_DROP_FACTOR,
    _RECOVERY_PATIENCE,
    _STUCK_STREAK_REQUIRED,
)

__all__ = [
    "TunerBase",
    "SmoothAdaptationTuner",
    "UnifiedPerceptronTuner",
    "CATASTROPHIC_DROP_FACTOR",
    "_RECOVERY_PATIENCE",
    "_STUCK_STREAK_REQUIRED",
]
from mimarsinan.tuning.orchestration.smooth_adaptation_cycle import SmoothAdaptationCycleMixin
from mimarsinan.tuning.orchestration.smooth_adaptation_run import SmoothAdaptationRunMixin


class SmoothAdaptationTuner(SmoothAdaptationCycleMixin, SmoothAdaptationRunMixin, TunerBase):
    """Orchestration loop for smooth rate-based adaptation."""


UnifiedPerceptronTuner = SmoothAdaptationTuner
