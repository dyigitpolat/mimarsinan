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
from mimarsinan.tuning.orchestration.fast_ladder import FastLadderMixin
from mimarsinan.tuning.orchestration.rate_tuner_seam import RateTunerSeamMixin
from mimarsinan.tuning.orchestration.smooth_adaptation_cycle import SmoothAdaptationCycleMixin
from mimarsinan.tuning.orchestration.smooth_adaptation_run import SmoothAdaptationRunMixin


class SmoothAdaptationTuner(
    RateTunerSeamMixin,
    FastLadderMixin,
    SmoothAdaptationCycleMixin,
    SmoothAdaptationRunMixin,
    TunerBase,
):
    """Orchestration loop for smooth rate-based adaptation.

    Exposes the uniform ``RateTunerSeam`` so an ``OptimizationDriver`` can drive any
    smooth tuner generically; ``FastLadderMixin`` carries the opt-in fast driver.
    """


UnifiedPerceptronTuner = SmoothAdaptationTuner
