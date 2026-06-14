"""Composable decorators and adjustment strategies for activations."""
from mimarsinan.models.nn.decorators.adjustment import (
    ActivationReplacementDecorator,
    DecoratedActivation,
    MixAdjustmentStrategy,
    NestedAdjustmentStrategy,
    NestedDecoration,
    RandomMaskAdjustmentStrategy,
    RateAdjustedDecorator,
)
from mimarsinan.models.nn.decorators.clamp_quantize import ClampDecorator, QuantizeDecorator
from mimarsinan.models.nn.decorators.rate_buffer import RateBuffer
from mimarsinan.models.nn.decorators.transforms import (
    NoisyDropout,
    SavedTensorDecorator,
    ScaleDecorator,
    ShiftDecorator,
    StatsDecorator,
)

__all__ = [
    "ActivationReplacementDecorator",
    "ClampDecorator",
    "DecoratedActivation",
    "MixAdjustmentStrategy",
    "NestedAdjustmentStrategy",
    "NestedDecoration",
    "NoisyDropout",
    "QuantizeDecorator",
    "RandomMaskAdjustmentStrategy",
    "RateBuffer",
    "RateAdjustedDecorator",
    "SavedTensorDecorator",
    "ScaleDecorator",
    "ShiftDecorator",
    "StatsDecorator",
]
