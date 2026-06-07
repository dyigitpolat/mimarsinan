"""Custom autograd activations and clamp."""
from mimarsinan.models.nn.activations.autograd import (
    ChipInputQuantizer,
    DifferentiableClamp,
    LeakyGradReLU,
    LeakyGradReLUFunction,
    RoundedStaircaseFunction,
    StaircaseFunction,
    TTFSInputGridQuantizer,
    TTFSStaircaseFunction,
)
from mimarsinan.models.nn.activations.lif import (
    LIFActivation,
    StrictATanSurrogate,
    run_cycle_accurate,
    uniform_encode_to_spike_train,
)
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation

__all__ = [
    "ChipInputQuantizer",
    "DifferentiableClamp",
    "LIFActivation",
    "TTFSCycleActivation",
    "LeakyGradReLU",
    "LeakyGradReLUFunction",
    "RoundedStaircaseFunction",
    "StaircaseFunction",
    "StrictATanSurrogate",
    "TTFSInputGridQuantizer",
    "TTFSStaircaseFunction",
    "run_cycle_accurate",
    "uniform_encode_to_spike_train",
]
