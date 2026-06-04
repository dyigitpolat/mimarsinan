"""Spike-train encoding shared by PyTorch (cycle-accurate LIF) and hybrid simulation."""

from mimarsinan.spiking.lif_utils import (
    apply_cycle_accurate_trains_to_model,
    unwrap_lif_activation,
)
from mimarsinan.spiking.segment_boundary import (
    BoundaryConfig,
    decode_segment_output,
    decode_segment_output_torch,
    encode_compute_boundary,
    encode_segment_input,
)
from mimarsinan.spiking.segment_forward import (
    LifSegmentPolicy,
    SegmentForwardDriver,
    TtfsSegmentPolicy,
)
from mimarsinan.spiking.spike_trains import (
    lif_spike_train,
    rates_to_spike_train,
    uniform_spike_train,
)

__all__ = [
    "BoundaryConfig",
    "LifSegmentPolicy",
    "SegmentForwardDriver",
    "TtfsSegmentPolicy",
    "apply_cycle_accurate_trains_to_model",
    "decode_segment_output",
    "decode_segment_output_torch",
    "encode_compute_boundary",
    "encode_segment_input",
    "lif_spike_train",
    "rates_to_spike_train",
    "uniform_spike_train",
    "unwrap_lif_activation",
]
