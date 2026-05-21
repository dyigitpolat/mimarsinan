"""Spike-train encoding shared by PyTorch (cycle-accurate LIF) and hybrid simulation."""

from mimarsinan.spiking.lif_utils import (
    apply_cycle_accurate_trains_to_model,
    unwrap_lif_activation,
)
from mimarsinan.spiking.segment_encoding import (
    SegmentEncodingConfig,
    build_segment_input_spike_train,
    emit_compute_spike_train,
)
from mimarsinan.spiking.spike_trains import (
    lif_spike_train,
    rates_to_spike_train,
    uniform_spike_train,
)

__all__ = [
    "SegmentEncodingConfig",
    "apply_cycle_accurate_trains_to_model",
    "build_segment_input_spike_train",
    "emit_compute_spike_train",
    "lif_spike_train",
    "rates_to_spike_train",
    "uniform_spike_train",
    "unwrap_lif_activation",
]
