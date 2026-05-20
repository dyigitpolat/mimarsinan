"""Spike-train encoding shared by PyTorch (cycle-accurate LIF) and hybrid simulation."""

from mimarsinan.spiking.chip_aligned_forward import (
    ChipAlignedForward,
    install_chip_aligned_forward,
    uninstall_chip_aligned_forward,
)
from mimarsinan.spiking.lif_utils import (
    apply_cycle_accurate_trains_to_model,
    boundary_lif_activation,
    unwrap_lif_activation,
)
from mimarsinan.spiking.segment_encoding import (
    BoundaryKind,
    BoundaryLifCache,
    SegmentEncodingConfig,
    build_segment_input_spike_train,
    classify_encoding_boundary,
    emit_compute_spike_train,
)
from mimarsinan.spiking.spike_trains import (
    lif_spike_train,
    rates_to_spike_train,
    uniform_spike_train,
)

__all__ = [
    "BoundaryKind",
    "BoundaryLifCache",
    "ChipAlignedForward",
    "SegmentEncodingConfig",
    "apply_cycle_accurate_trains_to_model",
    "boundary_lif_activation",
    "build_segment_input_spike_train",
    "classify_encoding_boundary",
    "emit_compute_spike_train",
    "install_chip_aligned_forward",
    "lif_spike_train",
    "rates_to_spike_train",
    "uninstall_chip_aligned_forward",
    "uniform_spike_train",
    "unwrap_lif_activation",
]
