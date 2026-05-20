"""Spike-train encoding shared by PyTorch (cycle-accurate LIF) and hybrid simulation."""

from mimarsinan.spiking.lif_utils import (
    apply_cycle_accurate_trains_to_model,
    unwrap_lif_activation,
)
from mimarsinan.spiking.spike_trains import (
    lif_spike_train,
    rates_to_spike_train,
    uniform_spike_train,
)

__all__ = [
    "apply_cycle_accurate_trains_to_model",
    "lif_spike_train",
    "rates_to_spike_train",
    "uniform_spike_train",
    "unwrap_lif_activation",
]
