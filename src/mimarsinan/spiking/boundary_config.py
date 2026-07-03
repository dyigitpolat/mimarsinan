"""Runtime configuration for segment-boundary encode/decode."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BoundaryConfig:
    """Runtime config for segment-boundary encode/decode."""

    simulation_length: int
    spiking_mode: str
    cycle_accurate: bool
    spike_mode: str = "Uniform"
    thresholding_mode: str = "<="
    firing_mode: str = "Default"
    compute_dtype: torch.dtype = torch.float64
    negative_shift: bool = False
    spike_generation_mode: str = "Uniform"

    @property
    def use_cycle_accurate_trains(self) -> bool:
        return self.spiking_mode == "lif" and self.cycle_accurate
