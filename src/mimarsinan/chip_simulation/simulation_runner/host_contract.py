"""Shared host-attribute contract for SimulationRunner mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mimarsinan.chip_simulation.nevresim.connectivity import ConnectivityMode


class SimulationHostContract:
    """Attributes and hooks provided by ``SimulationRunner.__init__`` to its mixins."""

    simulation_length: int
    input_size: int
    working_directory: str
    weight_type: type[int] | type[float]
    threshold_type: type[int] | type[float]
    spike_generation_mode: str
    firing_mode: str
    thresholding_mode: str
    spiking_mode: str
    nevresim_connectivity_mode: ConnectivityMode
    simulation_step_timeout_s: float
    test_data: list[tuple[np.ndarray, np.ndarray]]
    mapping: Any
    # [C2] deployed membrane decode: default-off by contract; SimulationRunner
    # arms it through the honesty gate.
    membrane_readout: bool = False
    membrane_half_step_charge: float = 0.0

    if TYPE_CHECKING:
        def _evaluate_chip_output(self, predictions) -> float: ...
