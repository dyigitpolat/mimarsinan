"""Shared host-attribute contract for SimulationRunner mixins."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    test_data: list[tuple[np.ndarray, np.ndarray]]

    if TYPE_CHECKING:
        def _evaluate_chip_output(self, predictions) -> float: ...
