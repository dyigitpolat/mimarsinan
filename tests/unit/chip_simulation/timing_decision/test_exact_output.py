"""A quantized timing program's nevresim output reaches the host as the exact level (S - k) / S.

The generated main printed every output at the stream's six significant digits, so a level the integer program holds
exactly came back rounded (5/7 as 0.714286, 127/128 as 0.992188): off the 1/(2S) grid by more than the grid dust from
S 128 on, which the next segment's lattice reading, and every host op, then read as another value.
"""

from __future__ import annotations

import tempfile

from spiking_points import point_of_mode

import numpy as np
import pytest

pytestmark = [pytest.mark.slow]


def _one_core_segment(weight: float, theta: float):
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    from mimarsinan.mapping.packing.softcore import HardCore, HardCoreMapping

    core = HardCore(4, 4, has_bias_capability=False)
    core.core_matrix = np.zeros((4, 4), dtype=np.float64)
    core.core_matrix[0, 0] = weight
    core.axon_sources = [SpikeSource(-2, 0, is_input=True)]
    core.available_axons = 3
    core.available_neurons = 3
    core.threshold = theta
    core.latency = 0
    segment = HardCoreMapping([])
    segment.cores = [core]
    segment.output_sources = np.asarray([SpikeSource(0, 0)], dtype=object)
    return segment


def _nevresim_level(s: int, weight: int, theta: int) -> float:
    from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
    from mimarsinan.common.env import nevresim_root

    NevresimDriver.nevresim_path = NevresimDriver.nevresim_path or nevresim_root()
    with tempfile.TemporaryDirectory() as tmp:
        driver = NevresimDriver(
            1, _one_core_segment(weight, theta), tmp, int,
            spike_generation_mode="TTFS", thresholding_mode="<=", firing_mode="TTFS",
            point=point_of_mode("ttfs_quantized"), threshold_type=float, verbose=False,
        )
        raw = driver.predict_spiking_raw([(np.asarray([1.0]), np.asarray([0.0]))], s, 0)
    return float(np.asarray(raw, dtype=np.float64)[0, 0])


@pytest.mark.timeout(600)
@pytest.mark.parametrize("s, weight, theta, steps_left", [
    (7, 5, 7, 5),        # V = 5 on theta 7: fires at step 2, the level 5/7
    (128, 127, 128, 127),  # V = 127 on theta 128: fires at step 1, the level 127/128
])
def test_the_quantized_timing_output_is_the_exact_level(s, weight, theta, steps_left):
    level = _nevresim_level(s, weight, theta)
    assert level == steps_left / s, (level, steps_left / s)
