"""SCM <-> nevresim parity pin for the analytical TTFS upper clamp (6b conservation).

The wire contract for analytical ``ttfs`` is ``clip(relu(V)/theta, 0, 1)``: the
torch kernel and the SCM continuous branch both clip, and nevresim's
``TTFSAnalyticalCompute`` neuron must agree — the unclamped ``relu(V)/theta``
was the fifth conservation-class sighting (82/10000 decision flips on t0_06).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw
from mimarsinan.chip_simulation.ttfs.ttfs_segment import (
    run_ttfs_continuous_segment,
    segment_ttfs_arrays_from_mapping,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.common.build_utils import find_cpp20_compiler
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping

NEVRESIM_ROOT = Path(__file__).resolve().parents[4] / "nevresim"


def _have_nevresim() -> bool:
    return NEVRESIM_ROOT.is_dir() and find_cpp20_compiler()[0] is not None


def _make_core(weights: np.ndarray, sources: list[SpikeSource]) -> HardCore:
    axons, neurons = weights.shape
    core = HardCore(axons, neurons, has_bias_capability=False)
    core.axon_sources = list(sources)
    core.core_matrix = weights.astype(np.float64)
    core.threshold = 1.0
    core.latency = 0
    core.available_axons = 0
    core.available_neurons = 0
    return core


def _saturating_two_core_chain() -> tuple[HardCoreMapping, int]:
    """Core 0 drives relu(V)/theta ABOVE 1 on neuron 0; core 1 consumes it.

    Weights (axons x neurons): neuron 0 sees 2.0 * x0 (saturates for x0 > 0.5),
    neuron 1 sees 0.5 * x1 (never saturates) — the downstream core exposes the
    unclamped wire value at the chip output.
    """
    core0 = _make_core(
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        [SpikeSource(-2, 0, is_input=True, is_off=False),
         SpikeSource(-2, 1, is_input=True, is_off=False)],
    )
    core1 = _make_core(
        np.eye(2),
        [SpikeSource(0, 0, is_input=False, is_off=False),
         SpikeSource(0, 1, is_input=False, is_off=False)],
    )
    hcm = HardCoreMapping(chip_cores=[])
    hcm.cores = [core0, core1]
    hcm.output_sources = np.array(
        [SpikeSource(1, 0, is_input=False, is_off=False),
         SpikeSource(1, 1, is_input=False, is_off=False)],
        dtype=object,
    )
    return hcm, 2


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    if not _have_nevresim():
        pytest.skip("nevresim or C++20 compiler unavailable")
    base = tmp_path_factory.mktemp("nevresim_ttfs_clamp")
    hcm, input_size = _saturating_two_core_chain()
    latency = ChipLatency(hcm).calculate()
    T = 4

    inputs = np.array(
        [[0.8, 0.6],   # x0 saturates: relu(1.6)/1 clips to 1.0
         [0.4, 1.0],   # nothing saturates: passthrough regression row
         [1.0, 0.0]],  # boundary: exactly 2.0 -> clips to 1.0
        dtype=np.float64,
    )

    NevresimDriver.nevresim_path = str(NEVRESIM_ROOT)
    driver = NevresimDriver(
        input_size,
        hcm,
        str(base),
        float,
        spike_generation_mode="TTFS",
        firing_mode="TTFS",
        spiking_mode="ttfs",
        connectivity_mode="runtime",
        verbose=False,
    )
    binary = driver.emit_main_and_compile(len(inputs), T, latency)
    raw = run_binary_raw(
        binary_path=binary,
        work_dir=str(base),
        input_loader=[(row, np.zeros(1)) for row in inputs],
        output_size=driver.chip.output_size,
        simulation_length=T,
        input_size=input_size,
        spike_generation_mode="TTFS",
        max_input_count=len(inputs),
        num_proc=1,
    )
    nevresim_out = np.asarray(raw, dtype=np.float64).reshape(len(inputs), -1)

    seg = segment_ttfs_arrays_from_mapping(hcm)
    scm_out, _ = run_ttfs_continuous_segment(seg, inputs)
    return nevresim_out, scm_out


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
class TestAnalyticalTtfsUpperClampParity:
    def test_nevresim_matches_the_scm_continuous_branch(self, outputs):
        nevresim_out, scm_out = outputs
        np.testing.assert_allclose(nevresim_out, scm_out, rtol=0, atol=1e-9)

    def test_saturating_neuron_clamps_to_one(self, outputs):
        nevresim_out, _ = outputs
        assert nevresim_out[0, 0] == pytest.approx(1.0, abs=1e-9)
        assert nevresim_out[2, 0] == pytest.approx(1.0, abs=1e-9)
        assert np.all(nevresim_out <= 1.0 + 1e-9)

    def test_interior_values_pass_through_unchanged(self, outputs):
        nevresim_out, _ = outputs
        assert nevresim_out[0, 1] == pytest.approx(0.3, abs=1e-9)
        assert nevresim_out[1, 0] == pytest.approx(0.8, abs=1e-9)
        assert nevresim_out[1, 1] == pytest.approx(0.5, abs=1e-9)
