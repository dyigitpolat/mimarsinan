"""Integration: static vs runtime nevresim connectivity parity on synthetic mapping."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import build_synthetic_mapping
from mimarsinan.common.build_utils import find_cpp20_compiler
from mimarsinan.mapping.latency.chip import ChipLatency


NEVRESIM_ROOT = Path(__file__).resolve().parents[2] / "nevresim"


def _have_nevresim() -> bool:
    return NEVRESIM_ROOT.is_dir() and find_cpp20_compiler()[0] is not None


def _run_parity_for_mode(
    *,
    spiking_mode: str,
    spike_generation_mode: str,
    firing_mode: str,
    threshold_type,
) -> dict[str, np.ndarray]:
    hcm, input_size = build_synthetic_mapping("contiguous_input", 3, 8, 4)
    latency = ChipLatency(hcm).calculate()
    T = 4

    class _Loader:
        def __iter__(self):
            x = np.ones(input_size, dtype=np.float64) * 0.8
            y = np.zeros(hcm.neurons_per_core, dtype=np.float64)
            y[0] = 1.0
            yield x, y

        def __len__(self):
            return 1

    NevresimDriver.nevresim_path = str(NEVRESIM_ROOT)
    outputs = {}
    for mode in ("compile_time", "runtime"):
        with tempfile.TemporaryDirectory() as tmp:
            driver = NevresimDriver(
                input_size,
                hcm,
                tmp,
                float,
                spike_generation_mode=spike_generation_mode,
                firing_mode=firing_mode,
                spiking_mode=spiking_mode,
                threshold_type=threshold_type,
                connectivity_mode=mode,
                verbose=False,
            )
            raw = driver.predict_spiking_raw(_Loader(), T, latency, max_input_count=1, num_proc=1)
            outputs[mode] = raw
    return outputs


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
def test_static_vs_runtime_output_parity_lif() -> None:
    outputs = _run_parity_for_mode(
        spiking_mode="lif",
        spike_generation_mode="Deterministic",
        firing_mode="Default",
        threshold_type=float,
    )
    np.testing.assert_array_equal(outputs["compile_time"], outputs["runtime"])


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
def test_static_vs_runtime_output_parity_ttfs() -> None:
    outputs = _run_parity_for_mode(
        spiking_mode="ttfs",
        spike_generation_mode="TTFS",
        firing_mode="TTFS",
        threshold_type=float,
    )
    np.testing.assert_allclose(outputs["compile_time"], outputs["runtime"], rtol=0, atol=1e-9)


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
def test_static_vs_runtime_output_parity_ttfs_quantized() -> None:
    outputs = _run_parity_for_mode(
        spiking_mode="ttfs_quantized",
        spike_generation_mode="TTFS",
        firing_mode="TTFS",
        threshold_type=float,
    )
    np.testing.assert_allclose(outputs["compile_time"], outputs["runtime"], rtol=0, atol=1e-9)
