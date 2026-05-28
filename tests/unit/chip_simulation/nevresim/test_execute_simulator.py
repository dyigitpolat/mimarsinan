"""Multiprocess nevresim stdout collection and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mimarsinan.chip_simulation.nevresim.execute_nevresim import execute_simulator
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import build_synthetic_mapping
from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw
from mimarsinan.common.build_utils import find_cpp20_compiler
from mimarsinan.common.file_utils import input_to_file
from mimarsinan.mapping.latency.chip import ChipLatency


NEVRESIM_ROOT = Path(__file__).resolve().parents[4] / "nevresim"


def _have_nevresim() -> bool:
    return NEVRESIM_ROOT.is_dir() and find_cpp20_compiler()[0] is not None


@pytest.fixture(scope="module")
def compiled_segment(tmp_path_factory):
    """Small runtime-mode binary for execute stress tests."""
    base = tmp_path_factory.mktemp("nevresim_execute")
    hcm, input_size = build_synthetic_mapping("contiguous_input", 2, 4, 2)
    latency = ChipLatency(hcm).calculate()
    T = 4

    NevresimDriver.nevresim_path = str(NEVRESIM_ROOT)
    driver = NevresimDriver(
        input_size,
        hcm,
        str(base),
        float,
        spike_generation_mode="Deterministic",
        firing_mode="Default",
        spiking_mode="lif",
        connectivity_mode="runtime",
        verbose=False,
    )
    binary = driver.emit_main_and_compile(1, T, latency)
    assert binary is not None
    return {
        "work_dir": str(base),
        "binary": binary,
        "input_size": input_size,
        "output_size": driver.chip.output_size,
        "T": T,
        "latency": latency,
    }


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
def test_execute_simulator_multiprocess_stress(compiled_segment) -> None:
    num_samples = 100
    num_proc = 8
    output_size = compiled_segment["output_size"]
    input_size = compiled_segment["input_size"]
    work_dir = Path(compiled_segment["work_dir"])

    inputs_dir = work_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for i in range(num_samples):
        vec = np.ones(input_size, dtype=np.float64) * 0.7
        input_to_file(vec, 0, str(inputs_dir / f"{i}.txt"))

    raw = execute_simulator(
        compiled_segment["binary"],
        num_samples,
        num_proc,
        expected_values=num_samples * output_size,
    )
    assert len(raw) == num_samples * output_size
    arr = np.array(raw).reshape(num_samples, output_size)
    assert arr.shape == (num_samples, output_size)


@pytest.mark.skipif(not _have_nevresim(), reason="nevresim or C++20 compiler unavailable")
def test_run_binary_raw_hybrid_precompiled_path(compiled_segment) -> None:
    num_samples = 50
    num_proc = 4
    output_size = compiled_segment["output_size"]
    input_size = compiled_segment["input_size"]

    loader = [
        (np.ones(input_size, dtype=np.float64) * 0.6, np.zeros(1))
        for _ in range(num_samples)
    ]
    raw = run_binary_raw(
        binary_path=compiled_segment["binary"],
        work_dir=compiled_segment["work_dir"],
        input_loader=loader,
        output_size=output_size,
        simulation_length=compiled_segment["T"],
        input_size=input_size,
        spike_generation_mode="Deterministic",
        max_input_count=num_samples,
        num_proc=num_proc,
    )
    assert raw.shape == (num_samples, output_size)
