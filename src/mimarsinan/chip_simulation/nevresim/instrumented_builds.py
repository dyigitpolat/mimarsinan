"""Dedicated nevresim diagnostic builds: spike-recording and membrane-export binaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator
from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw


def emit_and_compile_flagged(
    driver: Any,
    max_input_count: int,
    simulation_length: int,
    latency: int,
    *,
    extra_flag: str,
    label: str,
    output_path: str | None = None,
) -> str:
    """Compile a dedicated ``extra_flag`` build (never cached — keep it off the
    production binary path). ``output_path`` defaults to a flag-specific name so
    the production binary is never clobbered."""
    if output_path is None:
        suffix = extra_flag.removeprefix("-DNEVRESIM_").lower()
        output_path = str(
            Path(driver.generated_files_path) / "bin" / f"simulator_{suffix}"
        )
    driver.emit_main(max_input_count, simulation_length, latency, verbose=False)
    binary = compile_simulator(
        driver.generated_files_path,
        type(driver).nevresim_path,
        output_path=output_path,
        verbose=False,
        extra_flags=[extra_flag],
        timeout_s=driver.simulation_step_timeout_s,
    )
    if binary is None:
        raise Exception(f"{label} build compilation failed.")
    return binary


def predict_spiking_raw_with_records(
    driver: Any,
    input_loader,
    simulation_length: int,
    latency: int,
    max_input_count: int | None = None,
    num_proc: int = 1,
):
    """Run a ``NEVRESIM_RECORD_SPIKES`` build and return ``(raw, spike_records)``
    — per-sample ``{core: {"in","out"}}`` counts windowed to ``[lat, lat+T)``,
    the nevresim analogue of HCM ``CoreSpikeCounts`` (single-process to keep
    sample order)."""
    if max_input_count is None:
        max_input_count = len(input_loader)
    binary = emit_and_compile_flagged(
        driver, max_input_count, simulation_length, latency,
        extra_flag="-DNEVRESIM_RECORD_SPIKES", label="Recording",
    )
    return run_binary_raw(
        binary_path=binary,
        work_dir=driver.generated_files_path,
        input_loader=input_loader,
        output_size=driver.chip.output_size,
        simulation_length=int(simulation_length),
        input_size=driver.chip.input_size,
        spike_generation_mode=driver.spike_generation_mode,
        max_input_count=max_input_count,
        num_proc=num_proc,
        record_spikes=True,
        timeout_s=driver.simulation_step_timeout_s,
    )


def predict_spiking_raw_with_membrane(
    driver: Any,
    input_loader,
    simulation_length: int,
    latency: int,
    max_input_count: int | None = None,
    num_proc: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a ``NEVRESIM_EXPORT_MEMBRANE`` build; return ``(raw_counts,
    membranes)``, both ``(num_samples, num_outputs)`` — counts identical to the
    default build, membranes the final ``m_T/theta`` per output neuron (charge
    identity ``Q_T = theta*c_T + m_T``)."""
    if max_input_count is None:
        max_input_count = len(input_loader)
    binary = emit_and_compile_flagged(
        driver, max_input_count, simulation_length, latency,
        extra_flag="-DNEVRESIM_EXPORT_MEMBRANE", label="Membrane-export",
    )
    return run_binary_raw(
        binary_path=binary,
        work_dir=driver.generated_files_path,
        input_loader=input_loader,
        output_size=driver.chip.output_size,
        simulation_length=int(simulation_length),
        input_size=driver.chip.input_size,
        spike_generation_mode=driver.spike_generation_mode,
        max_input_count=max_input_count,
        num_proc=num_proc,
        export_membrane=True,
        timeout_s=driver.simulation_step_timeout_s,
    )
