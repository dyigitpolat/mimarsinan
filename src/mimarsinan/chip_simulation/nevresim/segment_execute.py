"""Shared nevresim binary execution: input serialization, run, reshape."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from mimarsinan.chip_simulation.nevresim.execute_nevresim import execute_simulator
from mimarsinan.common.file_utils import (
    save_inputs_to_files,
    save_spike_train_inputs_to_files,
)


def save_segment_inputs(
    work_dir: str,
    input_loader: Iterable,
    max_input_count: int,
    *,
    spike_generation_mode: str,
    input_size: int,
    simulation_length: int,
) -> None:
    """Write per-sample input files for a nevresim segment binary."""
    if spike_generation_mode == "SpikeTrain":
        save_spike_train_inputs_to_files(
            work_dir,
            input_loader,
            max_input_count,
            input_size=input_size,
            simulation_length=int(simulation_length),
        )
    else:
        save_inputs_to_files(work_dir, input_loader, max_input_count)


def run_binary_raw(
    *,
    binary_path: str,
    work_dir: str,
    input_loader: Iterable,
    output_size: int,
    simulation_length: int,
    input_size: int,
    spike_generation_mode: str,
    max_input_count: int,
    num_proc: int = 0,
    record_spikes: bool = False,
):
    """Run a pre-compiled nevresim binary and return ``(num_samples, output_size)``.

    When ``record_spikes`` is set (a ``NEVRESIM_RECORD_SPIKES`` build), returns
    ``(raw, spike_records)`` — the per-sample per-core counts from
    :func:`parse_spike_records`.
    """
    samples = list(input_loader)[:max_input_count]
    if spike_generation_mode == "SpikeTrain":
        expected_flat = int(input_size) * int(simulation_length)
    else:
        expected_flat = int(input_size)
    for idx, (x, _y) in enumerate(samples):
        flat_size = int(np.asarray(x).reshape(-1).size)
        if flat_size != expected_flat:
            raise ValueError(
                f"sample {idx}: input length {flat_size} != expected {expected_flat} "
                f"(spike_generation_mode={spike_generation_mode!r})"
            )

    save_segment_inputs(
        work_dir,
        samples,
        max_input_count,
        spike_generation_mode=spike_generation_mode,
        input_size=input_size,
        simulation_length=simulation_length,
    )

    expected_values = max_input_count * output_size
    result = execute_simulator(
        binary_path,
        max_input_count,
        num_proc,
        expected_values=expected_values,
        record_spikes=record_spikes,
    )
    if record_spikes:
        raw, spike_records = result
        out = np.array(raw, dtype=np.float64).reshape((max_input_count, output_size))
        return out, spike_records
    return np.array(result, dtype=np.float64).reshape((max_input_count, output_size))
