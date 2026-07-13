"""Shared nevresim binary execution: input serialization, run, reshape."""

from __future__ import annotations

from typing import Iterable, Literal, overload

import numpy as np

from mimarsinan.chip_simulation.nevresim.execute_nevresim import execute_simulator_full
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


@overload
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
    record_spikes: Literal[False] = False,
    export_membrane: Literal[False] = False,
    timeout_s: float | None = None,
) -> np.ndarray: ...


@overload
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
    record_spikes: Literal[True],
    export_membrane: Literal[False] = False,
    timeout_s: float | None = None,
) -> tuple[np.ndarray, list[dict[int, dict[str, list[int]]]]]: ...


@overload
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
    record_spikes: Literal[False] = False,
    export_membrane: Literal[True],
    timeout_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]: ...


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
    export_membrane: bool = False,
    timeout_s: float | None = None,
):
    """Run a pre-compiled nevresim binary and return ``(num_samples, output_size)``.

    With ``record_spikes`` (a ``NEVRESIM_RECORD_SPIKES`` build) returns
    ``(raw, spike_records)`` — per-sample per-core counts from ``parse_spike_records``.
    With ``export_membrane`` (a ``NEVRESIM_EXPORT_MEMBRANE`` build) returns
    ``(raw, membranes)`` — per-sample final ``m_T/theta`` per output, failing
    loud when the binary emits no membrane records."""
    if record_spikes and export_membrane:
        raise ValueError(
            "record_spikes and export_membrane are exclusive per run_binary_raw "
            "call; compile and run dedicated binaries"
        )
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
    raw, spike_records, membrane_rows = execute_simulator_full(
        binary_path,
        max_input_count,
        num_proc,
        expected_values=expected_values,
        record_spikes=record_spikes,
        export_membrane=export_membrane,
        timeout_s=timeout_s,
    )
    out = np.array(raw, dtype=np.float64).reshape((max_input_count, output_size))
    if record_spikes:
        return out, spike_records
    if export_membrane:
        if len(membrane_rows) != max_input_count or any(
            len(row) != output_size for row in membrane_rows
        ):
            raise RuntimeError(
                f"nevresim membrane export mismatch: expected {max_input_count} "
                f"MEMB rows of {output_size} values, got "
                f"{[len(row) for row in membrane_rows]} — was the binary "
                "compiled with -DNEVRESIM_EXPORT_MEMBRANE?"
            )
        return out, np.array(membrane_rows, dtype=np.float64)
    return out
