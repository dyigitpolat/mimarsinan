"""Pre-compiled nevresim segment execution and window-count assembly."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw
from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment


def run_prepared_segment(
    prepared: _PreparedSegment,
    input_data: list,
    *,
    simulation_length: int,
    spike_generation_mode: str,
    timeout_s: float | None,
    num_proc: int = 0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a neural segment's pre-compiled binary.

    Returns ``(counts, membranes_or_None)``. ``record_mode`` segments return
    WINDOW-gated counts assembled from the record build ([lat, lat+T) per
    core — the SSOT count currency); the plain and membrane-export builds
    return the stdout readout (raw), whose tail cycles keep integrating bias
    beyond the window."""
    if prepared.export_membrane and not prepared.record_mode:
        return run_binary_raw(
            binary_path=prepared.binary_path,
            work_dir=prepared.seg_dir,
            input_loader=input_data,
            output_size=prepared.output_size,
            simulation_length=int(simulation_length),
            input_size=prepared.input_size,
            spike_generation_mode=spike_generation_mode,
            max_input_count=len(input_data),
            num_proc=num_proc,
            export_membrane=True,
            timeout_s=timeout_s,
        )
    if prepared.record_mode:
        # Single-process keeps sample order aligned with the records.
        _raw, records = run_binary_raw(
            binary_path=prepared.binary_path,
            work_dir=prepared.seg_dir,
            input_loader=input_data,
            output_size=prepared.output_size,
            simulation_length=int(simulation_length),
            input_size=prepared.input_size,
            spike_generation_mode=spike_generation_mode,
            max_input_count=len(input_data),
            num_proc=1,
            record_spikes=True,
            timeout_s=timeout_s,
        )
        counts = window_counts_from_records(
            prepared, records, input_data, simulation_length,
        )
        membranes = None
        if prepared.membrane_binary_path is not None:
            _raw_m, membranes = run_binary_raw(
                binary_path=prepared.membrane_binary_path,
                work_dir=prepared.seg_dir,
                input_loader=input_data,
                output_size=prepared.output_size,
                simulation_length=int(simulation_length),
                input_size=prepared.input_size,
                spike_generation_mode=spike_generation_mode,
                max_input_count=len(input_data),
                num_proc=num_proc,
                export_membrane=True,
                timeout_s=timeout_s,
            )
        return counts, membranes
    raw = run_binary_raw(
        binary_path=prepared.binary_path,
        work_dir=prepared.seg_dir,
        input_loader=input_data,
        output_size=prepared.output_size,
        simulation_length=int(simulation_length),
        input_size=prepared.input_size,
        spike_generation_mode=spike_generation_mode,
        max_input_count=len(input_data),
        num_proc=num_proc,
        timeout_s=timeout_s,
    )
    return raw, None


def window_counts_from_records(
    prepared: _PreparedSegment,
    records: list,
    input_data: list,
    simulation_length: int,
) -> np.ndarray:
    """Assemble the segment's WINDOW-gated output counts from the record
    build: chip output wiring re-read through each source core's
    [lat, lat+T) window. Input passthroughs mirror the
    UniformSpikeGenerator comb count; always-on sources emit one spike per
    window cycle."""
    assert prepared.output_sources is not None
    T = int(simulation_length)
    n = len(input_data)
    out = np.zeros((n, len(prepared.output_sources)), dtype=np.float64)
    for s in range(n):
        rec = records[s]
        x = np.asarray(input_data[s][0], dtype=np.float64).reshape(-1)
        for j, (kind, core, neuron) in enumerate(prepared.output_sources):
            if kind == "core":
                out[s, j] = float(rec[core]["out"][neuron])
            elif kind == "input":
                comb_n = int(np.rint(x[neuron] * T))
                out[s, j] = float(min(max(comb_n, 0), T))
            elif kind == "on":
                out[s, j] = float(T)
    return out
