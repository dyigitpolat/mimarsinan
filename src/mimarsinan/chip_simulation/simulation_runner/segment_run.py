"""Pre-compiled nevresim segment execution and window-count assembly."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw
from mimarsinan.chip_simulation.recording.spike_modes import comb_spike_count_np
from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment


def run_prepared_segment(
    prepared: _PreparedSegment,
    input_data: list,
    *,
    simulation_length: int,
    spike_generation_mode: str,
    timeout_s: float | None,
    num_proc: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, list | None]:
    """Run a neural segment's pre-compiled binary.

    Returns ``(counts, membranes_or_None, spike_trains_or_None)``.
    ``record_mode`` segments return WINDOW-gated counts assembled from the
    record build ([lat, lat+T) per core — the SSOT count currency); the plain
    and membrane-export builds return the stdout readout (raw), whose tail
    cycles keep integrating bias beyond the window. ``record_trains`` segments
    (verbatim pass-carry producers) additionally return per-sample
    ``{core: [bitstring per neuron]}`` trains in producer-local time."""
    mode = prepared.input_mode or spike_generation_mode
    if prepared.export_membrane and not prepared.record_mode:
        raw_or_pair = run_binary_raw(
            binary_path=prepared.binary_path,
            work_dir=prepared.seg_dir,
            input_loader=input_data,
            output_size=prepared.output_size,
            simulation_length=int(simulation_length),
            input_size=prepared.input_size,
            spike_generation_mode=mode,
            max_input_count=len(input_data),
            num_proc=num_proc,
            export_membrane=True,
            timeout_s=timeout_s,
        )
        return raw_or_pair[0], raw_or_pair[1], None
    if prepared.record_mode:
        trains: list | None = None
        # Single-process keeps sample order aligned with the records.
        if prepared.record_trains:
            _raw, records, trains = run_binary_raw(
                binary_path=prepared.binary_path,
                work_dir=prepared.seg_dir,
                input_loader=input_data,
                output_size=prepared.output_size,
                simulation_length=int(simulation_length),
                input_size=prepared.input_size,
                spike_generation_mode=mode,
                max_input_count=len(input_data),
                num_proc=1,
                record_spikes=True,
                record_spike_trains=True,
                timeout_s=timeout_s,
            )
        else:
            _raw, records = run_binary_raw(
            binary_path=prepared.binary_path,
            work_dir=prepared.seg_dir,
            input_loader=input_data,
            output_size=prepared.output_size,
            simulation_length=int(simulation_length),
            input_size=prepared.input_size,
            spike_generation_mode=mode,
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
                spike_generation_mode=mode,
                max_input_count=len(input_data),
                num_proc=num_proc,
                export_membrane=True,
                timeout_s=timeout_s,
            )
        return counts, membranes, trains
    raw = run_binary_raw(
        binary_path=prepared.binary_path,
        work_dir=prepared.seg_dir,
        input_loader=input_data,
        output_size=prepared.output_size,
        simulation_length=int(simulation_length),
        input_size=prepared.input_size,
        spike_generation_mode=mode,
        max_input_count=len(input_data),
        num_proc=num_proc,
        timeout_s=timeout_s,
    )
    return raw, None, None


def raster_from_spike_trains(
    trains: dict,
    output_sources: "list[tuple[str, int, int]]",
    T: int,
    input_train: "np.ndarray | None",
) -> np.ndarray:
    """One sample's ``(T, n_out)`` segment-output raster from SPKTRN trains.

    SPKTRN records are already PRODUCER-LOCAL, so this is a row gather with no
    time shift: ``core`` sources take the neuron's bitstring, ``on`` sources
    spike every cycle, ``input`` passthroughs replay the segment input train
    (which a SpikeTrain-mode segment has and a value-mode one reconstructs via
    the encoder twin)."""
    out = np.zeros((int(T), len(output_sources)), dtype=np.uint8)
    for j, (kind, core, neuron) in enumerate(output_sources):
        if kind == "on":
            out[:, j] = 1
        elif kind == "input":
            if input_train is None:
                raise ValueError(
                    "an input-kind output source needs the segment input train "
                    "to be carried verbatim; none was provided"
                )
            out[:, j] = input_train[:, neuron]
        elif kind == "core":
            rows = trains.get(core)
            if rows is None or neuron >= len(rows):
                continue
            bits = rows[neuron]
            width = min(int(T), len(bits))
            out[:width, j] = np.frombuffer(
                bits[:width].encode(), dtype=np.uint8) - ord("0")
    return out


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
                if (prepared.input_mode or "") == "SpikeTrain":
                    # The input IS a train (cycle-major): count its spikes.
                    out[s, j] = float(
                        x.reshape(T, prepared.input_size)[:, neuron].sum())
                else:
                    # The chip's llround tie rule, not np.rint (half-to-even).
                    comb_n = int(comb_spike_count_np(x[neuron], T))
                    out[s, j] = float(min(max(comb_n, 0), T))
            elif kind == "on":
                out[s, j] = float(T)
    return out
