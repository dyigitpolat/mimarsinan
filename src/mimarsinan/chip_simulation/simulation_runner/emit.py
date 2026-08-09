from __future__ import annotations

import os
from dataclasses import dataclass

from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.connectivity import ConnectivityMode
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator


@dataclass
class _PreparedSegment:
    """Result of parallel emit+compile for one neural segment.

    ``record_mode`` segments run the NEVRESIM_RECORD_SPIKES build and the
    runner assembles WINDOW-gated output counts ([lat, lat+T) per core) via
    ``output_sources`` — the count currency of the SSOT calculus — instead
    of the whole-program stdout readout (which keeps integrating bias
    through the pipeline tail cycles)."""
    seg_idx: int
    seg_dir: str
    binary_path: str
    output_size: int
    input_size: int
    export_membrane: bool = False
    record_mode: bool = False
    output_sources: "list[tuple[str, int, int]] | None" = None
    membrane_binary_path: "str | None" = None


def _emit_and_compile_segment(
    seg_idx: int,
    seg_dir: str,
    seg_mapping: HardCoreMapping,
    input_size: int,
    latency: int,
    weight_type,
    threshold_type,
    spike_generation_mode: str,
    firing_mode: str,
    thresholding_mode: str,
    spiking_mode: str,
    num_samples: int,
    sim_length: int,
    nevresim_path: str,
    connectivity_mode: ConnectivityMode,
    timeout_s: float | None = None,
    export_membrane: bool = False,
    record_mode: bool = False,
) -> _PreparedSegment:
    """Top-level function for ProcessPoolExecutor: emit chip artifacts and compile."""
    NevresimDriver.nevresim_path = nevresim_path
    os.makedirs(seg_dir, exist_ok=True)

    driver = NevresimDriver(
        input_size,
        seg_mapping,
        seg_dir,
        weight_type,
        spike_generation_mode=spike_generation_mode,
        firing_mode=firing_mode,
        thresholding_mode=thresholding_mode,
        spiking_mode=spiking_mode,
        threshold_type=threshold_type,
        verbose=False,
        connectivity_mode=connectivity_mode,
    )
    driver.emit_main(num_samples, sim_length, latency, verbose=False)

    if driver.chip.input_size != input_size:
        raise ValueError(
            f"segment {seg_idx}: hybrid input_map size {input_size} != "
            f"chip input_size {driver.chip.input_size}"
        )

    if record_mode:
        extra_flags = ["-DNEVRESIM_RECORD_SPIKES"]
    elif export_membrane:
        extra_flags = ["-DNEVRESIM_EXPORT_MEMBRANE"]
    else:
        extra_flags = None
    output_path = os.path.join(seg_dir, "bin", "simulator")
    binary = compile_simulator(
        seg_dir, nevresim_path, output_path=output_path, verbose=False,
        extra_flags=extra_flags,
        timeout_s=timeout_s,
    )
    if binary is None:
        raise RuntimeError(f"Compilation failed for segment {seg_idx}")

    membrane_binary = None
    if export_membrane and record_mode:
        # Both observables: window counts (record build, the count currency)
        # AND final membranes (export build, the C2 decode side-channel).
        membrane_binary = compile_simulator(
            seg_dir, nevresim_path,
            output_path=os.path.join(seg_dir, "bin", "simulator_membrane"),
            verbose=False,
            extra_flags=["-DNEVRESIM_EXPORT_MEMBRANE"],
            timeout_s=timeout_s,
        )
        if membrane_binary is None:
            raise RuntimeError(
                f"Membrane-build compilation failed for segment {seg_idx}"
            )

    return _PreparedSegment(
        seg_idx=seg_idx,
        seg_dir=seg_dir,
        binary_path=os.path.abspath(binary),
        output_size=driver.chip.output_size,
        input_size=driver.chip.input_size,
        export_membrane=export_membrane,
        record_mode=record_mode,
        output_sources=(
            _serialize_output_sources(driver.chip) if record_mode else None
        ),
        membrane_binary_path=(
            os.path.abspath(membrane_binary) if membrane_binary else None
        ),
    )


def _serialize_output_sources(chip) -> "list[tuple[str, int, int]]":
    """Picklable output-buffer wiring: (kind, core, neuron) per output index —
    the runner re-reads the chip's readout through the WINDOW-gated records."""
    out: list[tuple[str, int, int]] = []
    for s in chip.output_buffer:
        if s.is_off_:
            out.append(("off", 0, 0))
        elif s.is_input_:
            out.append(("input", 0, int(s.neuron_)))
        elif s.is_always_on_:
            out.append(("on", 0, 0))
        else:
            out.append(("core", int(s.core_), int(s.neuron_)))
    return out
