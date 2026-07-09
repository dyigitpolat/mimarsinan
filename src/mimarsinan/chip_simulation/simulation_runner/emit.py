from __future__ import annotations

import os
from dataclasses import dataclass

from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.connectivity import ConnectivityMode
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator


@dataclass
class _PreparedSegment:
    """Result of parallel emit+compile for one neural segment."""
    seg_idx: int
    seg_dir: str
    binary_path: str
    output_size: int
    input_size: int


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

    output_path = os.path.join(seg_dir, "bin", "simulator")
    binary = compile_simulator(
        seg_dir, nevresim_path, output_path=output_path, verbose=False,
        timeout_s=timeout_s,
    )
    if binary is None:
        raise RuntimeError(f"Compilation failed for segment {seg_idx}")

    return _PreparedSegment(
        seg_idx=seg_idx,
        seg_dir=seg_dir,
        binary_path=os.path.abspath(binary),
        output_size=driver.chip.output_size,
        input_size=driver.chip.input_size,
    )
