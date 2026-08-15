from __future__ import annotations

import os
from dataclasses import dataclass

from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.connectivity import ConnectivityMode
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator
from mimarsinan.chip_simulation.execution_bounds import run_tasks_in_pool_bounded
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import (
    execution_neural_stages,
)
from mimarsinan.chip_simulation.simulation_runner.membrane_probe import (
    membrane_readout_slices,
)
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.support.schedule.pass_cut import VERBATIM
from mimarsinan.models.spiking.hybrid.carry import carried_output_ids
from typing import Dict

import numpy as np


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
    # Verbatim pass-carry roles (streamed scheduled runs only): a PRODUCING
    # segment's record build also defines NEVRESIM_RECORD_SPIKE_TRAINS and the
    # runner extracts its output rasters; a CONSUMING segment is compiled in
    # SpikeTrain input mode and the runner assembles its input train (carried
    # slices verbatim, host slices via the SSOT encoder twin).
    record_trains: bool = False
    input_mode: "str | None" = None
    carried_output_node_ids: "tuple[int, ...]" = ()


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
    record_trains: bool = False,
    input_mode: str | None = None,
    carried_output_node_ids: tuple[int, ...] = (),
) -> _PreparedSegment:
    """Top-level function for ProcessPoolExecutor: emit chip artifacts and compile."""
    NevresimDriver.nevresim_path = nevresim_path
    os.makedirs(seg_dir, exist_ok=True)

    if record_trains and not record_mode:
        raise ValueError(
            f"segment {seg_idx}: record_trains rides the record build "
            f"(spiking lif segments); a non-record segment cannot extract trains"
        )
    driver = NevresimDriver(
        input_size,
        seg_mapping,
        seg_dir,
        weight_type,
        spike_generation_mode=input_mode or spike_generation_mode,
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
        if record_trains:
            extra_flags.append("-DNEVRESIM_RECORD_SPIKE_TRAINS")
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
        record_trains=record_trains,
        input_mode=input_mode,
        carried_output_node_ids=tuple(carried_output_node_ids),
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


def prepare_all_segments(runner, hybrid) -> "Dict[int, _PreparedSegment]":
    """Emit all segment params and compile nevresim binaries in parallel (keyed by segment idx)."""
    stages = hybrid.stages
    num_samples = len(runner.test_data)
    original_input = np.stack([d[0] for d in runner.test_data])
    original_input = original_input.reshape(original_input.shape[0], -1)

    state_sizes: Dict[int, int] = {-2: original_input.shape[1]}
    segment_specs: list = []

    # Verbatim pass-carry roles (streamed scheduled runs): which stages'
    # outputs a later pass of the same segment replays, and which node ids
    # therefore cross as trains. COLLAPSE runs leave both empty and the
    # whole path below is byte-identical to before.
    carried_by_stage = (
        carried_output_ids(hybrid)
        if runner.pass_transfer == VERBATIM else {}
    )
    carried_node_ids = (
        set().union(*carried_by_stage.values()) if carried_by_stage else set()
    )

    for stage_index, stage in enumerate(stages):
        if stage.kind == "neural":
            # [C3 fused] a re-timed stage compiles one binary per LEVEL
            # stage — the execution units the shared stage loop runs.
            for exec_stage in execution_neural_stages(stage):
                seg_mapping = exec_stage.hard_core_mapping
                assert seg_mapping is not None
                input_size = max(
                    (s.offset + s.size for s in exec_stage.input_map), default=0)
                seg_idx = len(segment_specs)
                seg_dir = os.path.abspath(
                    os.path.join(runner.working_directory, f"segment_{seg_idx}"))
                # [C2] segments sourcing final-only output nodes compile the
                # NEVRESIM_EXPORT_MEMBRANE build when the honesty gate is armed.
                export_membrane = bool(runner.membrane_readout) and bool(
                    membrane_readout_slices(hybrid, exec_stage))
                # [nevresim parity] lif segments consume the WINDOW-gated
                # record counts ([lat, lat+T) per core) as their output —
                # the SSOT count currency — instead of the whole-program
                # stdout readout, whose tail cycles keep integrating bias
                # beyond the window. Membrane-export segments build BOTH
                # binaries: counts from the record build, membranes from
                # the export build (the C2 decode side-channel).
                record_mode = runner.spiking_mode == "lif"
                carried_out = carried_by_stage.get(stage_index, ())
                consumes_carry = any(
                    int(sl.node_id) in carried_node_ids
                    for sl in exec_stage.input_map
                )
                segment_specs.append(
                    (seg_idx, seg_dir, seg_mapping, input_size,
                     ChipLatency(seg_mapping).calculate(), export_membrane,
                     record_mode, bool(carried_out),
                     "SpikeTrain" if consumes_carry else None,
                     tuple(carried_out)))
                for s in exec_stage.output_map:
                    state_sizes[s.node_id] = max(
                        state_sizes.get(s.node_id, 0), s.offset + s.size)

        elif stage.kind == "compute":
            assert stage.compute_op is not None
            out_size = runner._get_compute_op_output_size(stage.compute_op, state_sizes)
            state_sizes[stage.compute_op.id] = out_size

    num_segs = len(segment_specs)
    print(f"  Emitting parameters and compiling {num_segs} segment(s) in parallel "
          f"(stage kinds: {[s.kind for s in stages]})...")

    nevresim_path = NevresimDriver.nevresim_path
    assert nevresim_path is not None
    sim_length = int(runner.simulation_length)

    if num_segs == 0:
        return {}

    max_workers = min(num_segs, max(1, (os.cpu_count() or 2) // 2))
    timeout_s = runner.simulation_step_timeout_s
    task_args = {
        seg_idx: (
            seg_idx, seg_dir, seg_mapping, input_size, latency,
            runner.weight_type,
            runner.threshold_type,
            runner.spike_generation_mode,
            runner.firing_mode,
            runner.thresholding_mode,
            runner.spiking_mode,
            num_samples,
            sim_length,
            nevresim_path,
            runner.nevresim_connectivity_mode,
            timeout_s,
            export_membrane,
            record_mode,
            record_trains,
            input_mode,
            carried_out,
        )
        for seg_idx, seg_dir, seg_mapping, input_size, latency,
            export_membrane, record_mode, record_trains, input_mode,
            carried_out in segment_specs
    }
    prepared: Dict[int, _PreparedSegment] = run_tasks_in_pool_bounded(
        _emit_and_compile_segment,
        task_args,
        max_workers=max_workers,
        timeout_s=timeout_s,
        description="nevresim segment emit+compile pool",
    )

    print(f"  All {num_segs} segment(s) ready")
    return prepared

