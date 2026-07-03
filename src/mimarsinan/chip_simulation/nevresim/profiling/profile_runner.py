"""Run a single nevresim emit+compile profile for a synthetic or real mapping."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from mimarsinan.chip_simulation.nevresim.compile_cache import (
    NevresimCompileCache,
    cache_key,
    mapping_connectivity_hash,
    policy_hash,
)
from mimarsinan.chip_simulation.nevresim.compile_nevresim import CompileResult, compile_simulator
from mimarsinan.chip_simulation.nevresim.connectivity import (
    ConnectivityMode,
    default_nevresim_connectivity_mode,
)
from mimarsinan.chip_simulation.nevresim.execute_nevresim import execute_simulator
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver, _python_to_cpp_type_name
from mimarsinan.chip_simulation.nevresim.profiling.compile_profile import (
    NevresimCompileProfile,
    ProfileTimer,
    metrics_for_mapping,
)
from mimarsinan.code_generation.generate_main import generate_main_function, generate_main_function_runtime, get_config
from mimarsinan.code_generation.main_cpp_template import main_cpp_template
from mimarsinan.common.file_utils import input_to_file, prepare_containing_directory, save_weights_and_chip_code
from mimarsinan.mapping.export.chip_export import hard_cores_to_chip
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore.hard_core_mapping import HardCoreMapping


def profile_mapping_compile(
    hcm: HardCoreMapping,
    input_size: int,
    *,
    experiment: str = "",
    preset: str = "",
    nevresim_path: str,
    out_dir: str | Path | None = None,
    connectivity_mode: ConnectivityMode | None = None,
    simulation_length: int = 4,
    num_execute_samples: int = 1,
    compile: bool = True,
    execute: bool = False,
    optimization: str = "-O3",
    time_trace: bool = False,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
) -> NevresimCompileProfile:
    """Emit artifacts, optionally compile and execute; return profiling record."""
    if connectivity_mode is None:
        connectivity_mode = default_nevresim_connectivity_mode()
    row = NevresimCompileProfile(experiment=experiment, preset=preset)
    row.apply_mapping_metrics(metrics_for_mapping(hcm))

    weight_type = float
    threshold_type = float
    wt_cpp = _python_to_cpp_type_name(weight_type)
    tt_cpp = _python_to_cpp_type_name(threshold_type)

    cleanup = out_dir is None
    work_dir = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="nevresim_profile_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ProfileTimer() as export_timer:
            chip = hard_cores_to_chip(
                input_size,
                hcm,
                hcm.axons_per_core,
                hcm.neurons_per_core,
                leak=0,
                weight_type=weight_type,
                threshold_type=threshold_type,
            )
        row.export_s = export_timer.elapsed

        with ProfileTimer() as codegen_timer:
            save_weights_and_chip_code(
                chip,
                str(work_dir),
                verbose=False,
                connectivity_mode=connectivity_mode,
                weight_cpp=wt_cpp,
                threshold_cpp=tt_cpp,
            )
            latency = ChipLatency(hcm).calculate()
            sim_cfg = get_config(
                spike_gen_mode="Deterministic",
                firing_mode="Default",
                weight_type=wt_cpp,
                spiking_mode="lif",
                threshold_type=tt_cpp,
            )
            if connectivity_mode == "runtime":
                generate_main_function_runtime(
                    str(work_dir),
                    num_execute_samples,
                    chip.output_size,
                    simulation_length,
                    latency,
                    simulation_config=sim_cfg,
                    verbose=False,
                )
            else:
                generate_main_function(
                    str(work_dir),
                    num_execute_samples,
                    chip.output_size,
                    simulation_length,
                    latency,
                    main_cpp_template,
                    sim_cfg,
                    verbose=False,
                )
        row.codegen_s = codegen_timer.elapsed
        row.record_artifact_sizes(work_dir)

        if not compile:
            return row

        m_hash = mapping_connectivity_hash(hcm)
        p_hash = policy_hash(
            spiking_mode="lif",
            spike_generation_mode="Deterministic",
            firing_mode="Default",
            thresholding_mode="<=",
            weight_type_name=wt_cpp,
            threshold_type_name=tt_cpp,
            simulation_length=simulation_length,
            latency=latency,
            connectivity_mode=connectivity_mode,
        )
        key = cache_key(m_hash, p_hash)
        binary_path = work_dir / "bin" / "simulator"

        cache: NevresimCompileCache | None = None
        if cache_dir is not None:
            cache = NevresimCompileCache(cache_dir)
            cached = cache.get_binary(key)
            if cached is not None:
                row.compile_s = 0.0
                row.compile_success = True
                row.extra["cache_hit"] = True
                if execute:
                    _execute_samples(work_dir, cached, input_size, num_execute_samples, row)
                return row

        NevresimDriver.nevresim_path = nevresim_path
        compile_result = compile_simulator(
            str(work_dir),
            nevresim_path,
            output_path=str(binary_path),
            verbose=verbose,
            optimization=optimization,
            time_trace=time_trace,
            trace_output_dir=work_dir,
            return_timing=True,
        )
        if isinstance(compile_result, CompileResult):
            row.compile_s = compile_result.compile_s
            row.compile_success = compile_result.success
            row.compiler = compile_result.compiler
            row.compiler_family = compile_result.compiler_family
            if compile_result.trace_json:
                row.extra["trace_json"] = compile_result.trace_json
            resolved_binary = compile_result.binary_path
        else:
            row.compile_success = compile_result is not None
            resolved_binary = compile_result

        if row.compile_success and cache is not None and resolved_binary:
            cache.store_binary(key, resolved_binary, metadata=row.as_dict())

        if execute and row.compile_success and resolved_binary:
            _execute_samples(work_dir, Path(resolved_binary), input_size, num_execute_samples, row)

        return row
    finally:
        if cleanup and out_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)


def _execute_samples(
    work_dir: Path,
    binary: Path,
    input_size: int,
    num_samples: int,
    row: NevresimCompileProfile,
) -> None:
    inputs_dir = work_dir / "inputs"
    prepare_containing_directory(str(inputs_dir / "0.txt"))
    for i in range(num_samples):
        vec = np.ones(input_size, dtype=np.float64) * 0.5
        input_to_file(vec, 0, str(inputs_dir / f"{i}.txt"))
    with ProfileTimer() as exec_timer:
        execute_simulator(str(binary), num_samples, num_proc=1)
    row.execute_s = exec_timer.elapsed