from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.test_subsample import compute_test_subsample_indices
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.compile_nevresim import compile_simulator
from mimarsinan.chip_simulation.nevresim.execute_nevresim import execute_simulator
from mimarsinan.common.file_utils import save_inputs_to_files
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader


@dataclass
class _PreparedSegment:
    """Result of parallel emit+compile for one neural segment."""
    seg_idx: int
    seg_dir: str
    binary_path: str
    output_size: int


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
    )
    driver.emit_main(num_samples, sim_length, latency, verbose=False)

    output_path = os.path.join(seg_dir, "bin", "simulator")
    binary = compile_simulator(
        seg_dir, nevresim_path, output_path=output_path, verbose=False
    )
    if binary is None:
        raise RuntimeError(f"Compilation failed for segment {seg_idx}")

    return _PreparedSegment(
        seg_idx=seg_idx,
        seg_dir=seg_dir,
        binary_path=os.path.abspath(binary),
        output_size=driver.chip.output_size,
    )
