"""Hybrid multi-segment nevresim execution."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import torch

from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping, SegmentIOSlice
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    resolve_stage_compute_scales,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_run.hybrid_stage_runner import run_hybrid_stages
from mimarsinan.chip_simulation.spiking_semantics import is_analytical_ttfs, requires_ttfs_firing
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.nevresim.segment_execute import run_binary_raw
from mimarsinan.chip_simulation.simulation_runner.emit import _PreparedSegment, _emit_and_compile_segment
from mimarsinan.chip_simulation.simulation_runner.host_contract import SimulationHostContract


class SimulationHybridMixin(SimulationHostContract):
    @staticmethod
    def _get_compute_op_output_size(op: ComputeOp, state_sizes: Dict[int, int]) -> int:
        """Infer flat output size for a ComputeOp. Uses output_shape when set, else dummy execute."""
        if op.output_shape is not None:
            return int(np.prod(op.output_shape))
        batch_size = 1
        dummy_input = torch.zeros(batch_size, max(state_sizes.get(-2, 1), 1), dtype=torch.float32)
        dummy_buffers = {
            k: torch.zeros(batch_size, sz, dtype=torch.float32)
            for k, sz in state_sizes.items()
            if k >= 0
        }
        result = op.execute(dummy_input, dummy_buffers)
        return int(result.shape[1])

    def _prepare_all_segments(
        self, hybrid: HybridHardCoreMapping
    ) -> Dict[int, _PreparedSegment]:
        """Emit all segment params and compile nevresim binaries in parallel (keyed by segment idx)."""
        stages = hybrid.stages
        num_samples = len(self.test_data)
        original_input = np.stack([d[0] for d in self.test_data])
        original_input = original_input.reshape(original_input.shape[0], -1)

        state_sizes: Dict[int, int] = {-2: original_input.shape[1]}
        segment_specs: List[Tuple[int, str, HardCoreMapping, int, int]] = []

        for stage in stages:
            if stage.kind == "neural":
                seg_mapping = stage.hard_core_mapping
                assert seg_mapping is not None

                input_size = max((s.offset + s.size for s in stage.input_map), default=0)
                seg_idx = len(segment_specs)
                seg_dir = os.path.abspath(
                    os.path.join(self.working_directory, f"segment_{seg_idx}")
                )
                latency = ChipLatency(seg_mapping).calculate()
                segment_specs.append((seg_idx, seg_dir, seg_mapping, input_size, latency))

                for s in stage.output_map:
                    state_sizes[s.node_id] = max(
                        state_sizes.get(s.node_id, 0), s.offset + s.size
                    )

            elif stage.kind == "compute":
                assert stage.compute_op is not None
                out_size = self._get_compute_op_output_size(stage.compute_op, state_sizes)
                state_sizes[stage.compute_op.id] = out_size

        num_segs = len(segment_specs)
        print(f"  Emitting parameters and compiling {num_segs} segment(s) in parallel "
              f"(stage kinds: {[s.kind for s in stages]})...")

        nevresim_path = NevresimDriver.nevresim_path
        assert nevresim_path is not None
        sim_length = int(self.simulation_length)

        prepared: Dict[int, _PreparedSegment] = {}
        if num_segs == 0:
            return prepared

        max_workers = min(num_segs, max(1, (os.cpu_count() or 2) // 2))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _emit_and_compile_segment,
                    seg_idx, seg_dir, seg_mapping, input_size, latency,
                    self.weight_type,
                    self.threshold_type,
                    self.spike_generation_mode,
                    self.firing_mode,
                    self.thresholding_mode,
                    self.spiking_mode,
                    num_samples,
                    sim_length,
                    nevresim_path,
                    self.nevresim_connectivity_mode,
                )
                for seg_idx, seg_dir, seg_mapping, input_size, latency in segment_specs
            ]
            for future in as_completed(futures):
                seg = future.result()
                prepared[seg.seg_idx] = seg

        print(f"  All {num_segs} segment(s) ready")
        return prepared

    def _run_neural_segment_precompiled(
        self,
        prepared: _PreparedSegment,
        input_data: list,
        num_proc: int = 0,
    ) -> np.ndarray:
        """Run a neural segment using its pre-compiled binary.

        Skips NevresimDriver creation entirely — only saves inputs and executes.
        Returns raw output as ``(num_samples, num_outputs)``.
        """
        max_input_count = len(input_data)
        return run_binary_raw(
            binary_path=prepared.binary_path,
            work_dir=prepared.seg_dir,
            input_loader=input_data,
            output_size=prepared.output_size,
            simulation_length=int(self.simulation_length),
            input_size=prepared.input_size,
            spike_generation_mode=self.spike_generation_mode,
            max_input_count=max_input_count,
            num_proc=num_proc,
        )

    def _run_neural_segment(
        self,
        segment_idx: int,
        hard_core_mapping: HardCoreMapping,
        input_data: list,
        input_size: int,
    ) -> np.ndarray:
        """Fallback: run a single neural segment from scratch (emit + compile + execute)."""
        seg_dir = os.path.join(self.working_directory, f"segment_{segment_idx}")
        os.makedirs(seg_dir, exist_ok=True)

        delay = ChipLatency(hard_core_mapping).calculate()
        print(f"  [segment {segment_idx}] delay: {delay}")

        driver = NevresimDriver(
            input_size,
            hard_core_mapping,
            seg_dir,
            self.weight_type,
            spike_generation_mode=self.spike_generation_mode,
            firing_mode=self.firing_mode,
            thresholding_mode=self.thresholding_mode,
            spiking_mode=self.spiking_mode,
            threshold_type=self.threshold_type,
            connectivity_mode=self.nevresim_connectivity_mode,
        )
        return driver.predict_spiking_raw(
            input_data,
            int(self.simulation_length),
            delay,
        )

    def _raw_to_rates(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw nevresim output to [0,1] rates.

        Analytical TTFS returns real-valued activations directly; LIF and cascaded
        ``ttfs_cycle_based`` return spike counts decoded as ``count / T``.
        """
        if is_analytical_ttfs(self.spiking_mode):
            return raw
        return raw / max(int(self.simulation_length), 1)

    def _run_hybrid(self, hybrid: HybridHardCoreMapping) -> float:
        """Execute a multi-stage hybrid mapping using the state buffer."""
        stages = hybrid.stages
        num_samples = len(self.test_data)
        is_ttfs = requires_ttfs_firing(self.spiking_mode)

        original_input = np.stack([d[0] for d in self.test_data])
        original_input = original_input.reshape(original_input.shape[0], -1)
        state_buffer: Dict[int, np.ndarray] = {-2: original_input}

        prepared_segments = self._prepare_all_segments(hybrid)

        seg_counter = 0

        def on_neural(_idx, stage, buf):
            nonlocal seg_counter
            seg_mapping = stage.hard_core_mapping
            assert seg_mapping is not None
            seg_input = self._assemble_segment_input_np(
                stage.input_map, buf, num_samples
            )
            input_size = seg_input.shape[1]
            seg_data = [(seg_input[i], np.zeros(1)) for i in range(num_samples)]
            print(
                f"  Running neural segment '{stage.name}' (input_size={input_size})"
            )
            prepared = prepared_segments[seg_counter]
            raw_output = self._run_neural_segment_precompiled(prepared, seg_data)
            seg_counter += 1
            rates = self._raw_to_rates(raw_output)
            self._store_segment_output_np(stage.output_map, buf, rates)

        def on_compute(_idx, stage, buf):
            assert stage.compute_op is not None
            print(f"  Executing compute op '{stage.name}' on host")
            op_id = stage.compute_op.id
            ttfs_in_scale, ttfs_out_scale = resolve_stage_compute_scales(
                hybrid, op_id, apply_ttfs=is_ttfs, op=stage.compute_op,
            )
            buf[op_id] = self._execute_compute_op_np(
                stage.compute_op,
                original_input,
                buf,
                ttfs_in_scale=ttfs_in_scale,
                ttfs_out_scale=ttfs_out_scale,
            )

        run_hybrid_stages(hybrid, state_buffer, on_neural=on_neural, on_compute=on_compute)

        final_output = self._gather_final_output_np(
            hybrid.output_sources, state_buffer, original_input, num_samples
        )
        predictions = np.argmax(final_output, axis=1)

        print("Evaluating simulator output...")
        return self._evaluate_chip_output(predictions)

    @staticmethod
    def _assemble_segment_input_np(
        input_map: list[SegmentIOSlice],
        state_buffer: Dict[int, np.ndarray],
        num_samples: int,
    ) -> np.ndarray:
        return assemble_segment_input_numpy(input_map, state_buffer, num_samples)

    @staticmethod
    def _store_segment_output_np(
        output_map: list[SegmentIOSlice],
        state_buffer: Dict[int, np.ndarray],
        output: np.ndarray,
    ) -> None:
        store_segment_output_numpy(output_map, state_buffer, output)

    @staticmethod
    def _execute_compute_op_np(
        op: ComputeOp,
        original_input: np.ndarray,
        state_buffer: Dict[int, np.ndarray],
        ttfs_in_scale: float = 1.0,
        ttfs_out_scale: float | None = None,
    ) -> np.ndarray:
        return execute_compute_op_numpy(
            op,
            original_input,
            state_buffer,
            in_scale=ttfs_in_scale,
            out_scale=ttfs_out_scale,
        )

    @staticmethod
    def _gather_final_output_np(
        output_sources: np.ndarray,
        state_buffer: Dict[int, np.ndarray],
        original_input: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        return gather_final_output_numpy(
            output_sources, state_buffer, original_input, num_samples,
        )
