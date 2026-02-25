from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
    SegmentIOSlice,
)
from mimarsinan.mapping.ir import ComputeOp, IRSource
from mimarsinan.mapping.softcore_mapping import HardCoreMapping
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.compile_nevresim import compile_simulator
from mimarsinan.chip_simulation.execute_nevresim import execute_simulator
from mimarsinan.common.file_utils import save_inputs_to_files
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory


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
    spike_generation_mode: str,
    firing_mode: str,
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
        spiking_mode=spiking_mode,
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


class SimulationRunner:
    def __init__(self, pipeline, mapping, simulation_length, preprocessor):
        self.spike_generation_mode = pipeline.config["spike_generation_mode"]
        self.firing_mode = pipeline.config["firing_mode"]
        self.spiking_mode = pipeline.config.get("spiking_mode", "rate")

        wt_q = pipeline.config.get("weight_quantization", True)
        self.weight_type = int if wt_q else float

        self.input_size = pipeline.config["input_size"]
        self.num_classes = pipeline.config["num_classes"]

        self.working_directory = pipeline.working_directory

        self.test_input = []
        self.test_targets = []

        data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        data_provider = data_loader_factory.create_data_provider()

        test_loader = data_loader_factory.create_test_loader(
            data_provider.get_test_batch_size(), data_provider)

        for xs, ys in test_loader:
            self.test_input.extend(preprocessor(xs).detach())
            self.test_targets.extend(ys)

        self.test_data = [*zip(np.stack(self.test_input), np.stack(self.test_targets))]

        max_samples = pipeline.config.get("max_simulation_samples", 0)
        total_samples = len(self.test_data)
        if max_samples and 0 < max_samples < total_samples:
            rng = np.random.RandomState(pipeline.config.get("seed", 0))
            indices = rng.choice(total_samples, size=max_samples, replace=False)
            self.test_data = [self.test_data[i] for i in indices]
            self.test_input = [self.test_input[i] for i in indices]
            self.test_targets = [self.test_targets[i] for i in indices]
            print(f"  [SimulationRunner] Subsampled {max_samples} / {total_samples} test samples")

        self.mapping = mapping
        self.simulation_length = simulation_length

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate_chip_output(self, predictions):
        confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=int)

        for (y, p) in zip(self.test_targets, predictions):
            confusion_matrix[y.item()][p] += 1
        print("Confusion matrix:")
        print(confusion_matrix)

        total = 0
        correct = 0
        for (y, p) in zip(self.test_targets, predictions):
            correct += int(y.item() == p)
            total += 1

        return float(correct) / total

    # ------------------------------------------------------------------
    # Single-segment nevresim (original path)
    # ------------------------------------------------------------------

    def _run_flat_mapping(self, hard_core_mapping: HardCoreMapping) -> float:
        """Run a flat (single-segment) HardCoreMapping through nevresim."""
        delay = ChipLatency(hard_core_mapping).calculate()
        print(f"  delay: {delay}")
        print(f"  simulation length: {self.simulation_length}")

        simulation_driver = NevresimDriver(
            self.input_size,
            hard_core_mapping,
            self.working_directory,
            self.weight_type,
            spike_generation_mode=self.spike_generation_mode,
            firing_mode=self.firing_mode,
            spiking_mode=self.spiking_mode,
        )

        simulation_steps = int(self.simulation_length)
        print(f"  total simulation steps: {simulation_steps}")

        predictions = simulation_driver.predict_spiking(
            self.test_data,
            simulation_steps,
            delay,
        )

        print("Evaluating simulator output...")
        accuracy = self._evaluate_chip_output(predictions)
        return accuracy

    # ------------------------------------------------------------------
    # Multi-segment nevresim (state-buffer driven)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_segment_input_size(hard_core_mapping: HardCoreMapping) -> int:
        """Determine the input buffer size for a neural segment."""
        max_idx = -1
        for hc in hard_core_mapping.cores:
            for src in hc.axon_sources:
                if getattr(src, "is_input_", False):
                    max_idx = max(max_idx, int(src.neuron_))
        return max_idx + 1 if max_idx >= 0 else 0

    @staticmethod
    def _get_compute_op_output_size(op: ComputeOp, state_sizes: Dict[int, int]) -> int:
        """Infer flat output size for a ComputeOp. Uses output_shape when set, else dummy execute."""
        if op.output_shape is not None:
            return int(np.prod(op.output_shape))
        # Dummy execute to infer shape
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
        """
        Emit all segment parameters and compile nevresim binaries in parallel.
        Returns dict mapping segment_idx -> _PreparedSegment.
        """
        stages = hybrid.stages
        num_samples = len(self.test_data)
        original_input = np.stack([d[0] for d in self.test_data])
        original_input = original_input.reshape(original_input.shape[0], -1)

        # ---- Fast sequential pass: compute input_size / latency per segment ----
        # Only arithmetic on IO-slice metadata — no chip creation or file I/O.
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

        # ---- Parallel pass: emit chip artifacts + compile for every segment ----
        num_segs = len(segment_specs)
        print(f"  Emitting parameters and compiling {num_segs} segment(s) in parallel...")

        nevresim_path = NevresimDriver.nevresim_path
        assert nevresim_path is not None
        sim_length = int(self.simulation_length)

        prepared: Dict[int, _PreparedSegment] = {}
        with ProcessPoolExecutor(max_workers=num_segs) as executor:
            futures = [
                executor.submit(
                    _emit_and_compile_segment,
                    seg_idx, seg_dir, seg_mapping, input_size, latency,
                    self.weight_type,
                    self.spike_generation_mode,
                    self.firing_mode,
                    self.spiking_mode,
                    num_samples,
                    sim_length,
                    nevresim_path,
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
        num_proc: int = 50,
    ) -> np.ndarray:
        """Run a neural segment using its pre-compiled binary.

        Skips NevresimDriver creation entirely — only saves inputs and executes.
        Returns raw output as ``(num_samples, num_outputs)``.
        """
        max_input_count = len(input_data)
        save_inputs_to_files(prepared.seg_dir, input_data, max_input_count)
        raw = execute_simulator(prepared.binary_path, max_input_count, num_proc)
        return np.array(raw).reshape((max_input_count, prepared.output_size))

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
            spiking_mode=self.spiking_mode,
        )
        return driver.predict_spiking_raw(
            input_data,
            int(self.simulation_length),
            delay,
        )

    def _raw_to_rates(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw nevresim output to [0,1] rates."""
        if self.spiking_mode in ("ttfs", "ttfs_quantized"):
            return raw
        return raw / max(int(self.simulation_length), 1)

    def _run_hybrid(self, hybrid: HybridHardCoreMapping) -> float:
        """Execute a multi-stage hybrid mapping using the state buffer."""
        stages = hybrid.stages
        num_samples = len(self.test_data)

        original_input = np.stack([d[0] for d in self.test_data])
        original_input = original_input.reshape(original_input.shape[0], -1)
        state_buffer: Dict[int, np.ndarray] = {-2: original_input}

        # Emit + compile all neural segments in parallel upfront
        prepared_segments = self._prepare_all_segments(hybrid)

        seg_counter = 0
        for stage in stages:
            if stage.kind == "neural":
                seg_mapping = stage.hard_core_mapping
                assert seg_mapping is not None

                seg_input = self._assemble_segment_input_np(
                    stage.input_map, state_buffer, num_samples
                )
                input_size = seg_input.shape[1]

                seg_data = [
                    (seg_input[i], np.zeros(1))
                    for i in range(num_samples)
                ]

                print(f"  Running neural segment '{stage.name}' "
                      f"(input_size={input_size})")

                prepared = prepared_segments[seg_counter]
                raw_output = self._run_neural_segment_precompiled(
                    prepared, seg_data,
                )
                seg_counter += 1

                rates = self._raw_to_rates(raw_output)
                self._store_segment_output_np(stage.output_map, state_buffer, rates)

            elif stage.kind == "compute":
                assert stage.compute_op is not None
                print(f"  Executing compute op '{stage.name}' on host")

                result = self._execute_compute_op_np(
                    stage.compute_op, original_input, state_buffer
                )
                state_buffer[stage.compute_op.id] = result

            else:
                raise ValueError(f"Unknown hybrid stage kind: {stage.kind}")

        final_output = self._gather_final_output_np(
            hybrid.output_sources, state_buffer, original_input, num_samples
        )
        predictions = np.argmax(final_output, axis=1)

        print("Evaluating simulator output...")
        return self._evaluate_chip_output(predictions)

    # ------------------------------------------------------------------
    # State-buffer helpers (numpy)
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble_segment_input_np(
        input_map: list[SegmentIOSlice],
        state_buffer: Dict[int, np.ndarray],
        num_samples: int,
    ) -> np.ndarray:
        total_size = max((s.offset + s.size for s in input_map), default=0)
        inp = np.zeros((num_samples, total_size), dtype=np.float32)
        for s in input_map:
            buf = state_buffer[s.node_id]
            inp[:, s.offset : s.offset + s.size] = buf[:, :s.size]
        return inp

    @staticmethod
    def _store_segment_output_np(
        output_map: list[SegmentIOSlice],
        state_buffer: Dict[int, np.ndarray],
        output: np.ndarray,
    ) -> None:
        for s in output_map:
            state_buffer[s.node_id] = output[:, s.offset : s.offset + s.size]

    @staticmethod
    def _execute_compute_op_np(
        op: ComputeOp,
        original_input: np.ndarray,
        state_buffer: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Execute a ComputeOp on host using the state buffer."""
        x_torch = torch.tensor(original_input, dtype=torch.float32)
        buffers_torch = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in state_buffer.items()
        }
        result = op.execute(x_torch, buffers_torch)
        return result.detach().numpy()

    @staticmethod
    def _gather_final_output_np(
        output_sources: np.ndarray,
        state_buffer: Dict[int, np.ndarray],
        original_input: np.ndarray,
        num_samples: int,
    ) -> np.ndarray:
        flat_sources = output_sources.flatten()
        out = np.zeros((num_samples, len(flat_sources)), dtype=np.float32)
        for idx, src in enumerate(flat_sources):
            if not isinstance(src, IRSource):
                continue
            if src.is_off():
                continue
            elif src.is_input():
                out[:, idx] = original_input[:, src.index]
            elif src.is_always_on():
                out[:, idx] = 1.0
            else:
                out[:, idx] = state_buffer[src.node_id][:, src.index]
        return out

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> float:
        if isinstance(self.mapping, HybridHardCoreMapping):
            segments = self.mapping.get_neural_segments()
            compute_ops = self.mapping.get_compute_ops()

            if len(segments) == 1 and len(compute_ops) == 0:
                return self._run_flat_mapping(segments[0])

            print(f"  Hybrid mapping: {len(segments)} neural segments, "
                  f"{len(compute_ops)} compute ops")
            return self._run_hybrid(self.mapping)

        # Legacy: plain HardCoreMapping (backward compatibility)
        return self._run_flat_mapping(self.mapping)
