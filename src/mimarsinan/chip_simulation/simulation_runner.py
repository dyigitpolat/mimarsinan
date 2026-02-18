from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.hybrid_hardcore_mapping import HybridHardCoreMapping, HybridStage
from mimarsinan.mapping.ir import ComputeOp
from mimarsinan.mapping.softcore_mapping import HardCoreMapping
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory


class SimulationRunner:
    def __init__(self, pipeline, mapping, simulation_length, preprocessor):
        self.spike_generation_mode = pipeline.config["spike_generation_mode"]
        self.firing_mode = pipeline.config["firing_mode"]
        self.spiking_mode = pipeline.config.get("spiking_mode", "rate")

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
            int,
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
    # Multi-segment nevresim
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_segment_input_size(hard_core_mapping: HardCoreMapping) -> int:
        """Determine the input buffer size for a neural segment.

        The input buffer size is ``1 + max(index)`` over all axon sources
        that are flagged as external inputs (is_input_ == True /
        node_id == -2).
        """
        max_idx = -1
        for hc in hard_core_mapping.cores:
            for src in hc.axon_sources:
                if getattr(src, "is_input_", False):
                    max_idx = max(max_idx, int(src.neuron_))
        return max_idx + 1 if max_idx >= 0 else 0

    @staticmethod
    def _execute_compute_op(op: ComputeOp, in_rates: np.ndarray) -> np.ndarray:
        """Execute a ComputeOp on host in rate-space.

        ``in_rates`` has shape ``(num_samples, N)`` with values in [0, 1].
        Returns ``(num_samples, M)`` with values clamped to [0, 1].
        """
        x = torch.tensor(in_rates, dtype=torch.float32)
        batch_size = x.shape[0]

        if op.input_shape is not None:
            x = x.view(batch_size, *op.input_shape)

        if op.op_type == "max_pool2d":
            y = F.max_pool2d(
                x,
                kernel_size=op.params.get("kernel_size", 2),
                stride=op.params.get("stride", None),
                padding=op.params.get("padding", 0),
            )
        elif op.op_type == "avg_pool2d":
            y = F.avg_pool2d(
                x,
                kernel_size=op.params.get("kernel_size", 2),
                stride=op.params.get("stride", None),
                padding=op.params.get("padding", 0),
            )
        elif op.op_type == "adaptive_avg_pool2d":
            y = F.adaptive_avg_pool2d(x, op.params.get("output_size", (1, 1)))
        elif op.op_type in ("flatten", "identity"):
            y = x
        else:
            raise NotImplementedError(
                f"ComputeOp '{op.op_type}' not implemented in SimulationRunner"
            )

        return y.view(batch_size, -1).clamp(0.0, 1.0).numpy()

    def _run_neural_segment(
        self,
        segment_idx: int,
        hard_core_mapping: HardCoreMapping,
        input_data: list,
        input_size: int,
    ) -> np.ndarray:
        """Run a single neural segment through nevresim.

        Returns raw spike counts as ``(num_samples, num_outputs)``.
        """
        seg_dir = os.path.join(self.working_directory, f"segment_{segment_idx}")
        os.makedirs(seg_dir, exist_ok=True)

        delay = ChipLatency(hard_core_mapping).calculate()
        print(f"  [segment {segment_idx}] delay: {delay}")

        driver = NevresimDriver(
            input_size,
            hard_core_mapping,
            seg_dir,
            int,
            spike_generation_mode=self.spike_generation_mode,
            firing_mode=self.firing_mode,
            spiking_mode=self.spiking_mode,
        )

        simulation_steps = int(self.simulation_length)
        raw_output = driver.predict_spiking_raw(
            input_data,
            simulation_steps,
            delay,
        )
        return raw_output

    def _raw_to_rates(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw nevresim output to [0,1] rates.

        For TTFS modes the output is already in activation space [0, 1];
        for rate-coded modes it is spike counts that must be normalised
        by ``simulation_length``.
        """
        if self.spiking_mode in ("ttfs", "ttfs_quantized"):
            return raw
        return raw / max(int(self.simulation_length), 1)

    def _run_hybrid(self, hybrid: HybridHardCoreMapping) -> float:
        """Execute a multi-stage hybrid mapping through chained nevresim calls."""
        stages = hybrid.stages

        current_data: list = list(self.test_data)
        current_input_size: int = self.input_size
        last_raw_output: np.ndarray | None = None

        seg_counter = 0
        for stage in stages:
            if stage.kind == "neural":
                seg_mapping = stage.hard_core_mapping
                assert seg_mapping is not None

                print(f"  Running neural segment '{stage.name}' "
                      f"(input_size={current_input_size})")

                last_raw_output = self._run_neural_segment(
                    seg_counter, seg_mapping, current_data, current_input_size,
                )
                seg_counter += 1

                # Convert to rates for potential next stage
                rates = self._raw_to_rates(last_raw_output)
                current_data = [
                    (rates[i], np.zeros(1))
                    for i in range(rates.shape[0])
                ]
                current_input_size = int(rates.shape[1])

            elif stage.kind == "compute":
                assert stage.compute_op is not None
                assert last_raw_output is not None

                print(f"  Executing compute op '{stage.name}' on host")
                rates = self._raw_to_rates(last_raw_output)
                transformed = self._execute_compute_op(stage.compute_op, rates)

                current_data = [
                    (transformed[i], np.zeros(1))
                    for i in range(transformed.shape[0])
                ]
                current_input_size = int(transformed.shape[1])
                last_raw_output = None  # must go through another neural segment
            else:
                raise ValueError(f"Unknown hybrid stage kind: {stage.kind}")

        # Final predictions from the last neural segment's output
        assert last_raw_output is not None, "Hybrid mapping must end with a neural segment"
        predictions = np.argmax(last_raw_output, axis=1)

        print("Evaluating simulator output...")
        return self._evaluate_chip_output(predictions)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> float:
        if isinstance(self.mapping, HybridHardCoreMapping):
            segments = self.mapping.get_neural_segments()
            compute_ops = self.mapping.get_compute_ops()

            if len(segments) == 1 and len(compute_ops) == 0:
                # Single neural segment — use the optimised flat path.
                return self._run_flat_mapping(segments[0])

            print(f"  Hybrid mapping: {len(segments)} neural segments, "
                  f"{len(compute_ops)} compute ops")
            return self._run_hybrid(self.mapping)

        # Legacy: plain HardCoreMapping (backward compatibility)
        return self._run_flat_mapping(self.mapping)
