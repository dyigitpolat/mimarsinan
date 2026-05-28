"""Flat nevresim mapping execution."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver


class SimulationFlatMixin:
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
            thresholding_mode=self.thresholding_mode,
            spiking_mode=self.spiking_mode,
            threshold_type=self.threshold_type,
            connectivity_mode=self.nevresim_connectivity_mode,
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

    @staticmethod
    def _compute_segment_input_size(hard_core_mapping: HardCoreMapping) -> int:
        """Determine the input buffer size for a neural segment."""
        max_idx = -1
        for hc in hard_core_mapping.cores:
            for src in hc.axon_sources:
                if getattr(src, "is_input_", False):
                    max_idx = max(max_idx, int(src.neuron_))
        return max_idx + 1 if max_idx >= 0 else 0
