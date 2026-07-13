"""Flat nevresim mapping execution."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.simulation_runner.host_contract import SimulationHostContract
from mimarsinan.chip_simulation.simulation_runner.membrane_probe import flat_membrane_slices


class SimulationFlatMixin(SimulationHostContract):
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
            simulation_step_timeout_s=self.simulation_step_timeout_s,
        )

        simulation_steps = int(self.simulation_length)
        print(f"  total simulation steps: {simulation_steps}")

        eligible_slices = flat_membrane_slices(
            self.mapping, armed=bool(self.membrane_readout),
        )
        if eligible_slices:
            print(
                "  [C2] applying the deployed membrane decode to the probe's "
                f"final read ({len(eligible_slices)} eligible node slice(s))"
            )
            raw, membranes = simulation_driver.predict_spiking_raw_with_membrane(
                self.test_data, simulation_steps, delay,
            )
            corrected = np.asarray(raw, dtype=np.float64).copy()
            half_step = float(self.membrane_half_step_charge)
            for _node_id, start, end in eligible_slices:
                corrected[:, start:end] += membranes[:, start:end] - half_step
            predictions = np.argmax(corrected, axis=1)
        else:
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
