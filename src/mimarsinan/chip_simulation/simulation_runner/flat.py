"""Flat nevresim mapping execution."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver
from mimarsinan.chip_simulation.hybrid_run.output_decode import (
    bare_mapping_output_gauge, decode_held, program_output_gauge)
from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import HybridHardCoreMapping
from mimarsinan.chip_simulation.simulation_runner.host_contract import SimulationHostContract
from mimarsinan.chip_simulation.simulation_runner.membrane_probe import flat_membrane_slices


class SimulationFlatMixin(SimulationHostContract):
    def _run_flat_mapping(self, hard_core_mapping: HardCoreMapping) -> float:
        """Run a flat (single-segment) HardCoreMapping through nevresim."""
        # [U-TGAUGE2] the readout's classes are the value domain's: each output at its program's gauge; [U-TGAUGE3] a
        # bare HardCoreMapping carries no gauge table: its counts rank only at one output gauge, else it refuses by name
        gauge = (program_output_gauge(self.mapping, timing=requires_ttfs_firing(self.spiking_mode))
                 if isinstance(self.mapping, HybridHardCoreMapping) else bare_mapping_output_gauge(self.mapping))
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
            point=self.spiking_point,
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
            predictions = np.argmax(decode_held(corrected, gauge), axis=1)
        else:
            predictions, total_spikes = simulation_driver.predict_spiking(
                self.test_data,
                simulation_steps,
                delay,
                output_gauge=gauge,
            )
            # [W4.3] the driver's measured total-output-spikes read, formerly
            # printed-and-dropped (deployment_record_schema.md §4).
            self.nevresim_total_spikes = float(total_spikes)

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
