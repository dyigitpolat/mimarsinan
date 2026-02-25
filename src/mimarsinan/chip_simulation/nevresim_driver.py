from mimarsinan.mapping.mapping_utils import *
from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *

import numpy as np
import json


class NevresimDriver:
    nevresim_path = None

    def __init__(
        self, input_buffer_size, hard_core_mapping, generated_files_path, weight_type,
        spike_generation_mode="Stochastic", firing_mode="Default", spiking_mode="rate",
        verbose=True,
    ):
        """Create driver and emit chip artifacts (JSON, weights, chip code). Does not compile."""
        assert NevresimDriver.nevresim_path is not None, "nevresim path is not set."

        self.spike_generation_mode = spike_generation_mode
        self.firing_mode = firing_mode
        self.spiking_mode = spiking_mode
        self.weight_type = weight_type

        self.chip = hard_cores_to_chip(
            input_buffer_size,
            hard_core_mapping,
            hard_core_mapping.axons_per_core,
            hard_core_mapping.neurons_per_core,
            leak=0,
            weight_type=self.weight_type)

        chip_json = self.chip.get_chip_json()
        with open(generated_files_path + "/chip.json", "w") as f:
            f.write(chip_json)

        self.generated_files_path = generated_files_path
        save_weights_and_chip_code(self.chip, generated_files_path, verbose=verbose)

    def _simulator_output_to_predictions(self, simulator_output, number_of_classes):
        total_spikes = sum(simulator_output)
        print("  Total spikes: {}".format(total_spikes))

        prediction_count = int(len(simulator_output) / number_of_classes)
        output_array = np.array(simulator_output).reshape((prediction_count, number_of_classes))

        predictions = np.zeros(prediction_count, dtype=int)
        for i in range(prediction_count):
            predictions[i] = np.argmax(output_array[i])
        return predictions

    def emit_main(self, max_input_count, simulation_length, latency, verbose=True):
        """Generate main.cpp only (no compilation)."""
        if verbose:
            print(f"spiking mode: {self.spiking_mode}, firing mode: {self.firing_mode}")
        generate_main_function(
            self.generated_files_path, max_input_count, self.chip.output_size, simulation_length, latency,
            main_cpp_template, get_config(
                self.spike_generation_mode, self.firing_mode,
                self.weight_type.__name__, self.spiking_mode),
            verbose=verbose)

    def emit_main_and_compile(
        self, max_input_count, simulation_length, latency, output_path=None
    ) -> str | None:
        """Generate main.cpp and compile. Returns binary path or None on failure."""
        self.emit_main(max_input_count, simulation_length, latency)
        simulator_filename = compile_simulator(
            self.generated_files_path, NevresimDriver.nevresim_path, output_path=output_path
        )
        if simulator_filename is None:
            raise Exception("Compilation failed.")
        return simulator_filename

    def _run_simulator(
        self, input_loader, simulation_length, latency, max_input_count=None, num_proc=50,
        simulator_filename=None,
    ):
        """Save inputs, optionally compile, run, return raw float list."""
        if max_input_count is None:
            max_input_count = len(input_loader)
            print("  Max input count:", max_input_count)

        save_inputs_to_files(self.generated_files_path, input_loader, max_input_count)

        if simulator_filename is None:
            simulator_filename = self.emit_main_and_compile(
                max_input_count, simulation_length, latency
            )

        simulator_output = execute_simulator(simulator_filename, max_input_count, num_proc)
        return simulator_output, max_input_count

    def predict_spiking(self, input_loader, simulation_length, latency, max_input_count=None, num_proc=50):
        simulator_output, _ = self._run_simulator(
            input_loader, simulation_length, latency, max_input_count, num_proc)
        return self._simulator_output_to_predictions(simulator_output, self.chip.output_size)

    def predict_spiking_raw(self, input_loader, simulation_length, latency, max_input_count=None, num_proc=50):
        """Run simulation and return raw spike counts as (num_samples, num_outputs)."""
        simulator_output, max_input_count = self._run_simulator(
            input_loader, simulation_length, latency, max_input_count, num_proc)
        num_outputs = self.chip.output_size
        prediction_count = int(len(simulator_output) / num_outputs)
        return np.array(simulator_output).reshape((prediction_count, num_outputs))

    def predict_spiking_raw_from_binary(
        self,
        simulator_filename: str,
        input_loader,
        simulation_length,
        latency,
        max_input_count=None,
        num_proc=50,
    ) -> np.ndarray:
        """Run simulation using a pre-compiled binary. Does not compile."""
        simulator_output, max_input_count = self._run_simulator(
            input_loader, simulation_length, latency, max_input_count, num_proc,
            simulator_filename=simulator_filename,
        )
        num_outputs = self.chip.output_size
        prediction_count = int(len(simulator_output) / num_outputs)
        return np.array(simulator_output).reshape((prediction_count, num_outputs))