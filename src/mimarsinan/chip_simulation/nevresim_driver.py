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
        self, input_buffer_size, hard_core_mapping, generated_files_path, weight_type, spike_generation_mode = "Stochastic", firing_mode = "Default"):
        assert NevresimDriver.nevresim_path is not None, "nevresim path is not set."

        self.spike_generation_mode = spike_generation_mode
        self.firing_mode = firing_mode
        self.weight_type = weight_type

        self.chip = hard_cores_to_chip(
            input_buffer_size,
            hard_core_mapping,
            hard_core_mapping.axons_per_core,
            hard_core_mapping.neurons_per_core,
            leak=0,
            weight_type=self.weight_type)
        
        chip_json = self.chip.get_chip_json()
        # save json str to file
        with open(generated_files_path + "/chip.json", "w") as f:
            f.write(chip_json)
        
        self.generated_files_path = generated_files_path
        save_weights_and_chip_code(self.chip, generated_files_path)
        
    def _simulator_output_to_predictions(self, simulator_output, number_of_classes):
        total_spikes = sum(simulator_output)
        print("  Total spikes: {}".format(total_spikes))

        prediction_count = int(len(simulator_output) / number_of_classes)
        output_array = np.array(simulator_output).reshape((prediction_count, number_of_classes))

        predictions = np.zeros(prediction_count, dtype=int)
        for i in range(prediction_count):
            predictions[i] = np.argmax(output_array[i])
        return predictions
    
    def _prepare_simulator(self, max_input_count, simulation_length, latency, spike_generation_mode, firing_mode):
        print("firing mode: ", firing_mode)
        generate_main_function(
            self.generated_files_path, max_input_count, self.chip.output_size, simulation_length, latency,
            main_cpp_template, get_config(spike_generation_mode, firing_mode, self.weight_type.__name__))
        
        self.simulator_filename = \
            compile_simulator(self.generated_files_path, NevresimDriver.nevresim_path)
        
        if self.simulator_filename is None:
            raise Exception("Compilation failed.")

    def predict_spiking(self, input_loader, simulation_length, latency, max_input_count=None, num_proc=50):
        if max_input_count is None:
            max_input_count = len(input_loader)
            print("  Max input count:", max_input_count)
        
        save_inputs_to_files(self.generated_files_path, input_loader, max_input_count)
        
        self._prepare_simulator(max_input_count, simulation_length, latency, self.spike_generation_mode, self.firing_mode)
        simulator_output = execute_simulator(self.simulator_filename, max_input_count, num_proc)
        
        return self._simulator_output_to_predictions(simulator_output, self.chip.output_size)