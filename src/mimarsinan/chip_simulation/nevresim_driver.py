from mimarsinan.mapping.mapping_utils import *
from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *

import numpy as np

class NevresimDriver:
    nevresim_path = None

    def __init__(
        self, input_buffer_size, hard_core_mapping, generated_files_path, weight_type):
        assert NevresimDriver.nevresim_path is not None, "nevresim path is not set."

        self.weight_type = weight_type

        self.chip = hard_cores_to_chip(
            input_buffer_size,
            hard_core_mapping,
            hard_core_mapping.axons_per_core,
            hard_core_mapping.neurons_per_core,
            leak=0,
            weight_type=self.weight_type)
        
        self.generated_files_path = generated_files_path
        save_weights_and_chip_code(self.chip, generated_files_path)

        self.simulator_filename = \
            compile_simulator(generated_files_path, NevresimDriver.nevresim_path)
        
        if self.simulator_filename is None:
            raise Exception("Compilation failed.")
        
    def __simulator_output_to_predictions(self, simulator_output, number_of_classes):
        total_spikes = sum(simulator_output)
        print("  Total spikes: {}".format(total_spikes))

        prediction_count = int(len(simulator_output) / number_of_classes)
        output_array = np.array(simulator_output).reshape((prediction_count, number_of_classes))

        predictions = np.zeros(prediction_count, dtype=int)
        for i in range(prediction_count):
            predictions[i] = np.argmax(output_array[i])
        return predictions

    def predict_spiking(self, input_loader, simulation_length, max_input_count=None, num_proc=50):
        if max_input_count is None:
            max_input_count = len(input_loader)
            print("  Max input count:", max_input_count)
        
        save_inputs_to_files(self.generated_files_path, input_loader, max_input_count)
        generate_main_function(
            self.generated_files_path, max_input_count, self.chip.output_size, simulation_length,
            main_cpp_template, get_config("Stochastic", "Novena", self.weight_type.__name__))
        
        simulator_output = execute_simulator(self.simulator_filename, max_input_count, num_proc)
        
        return self.__simulator_output_to_predictions(simulator_output, self.chip.output_size)