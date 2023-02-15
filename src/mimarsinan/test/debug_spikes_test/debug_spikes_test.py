from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *

from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

from mimarsinan.code_generation.main_cpp_template_debug_spikes import *

import time

def test_debug_spikes():
    input_size = 17
    simulation_length = 10

    generated_files_path = "../generated/debug_spikes/"
    input_count = 1
    output_count = 33

    chip = ChipModel()
    chip_json = open("../generated/debug_spikes/chip.json", "r").read()
    chip.load_from_json(chip_json)

    print("Saving trained weights and chip generation code...")
    input_path = generated_files_path + "inputs/0.txt"
    input_to_file(np.array([1.0 for _ in range(input_size)]), 0, input_path)
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function(
        generated_files_path, input_count, output_count, simulation_length, 
        main_cpp_template_debug_spikes, get_config("Deterministic", "Novena", "int"))

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    start_time = time.time()
    chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)

    print("Printing simulator output...")
    idx = 0
    print(idx, end=": ")
    for i in chip_output:
        if(i != -1):
            print(i, end=" ")
        else:
            print()
            idx += 1
            print(idx, end=": ")
    print()

    print("DEBUG SPIKES test done.")