from mimarsinan.models.omihub_mlp_mixer import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.test.test_utils import *
from mimarsinan.mapping.omihub_mlp_mixer_mapper import *

from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *

import torch
import time

def test_mapping():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    mnist_input_size = 28*28
    mnist_output_size = 10
    epochs = 1

    mnist_output_size = 10
    input_count = 1

    ann_model = MLPMixer()

    generated_files_path = "../generated/mapping/"
    test_input = torch.randn(input_count, 1, 28, 28)

    out_size = 10
    test_loader = [(test_input, torch.ones(1,out_size))]

    ann_model = MLPMixer(1, 28, 4, 32, 33, 34, 1, out_size)

    ann_model(test_input)

    print("Mapping trained model to chip...")
    chip = omihub_mlp_mixer_to_chip(ann_model, leak=0, quantize=False, weight_type=float)

    print("Saving trained weights and chip generation code...")
    save_inputs_to_files(generated_files_path, test_loader, input_count)
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function_for_real_valued_exec(generated_files_path)

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    start_time = time.time()
    chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)

    print("chipsum", sum(chip_output))
    tot = 0
    inc = 0
    layer_debug_tensor = ann_model.debug.detach().numpy().flatten()
    for i in range(len(chip_output)):
        if(abs(chip_output[i]-layer_debug_tensor[i]) > 0.001):
            inc +=1
        tot += 1
    
    print("inc", inc)
    print("tot", tot)

    print("Mapping test done.")