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
    
    input_count = 1
    generated_files_path = "../generated/mapping/"
    test_input = torch.randn(input_count, 3, 14, 14)

    out_size = 10
    test_loader = [(test_input, torch.ones(1,out_size))]
    ann_model = MLPMixer(3, 14, 7, 32, 33, 34, 1, out_size)

    print("Forward pass with test input on original model...")
    ann_model(test_input)

    print("Mapping model to soft cores...")
    model_repr = get_omihub_repr(test_input.shape[1:], ann_model)
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)
    print("  Number of soft cores:", len(soft_core_mapping.cores))
    print("  Soft core mapping delay: ", ChipLatency(soft_core_mapping).calculate())

    print("Mapping soft cores to hard cores...")
    axons_per_core = 256
    neurons_per_core = 256
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)
    print("  Number of hard cores:", len(hard_core_mapping.cores))
    print("  Hard core mapping delay: ", ChipLatency(hard_core_mapping).calculate())

    print("Creating CoreFlow model...")
    cf = CoreFlow(test_input.shape, hard_core_mapping)

    print("Forward pass with test input on CoreFlow model...")
    cf_out = cf(test_input).squeeze()

    print("Mapping hard cores to chip...")
    chip = hard_cores_to_chip(
        sum(test_input.shape[1:]),
        hard_core_mapping, 
        axons_per_core,
        neurons_per_core, 
        leak=0,
        weight_type=float)

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

    print("Comparing outputs (original vs chip)...")
    tot = 0
    inc = 0
    layer_debug_tensor = ann_model.debug.detach().numpy().flatten()
    for i in range(len(chip_output)):
        if(not almost_equal(chip_output[i], layer_debug_tensor[i])):
            inc +=1
        tot += 1

    print("inc", inc)
    print("tot", tot)

    print(chip_output)
    print(layer_debug_tensor)
    print(cf_out)
    print(cf_out.shape)
    
    print("Comparing outputs (cf vs chip)...")
    tot = 0
    inc = 0
    for i in range(len(chip_output)):
        if(not almost_equal(chip_output[i], cf_out[i])):
            inc +=1
        tot += 1
    
    print("inc", inc)
    print("tot", tot)

    print("Mapping test done.")