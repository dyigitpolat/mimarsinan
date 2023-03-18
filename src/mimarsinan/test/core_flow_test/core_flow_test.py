from mimarsinan.models.core_flow import *
from mimarsinan.mapping.softcore_mapping import *
from mimarsinan.mapping.mapping_utils import *

from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *

from mimarsinan.test.test_utils import *

import time

def get_model():
    activation = nn.ReLU()
    return nn.Sequential(
        nn.Linear(100, 10),
        activation
    )

def get_model_repr(input_shape, model):
    out = InputMapper(input_shape)
    out = ReshapeMapper(out, (input_shape[0], -1))
    out = LinearMapper(out, model[0])
    return ModelRepresentation(out)


def test_core_flow():
    input_count = 1
    test_input = torch.randn(input_count, 1, 10, 10)
    test_loader = [(test_input, torch.ones(1,10))]

    model = get_model()
    model_repr = get_model_repr(test_input.shape[1:], model)
    
    print("Mapping model to soft cores...")
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)
    print("  Number of soft cores:", len(soft_core_mapping.cores))
    print("  Soft core mapping delay: ", ChipDelay(soft_core_mapping).calculate())

    print("Mapping soft cores to hard cores...")
    axons_per_core = 256
    neurons_per_core = 256
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)
    print("  Number of hard cores:", len(hard_core_mapping.cores))
    print("  Hard core mapping delay: ", ChipDelay(hard_core_mapping).calculate())

    print("Forward pass with test input on CoreFlow model...")
    cf = CoreFlow(test_input.shape, hard_core_mapping)
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
    generated_files_path = "../generated/core_flow/"
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
    original_out = model(test_input[:, -1].flatten()).squeeze()
    for i in range(len(chip_output)):
        if(not almost_equal(chip_output[i], original_out[i])):
            inc +=1
        tot += 1
    print("  inc", inc)
    print("  tot", tot)

    print("Comparing outputs (cf vs original)...")
    tot = 0
    inc = 0
    for i in range(len(original_out)):
        if(not almost_equal(original_out[i], cf_out[i])):
            inc +=1
        tot += 1
    print("  inc", inc)
    print("  tot", tot)

    print(chip_output)
    print(original_out)
    print(cf_out)






