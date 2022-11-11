from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.test.cifar10_test.cifar10_test_utils import *

from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

import torch

def test_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cifar10_h = 32
    cifar10_w = 32
    cifar10_c = 3
    cifar10_output_size = 10
    inner_mlp_width = 256
    inner_mlp_count = 3
    epochs = 20

    ann_model = SimpleMLPMixer(
        3, 3,
        256,
        16,
        256,
        inner_mlp_width,
        inner_mlp_count,
        [0, 0.3, 0.7, 1.0],
        [0, 0.3, 0.7, 1.0],
        cifar10_h,cifar10_w,cifar10_c, 
        cifar10_output_size)

    print("Training model...")
    train_on_cifar10(ann_model, device, epochs)

    generated_files_path = "../generated/cifar10/"
    simulation_length = 200
    input_count = 100

    _, test_loader = get_cifar10_data(1)

    print("Mapping trained model to chip...")
    chip = simple_mlp_to_chip(ann_model, leak=0.0)

    print("Saving trained weights and chip generation code...")
    save_inputs_to_files(generated_files_path, test_loader, input_count)
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function(generated_files_path, input_count, simulation_length)

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    chip_output = execute_simulator(simulator_filename)

    print("Evaluating simulator output...")
    _, test_loader = get_cifar10_data(1)
    accuracy = evaluate_chip_output(chip_output, test_loader, cifar10_output_size)

    print("SNN accuracy on CIFAR10 is:", accuracy*100, "%")
    print("CIFAR10 test done.")