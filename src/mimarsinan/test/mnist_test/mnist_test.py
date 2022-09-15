from cgi import test
from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *

import torch

from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

def test_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_input_size = 28*28
    mnist_output_size = 10
    inner_mlp_width = 64
    inner_mlp_count = 4
    epochs = 10

    ann_model = SimpleMLP(
        inner_mlp_width, 
        inner_mlp_count, 
        mnist_input_size, 
        mnist_output_size)

    print("Training model...")
    train_on_mnist(ann_model, device, epochs)

    generated_files_path = "../generated/mnist/"
    simulation_length = 200
    input_count = 100

    _, test_loader = get_mnist_data(1)

    print("Mapping trained model to chip...")
    chip = simple_mlp_to_chip(ann_model)

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
    _, test_loader = get_mnist_data(1)
    accuracy = evaluate_chip_output(chip_output, test_loader, mnist_output_size)

    print("SNN accuracy on MNIST is:", accuracy*100, "%")
    print("MNIST test done.")