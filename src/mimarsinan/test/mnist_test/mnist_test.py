from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *

from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

import torch
import time

def test_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_input_size = 28*28
    mnist_output_size = 10
    inner_mlp_width = 256
    inner_mlp_count = 1
    pretrain_epochs = 3
    cq_only_epochs = 3
    cq_quantize_epochs = 3

    Tq = 37
    simulation_length = Tq + 3

    generated_files_path = "../generated/mnist/"
    input_count = 10000

    _, test_loader = get_mnist_data(1)

    ann_model = SimpleMLP(
        inner_mlp_width, 
        inner_mlp_count, 
        mnist_input_size, 
        mnist_output_size,
        bias=False)

    print("Pretraining model...")
    train_on_mnist(ann_model, device, pretrain_epochs)

    print("Tuning model with CQ...")
    cq_ann_model = SimpleMLP_CQ(ann_model, Tq)
    train_on_mnist(cq_ann_model, device, cq_only_epochs)

    print("Tuning model with CQ and weight quantization...")
    train_on_mnist_quantized(cq_ann_model, device, cq_quantize_epochs)

    #######################################

    print("Mapping trained model to chip...")
    chip = simple_mlp_to_chip(cq_ann_model, leak=0, quantize=True, weight_type=int)

    print("Saving trained weights and chip generation code...")
    save_inputs_to_files(generated_files_path, test_loader, input_count)
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function(
        generated_files_path, input_count, mnist_output_size, simulation_length,
        main_cpp_template, get_config("Deterministic", "Novena", "int"))

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    start_time = time.time()
    chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)

    print("Evaluating simulator output...")
    _, test_loader = get_mnist_data(1)
    accuracy = evaluate_chip_output(chip_output, test_loader, mnist_output_size, verbose=True)
    print("SNN accuracy on MNIST is:", accuracy*100, "%")

    #######################################

    print("Mapping trained model to chip...")
    chip = simple_mlp_to_chip_v2(cq_ann_model, leak=0, quantize=True, weight_type=int)

    print("Saving trained weights and chip generation code...")
    save_inputs_to_files(generated_files_path, test_loader, input_count)
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function(
        generated_files_path, input_count, mnist_output_size, simulation_length,
        main_cpp_template, get_config("Deterministic", "Novena", "int"))

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    start_time = time.time()
    chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)

    print("Evaluating simulator output...")
    _, test_loader = get_mnist_data(1)
    accuracy = evaluate_chip_output(chip_output, test_loader, mnist_output_size, verbose=True)
    print("SNN accuracy on MNIST is:", accuracy*100, "%")

    #export_json_to_file(chip, generated_files_path + "chip.json")
    print("MNIST test done.")