from mimarsinan.test.cifar10_test.cifar10_test_utils import *

from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

import torch
import json

def test_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cifar10_output_size = 10
    epochs = 40

    parameters_json = """
{"patch_cols": 3, "patch_rows": 6, "features_per_patch": 64, "mixer_channels": 16, "mixer_features": 256, "inner_mlp_count": 1, "inner_mlp_width": 256, "patch_center_x": 0.14759132338024478, "patch_center_y": -0.009039915551949072, "patch_lensing_exp_x": 1.6640504814631036, "patch_lensing_exp_y": 1.3571390903600555}
    """

    parameters = json.loads(parameters_json)
    print(parameters)
    ann_model = get_mlp_mixer_model(parameters)

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