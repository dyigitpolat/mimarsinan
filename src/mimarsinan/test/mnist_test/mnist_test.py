from mimarsinan.models.simple_mlp import *
from mimarsinan.models.polat_mlp_mixer import *
from mimarsinan.test.mnist_test.mnist_test_utils import *

from mimarsinan.mapping.chip_latency import *
from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.mapping.polat_mlp_mixer_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

import torch
import time
import math

def test_mnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_input_size = 28*28
    mnist_input_shape = (1,28*28)
    mnist_output_size = 10
    inner_mlp_width = 255
    inner_mlp_count = 3
    pretrain_epochs = 12
    cq_no_clamp_epochs = 3
    cq_only_epochs = 6
    cq_quantize_epochs = 12

    Tq = 30

    generated_files_path = "../generated/mnist/"
    input_count = 10000

    neurons_per_core = 256
    axons_per_core = 785

    _, test_loader = get_mnist_data(1)

    ann_model = SimpleMLP(
        inner_mlp_width, 
        inner_mlp_count, 
        mnist_input_size, 
        mnist_output_size,
        bias=True)

    print("Pretraining model...")
    train_on_mnist(ann_model, device, pretrain_epochs)

    print("Testing pretrained model...")
    correct, total = test_on_mnist(ann_model, device)
    print("  Correct:", correct, "Total:", total)

    print("Mapping model to soft cores...")
    model_repr = get_simple_mlp_repr(mnist_input_shape, ann_model)
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)
    print("  Number of soft cores:", len(soft_core_mapping.cores))
    print("  Soft core mapping delay: ", ChipLatency(soft_core_mapping).calculate())

    print("Testing CoreFlow with soft core mapping...")
    cf = CoreFlow(mnist_input_shape, soft_core_mapping)
    cf.set_activation(nn.LeakyReLU())
    correct, total = test_on_mnist(cf, device)
    print("  Correct:", correct, "Total:", total)

    calculate_core_thresholds(soft_core_mapping.cores, 4)

    print("Mapping soft cores to hard cores...")
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)
    print("  Number of hard cores:", len(hard_core_mapping.cores))
    print("  Hard core mapping delay: ", ChipLatency(hard_core_mapping).calculate())

    print("Testing CoreFlow with hard core mapping...")
    cf = CoreFlow(mnist_input_shape, hard_core_mapping)
    cf.set_activation(nn.LeakyReLU())
    correct, total = test_on_mnist(cf, device)
    print("  Correct:", correct, "Total:", total)

    print("Tuning model with CQ no clamp...")
    lr = 0.001
    cf.set_activation(CQ_Activation_NoClamp(Tq))
    train_on_mnist(cf, device, cq_no_clamp_epochs, lr = lr)

    # print("Tuning model with CQ leaky clamp...")
    # cycles = 10
    # for i in range(cycles):
    #     clamp_leak = math.sin((i * (1.0/cycles)))**0.5
    #     print("  Cycle:", i+1, "/", cycles)
    #     print("  Clamp leak:", clamp_leak)
    #     cf.set_activation(CQ_Activation_LeakyClamp(Tq, clamp_leak))
    #     train_on_mnist(cf, device, cq_only_epochs, lr = lr)
    #     lr = lr * 0.9

    print("Tuning model with CQ...")
    cf.set_activation(CQ_Activation(Tq))
    train_on_mnist(cf, device, cq_only_epochs, lr=lr)

    print("Tuning model with CQ and weight quantization...")
    cf.set_activation(CQ_Activation(Tq))
    train_on_mnist_quantized(cf, device, cq_quantize_epochs, lr=0.1*lr)

    # save model state dict to file:
    torch.save(cf.state_dict(), "../generated/mnist/quantized_coreflow_state_dict.pt")

    print("Updating model weights...")
    cf.update_cores()
    hard_core_mapping.cores = cf.cores

    print("Quantizing model weights...")
    quantize_cores(hard_core_mapping.cores, bits=4)

    ###### 

    print("Calculating delay for hard core mapping...")
    print(f"delay: {ChipLatency(hard_core_mapping).calculate()}")

    print("Mapping hard cores to chip...")
    chip = hard_cores_to_chip(
        mnist_input_size,
        hard_core_mapping, 
        axons_per_core,
        neurons_per_core, 
        leak=0,
        weight_type=int)

    print("Saving input data to files...")
    save_inputs_to_files(generated_files_path, test_loader, input_count)

    print("Saving trained weights and chip generation code...")
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    simulation_length = Tq + ChipLatency(hard_core_mapping).calculate()
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