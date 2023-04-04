from mimarsinan.models.perceptron_flow import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.cifar10_test.cifar10_test_utils import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.transformations.chip_quantization import *
from mimarsinan.visualization.hardcore_visualization import *

import time

cifar10_patched_perceptron_test_clipping_rate = 0.01

def special_decay(w):
    return torch.sin(torch.arctan(w))

def clip_model_weights_and_decay(model, clipping_rate=cifar10_patched_perceptron_test_clipping_rate):
    clipper = SoftTensorClipping(clipping_rate)
    for param in model.parameters():
        param.data = clipper.get_clipped_weights(param.data)
        param.data = special_decay(param.data)

def quantize_model(model, bits=4):
    quantizer = TensorQuantization(bits)
    for param in model.parameters():
        param.data = quantizer.quantize(param.data)

def clip_and_quantize_model(model, bits=4, clipping_rate=cifar10_patched_perceptron_test_clipping_rate):
    clip_model_weights_and_decay(model, clipping_rate)
    quantize_model(model, bits)

def test_cifar10_patched_perceptron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cifar10_input_shape = (3, 32, 32)
    cifar10_output_size = 10
    Tq = 4
    steps = 30
    batch_size = 2000

    pretrain_epochs = 3
    fused_tuning_epochs = 3
    hardcore_tuning_epochs = 1

    perceptron_flow = PatchedPerceptronFlow(
        cifar10_input_shape, cifar10_output_size,
        192, 256, 8, 8, fc_depth=3)

    print("Pretraining model...")
    lr = 0.001
    train_with_weight_trasformation(
        model=perceptron_flow, 
        device=device,
        train_dataloader=get_cifar10_data(batch_size)[0],
        test_dataloader=get_cifar10_data(50000)[1],
        weight_transformation=clip_model_weights_and_decay,
        epochs=pretrain_epochs,
        lr=lr)
    
    print("Fusing normalization...")
    perceptron_flow.fuse_normalization()

    print("Tuning model with CQ and weight quantization after fuse...")
    perceptron_flow.set_activation(CQ_Activation(Tq))
    train_with_weight_trasformation(
        model=perceptron_flow, 
        device=device,
        train_dataloader=get_cifar10_data(batch_size*2)[0],
        test_dataloader=get_cifar10_data(50000)[1],
        weight_transformation=clip_and_quantize_model,
        epochs=fused_tuning_epochs,
        lr=lr)

    print("Soft core mapping...")
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(perceptron_flow.get_mapper_repr())

    print("Core flow mapping...")
    core_flow = CoreFlow(cifar10_input_shape, soft_core_mapping)
    core_flow.set_activation(CQ_Activation(Tq))

    print("Testing with core flow...")
    correct, total = test_on_cifar10(core_flow, device)
    print("  Correct:", correct, "Total:", total)

    print("Calculating soft core thresholds for hardcore mapping...")
    ChipQuantization(bits = 4).calculate_core_thresholds(soft_core_mapping.cores)

    print("Hard core mapping...")
    axons_per_core = 256
    neurons_per_core = 256
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)
    
    print("Core flow mapping...")
    core_flow = CoreFlow(cifar10_input_shape, hard_core_mapping)
    core_flow.set_activation(CQ_Activation(Tq))

    print("Testing with core flow...")
    correct, total = test_on_cifar10(core_flow, device)
    print("  Correct:", correct, "Total:", total)

    # print("Tuning hard core mapping...")
    # lr *= 0.5
    # train_with_weight_trasformation(
    #     model=core_flow, 
    #     device=device,
    #     train_dataloader=get_cifar10_data(batch_size*2)[0],
    #     test_dataloader=get_cifar10_data(50000)[1],
    #     weight_transformation=clip_and_quantize_model,
    #     epochs=hardcore_tuning_epochs,
    #     lr=lr)
    
    # print("Updating hard core parameters...")
    # core_flow.update_cores()
    # hard_core_mapping.cores = core_flow.cores
    
    print("Quantizing hard core mapping...")
    scale = ChipQuantization(bits = 4).quantize(hard_core_mapping.cores)
    print("  Scale:", scale)

    print("Visualizing final mapping...")
    HardCoreMappingVisualizer(hard_core_mapping).visualize("../generated/cifar10/final_hard_core_mapping.png")
    
    print("Final hard core mapping latency: ", ChipLatency(hard_core_mapping).calculate())

    ######
    cifar10_input_size = 3*32*32
    generated_files_path = "../generated/cifar10/"
    input_count = 10000
    _, test_loader = get_cifar10_data(1)

    print("Calculating delay for hard core mapping...")
    print(f"delay: {ChipLatency(hard_core_mapping).calculate()}")

    print("Mapping hard cores to chip...")
    chip = hard_cores_to_chip(
        cifar10_input_size,
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
    simulation_length = int(scale * steps + ChipLatency(hard_core_mapping).calculate())
    generate_main_function(
        generated_files_path, input_count, cifar10_output_size, simulation_length,
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
    _, test_loader = get_cifar10_data(1)
    accuracy = evaluate_chip_output(chip_output, test_loader, cifar10_output_size, verbose=True)
    print("SNN accuracy on cifar10 is:", accuracy*100, "%")

    print("cifar10 perceptron test done.")