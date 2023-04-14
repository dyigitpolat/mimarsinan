from mimarsinan.models.perceptron_flow import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *
from mimarsinan.transformations.chip_quantization import *
from mimarsinan.visualization.hardcore_visualization import *

from mimarsinan.chip_simulation.nevresim_driver import *


import time
import math

mnist_patched_perceptron_test_clipping_rate = 0.01

def special_decay(w):
    return torch.sin(torch.arctan(w))

def clip_model_weights_and_decay(model):
    clipper = SoftTensorClipping(mnist_patched_perceptron_test_clipping_rate)
    for param in model.parameters():
        param.data = clipper.get_clipped_weights(param.data)
        param.data = special_decay(param.data)

def quantize_model(model):
    quantizer = TensorQuantization(bits=4)
    for param in model.parameters():
        param.data = quantizer.quantize(param.data)

def clip_and_quantize_model(model):
    clip_model_weights_and_decay(model)
    quantize_model(model)

def test_mnist_patched_perceptron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_input_shape = (1, 28, 28)
    mnist_output_size = 10
    Tq = 4
    simulation_length = 32
    batch_size = 2000

    pretrain_epochs = 4
    max_epochs = 10 * pretrain_epochs
    Tq_start = 64

    perceptron_flow = PatchedPerceptronFlow(
        mnist_input_shape, mnist_output_size,
        240, 240, 7, 7, fc_depth=3)

    print("Pretraining model...")
    lr = 0.001
    perceptron_flow.set_activation(ClampedShiftReLU(0.5/Tq))
    lr, prev_acc = train_until_target_accuracy_with_weight_transformation(
        model=perceptron_flow,
        device=device,
        train_dataloader=get_mnist_data(batch_size)[0],
        test_dataloader=get_mnist_data(50000)[1],
        weight_transformation=clip_model_weights_and_decay,
        max_epochs=pretrain_epochs,
        lr=lr,
        target_accuracy=1.0)

    print("Fusing normalization...")
    perceptron_flow.fuse_normalization()

    # lr *= 0.5
    # perceptron_flow.set_activation(ClampedReLU())
    # train_until_target_accuracy_with_weight_transformation(
    #     model=perceptron_flow,
    #     device=device,
    #     train_dataloader=get_mnist_data(batch_size)[0],
    #     test_dataloader=get_mnist_data(50000)[1],
    #     weight_transformation=clip_model_weights_and_decay,
    #     max_epochs=max_epochs,
    #     lr=lr,
    #     target_accuracy=prev_acc)
    
    # print("Tuning model with soft CQ...")
    # currentTq = Tq_start
    # while currentTq >= Tq:
    #     for i in range(0, 20 + 1):
    #         alpha = 0.5 * i + 0.1
    #         # shift = \
    #         #     (1.0+math.atan(alpha/(4*math.pi))) \
    #         #     / (2 * Tq)
    #         # print(f"  Tuning with shift = {shift*Tq}/Tq...")
    #         # perceptron_flow.set_activation(ClampedShiftReLU(shift))
    #         # train_until_target_accuracy_with_weight_transformation(
    #         #     model=perceptron_flow, 
    #         #     device=device,
    #         #     train_dataloader=get_mnist_data(batch_size)[0],
    #         #     test_dataloader=get_mnist_data(50000)[1],
    #         #     weight_transformation=clip_model_weights_and_decay,
    #         #     max_epochs=max_epochs,
    #         #     lr=lr,
    #         #     target_accuracy=prev_acc)
    #         print(f"  Tuning with alpha = {alpha}...")
    #         perceptron_flow.set_activation(CQ_Activation_Soft(currentTq, alpha))
    #         train_until_target_accuracy_with_weight_transformation(
    #             model=perceptron_flow, 
    #             device=device,
    #             train_dataloader=get_mnist_data(batch_size)[0],
    #             test_dataloader=get_mnist_data(50000)[1],
    #             weight_transformation=clip_model_weights_and_decay,
    #             max_epochs=max_epochs,
    #             lr=lr,
    #             target_accuracy=prev_acc)
    #     currentTq = currentTq // 2

    # print("Tuning model with soft CQ and weight quantization...")
    # perceptron_flow.set_activation(CQ_Activation_Soft(Tq, alpha))
    # train_until_target_accuracy_with_weight_transformation(
    #     model=perceptron_flow, 
    #     device=device,
    #     train_dataloader=get_mnist_data(batch_size)[0],
    #     test_dataloader=get_mnist_data(50000)[1],
    #     weight_transformation=clip_and_quantize_model,
    #     max_epochs=max_epochs,
    #     lr=lr,
    #     target_accuracy=prev_acc)
    
    # print("Tuning model with CQ and weight quantization...")
    # perceptron_flow.set_activation(CQ_Activation(Tq))
    # train_until_target_accuracy_with_weight_transformation(
    #     model=perceptron_flow, 
    #     device=device,
    #     train_dataloader=get_mnist_data(batch_size)[0],
    #     test_dataloader=get_mnist_data(50000)[1],
    #     weight_transformation=clip_and_quantize_model,
    #     max_epochs=max_epochs,
    #     lr=lr,
    #     target_accuracy=prev_acc)

    print("Tuning model with CQ and weight quantization...")
    perceptron_flow.set_activation(CQ_Activation(Tq))
    train_until_target_accuracy_with_weight_transformation(
        model=perceptron_flow, 
        device=device,
        train_dataloader=get_mnist_data(batch_size)[0],
        test_dataloader=get_mnist_data(50000)[1],
        weight_transformation=clip_and_quantize_model,
        max_epochs=10,
        lr=lr,
        target_accuracy=prev_acc)

    print("Soft core mapping...")
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(perceptron_flow.get_mapper_repr())

    print("Core flow mapping...")
    core_flow = CoreFlow(mnist_input_shape, soft_core_mapping)
    core_flow.set_activation(CQ_Activation(Tq))

    print("Testing with core flow...")
    correct, total = test_on_mnist(core_flow, device)
    print("  Correct:", correct, "Total:", total)

    print("Calculating soft core thresholds for hardcore mapping...")
    ChipQuantization(bits = 4).calculate_core_thresholds(soft_core_mapping.cores)

    print("Hard core mapping...")
    axons_per_core = 256
    neurons_per_core = 256
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)

    print("Core flow mapping...")
    core_flow = CoreFlow(mnist_input_shape, hard_core_mapping)
    core_flow.set_activation(CQ_Activation(Tq))

    print("Testing with core flow...")
    correct, total = test_on_mnist(core_flow, device)
    print("  Correct:", correct, "Total:", total)

    # print("Tuning hard core mapping...")
    # lr *= 0.5
    # core_flow.set_activation(CQ_Activation(Tq))
    # lr, _ = train_until_target_accuracy_with_weight_transformation(
    #     model=core_flow,
    #     device=device,
    #     train_dataloader=get_mnist_data(batch_size)[0],
    #     test_dataloader=get_mnist_data(50000)[1],
    #     weight_transformation=clip_and_quantize_model,
    #     max_epochs=max_epochs,
    #     lr_max=lr_max,
    #     lr=lr,
    #     target_accuracy=prev_acc)

    # print("Updating hard core parameters...")
    # core_flow.update_cores()
    # hard_core_mapping.cores = core_flow.cores

    print("Quantizing hard core mapping...")
    scale = ChipQuantization(bits = 4).quantize(hard_core_mapping.cores)
    print("  Scale:", scale)

    print("Visualizing final mapping...")
    HardCoreMappingVisualizer(hard_core_mapping).visualize("../generated/mnist/final_hard_core_mapping.png")

    print("Final hard core mapping latency: ", ChipLatency(hard_core_mapping).calculate())

    ######
    mnist_input_size = 28*28
    generated_files_path = "../generated/mnist/"
    _, test_loader = get_mnist_data(1)

    print("Calculating delay for hard core mapping...")
    delay = ChipLatency(hard_core_mapping).calculate()
    print(f"  delay: {delay}")

    simulation_driver = NevresimDriver(
        mnist_input_size,
        hard_core_mapping,
        generated_files_path,
        int
    )

    simulation_steps = delay + int(scale * simulation_length)
    predictions = simulation_driver.predict_spiking(
        test_loader,
        simulation_steps)
    



    # print("Mapping hard cores to chip...")
    # chip = hard_cores_to_chip(
    #     mnist_input_size,
    #     hard_core_mapping,
    #     axons_per_core,
    #     neurons_per_core,
    #     leak=0,
    #     weight_type=int)

    # print("Saving input data to files...")
    # save_inputs_to_files(generated_files_path, test_loader, input_count)

    # print("Saving trained weights and chip generation code...")
    # save_weights_and_chip_code(chip, generated_files_path)

    # print("Generating main function code...")
    # simulation_length = int(scale * simulation_length + ChipLatency(hard_core_mapping).calculate())
    # generate_main_function(
    #     generated_files_path, input_count, mnist_output_size, simulation_length,
    #     main_cpp_template, get_config("Stochastic", "Novena", "int"))

    # print("Compiling nevresim for mapped chip...")
    # simulator_filename = \
    #     compile_simulator(generated_files_path, "../nevresim/")
    # print("Compilation outcome:", simulator_filename)

    # print("Executing simulator...")
    # start_time = time.time()
    # chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    # end_time = time.time()
    # print("Simulation time:", end_time - start_time)

    print("Evaluating simulator output...")
    _, test_loader = get_mnist_data(1)
    accuracy = evaluate_chip_output(predictions, test_loader, verbose=True)
    print("SNN accuracy on MNIST is:", accuracy*100, "%")

    print("MNIST perceptron test done.")