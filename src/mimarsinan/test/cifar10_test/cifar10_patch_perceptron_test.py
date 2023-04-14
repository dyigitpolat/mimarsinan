from mimarsinan.models.perceptron_flow import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.cifar10_test.cifar10_test_utils import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *
from mimarsinan.transformations.chip_quantization import *
from mimarsinan.visualization.hardcore_visualization import *
from mimarsinan.chip_simulation.nevresim_driver import *

cifar10_patched_perceptron_test_clipping_rate = 0.01

def clip_model_weights_and_decay(model):
    clipper = SoftTensorClipping(cifar10_patched_perceptron_test_clipping_rate)
    for param in model.parameters():
        param.data = clipper.get_clipped_weights(param.data)
        param.data = torch.sin(torch.arctan(param.data))

def quantize_model(model):
    quantizer = TensorQuantization(bits = 4)
    for param in model.parameters():
        param.data = quantizer.quantize(param.data)

def clip_and_quantize_model(model):
    clip_model_weights_and_decay(model)
    quantize_model(model)

def generate_clip_quantize(bits):
    def clip_quantize(model):
        clipper = SoftTensorClipping(cifar10_patched_perceptron_test_clipping_rate)
        quantizer = TensorQuantization(bits = bits)
        for param in model.parameters():
            param.data = clipper.get_clipped_weights(param.data)
            param.data = torch.sin(torch.arctan(param.data))
            param.data = quantizer.quantize(param.data)
    return clip_quantize

def test_cifar10_patched_perceptron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cifar10_input_shape = (3, 32, 32)
    cifar10_output_size = 10
    Tq = 4
    simulation_length = 64
    batch_size = 1000

    pretrain_epochs = 15
    max_epochs = 10 * pretrain_epochs
    Tq_start = 100 

    perceptron_flow = PatchedPerceptronFlow(
        cifar10_input_shape, cifar10_output_size,
        192, 256, 8, 8, fc_depth=3)

    print("Pretraining model...")
    lr = 0.001
    perceptron_flow.set_activation(ClampedShiftReLU(0.5/Tq))
    lr, prev_acc = train_until_target_accuracy_with_weight_transformation(
        model=perceptron_flow,
        device=device,
        train_dataloader=get_cifar10_data(batch_size)[0],
        test_dataloader=get_cifar10_data(50000)[1],
        weight_transformation=clip_model_weights_and_decay,
        max_epochs=pretrain_epochs,
        lr=lr,
        target_accuracy=1.0)

    print("Fusing normalization...")
    perceptron_flow.fuse_normalization()

    print("Tuning model with CQ and weight quantization...")
    perceptron_flow.set_activation(CQ_Activation(Tq))
    train_until_target_accuracy_with_weight_transformation(
        model=perceptron_flow, 
        device=device,
        train_dataloader=get_cifar10_data(batch_size)[0],
        test_dataloader=get_cifar10_data(50000)[1],
        weight_transformation=clip_and_quantize_model,
        max_epochs=10,
        lr=lr,
        target_accuracy=prev_acc)

    print("Soft core mapping...")
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(perceptron_flow.get_mapper_repr())

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

    print("Quantizing hard core mapping...")
    scale = ChipQuantization(bits = 4).quantize(hard_core_mapping.cores)
    print("  Scale:", scale)

    print("Visualizing final mapping...")
    HardCoreMappingVisualizer(hard_core_mapping).visualize("../generated/cifar10/final_hard_core_mapping.png")
    
    print("Final hard core mapping latency: ", ChipLatency(hard_core_mapping).calculate())

    ######
    cifar10_input_size = 3*32*32
    generated_files_path = "../generated/cifar10/"
    _, test_loader = get_cifar10_data(1)

    print("Calculating delay for hard core mapping...")
    delay = ChipLatency(hard_core_mapping).calculate()
    print(f"  delay: {delay}")

    simulation_driver = NevresimDriver(
        cifar10_input_size,
        hard_core_mapping,
        generated_files_path,
        int
    )

    simulation_steps = delay + int(scale * simulation_length)
    predictions = simulation_driver.predict_spiking(
        test_loader,
        simulation_steps)

    print("Evaluating simulator output...")
    _, test_loader = get_cifar10_data(1)
    accuracy = evaluate_chip_output(predictions, test_loader, verbose=True)
    print("SNN accuracy on cifar10 is:", accuracy*100, "%")

    print("cifar10 perceptron test done.")