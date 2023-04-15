from mimarsinan.models.perceptron_flow import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *
from mimarsinan.transformations.chip_quantization import *
from mimarsinan.visualization.hardcore_visualization import *


from mimarsinan.model_training.weight_transform_trainer import *
from mimarsinan.chip_simulation.nevresim_driver import *

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

def clip_param(param_data):
    clipper = SoftTensorClipping(mnist_patched_perceptron_test_clipping_rate)

    out = clipper.get_clipped_weights(param_data)
    out = special_decay(out)
    return out

def clip_and_quantize_param(param_data):
    clipper = SoftTensorClipping(mnist_patched_perceptron_test_clipping_rate)
    quantizer = TensorQuantization(bits=4)

    out = clipper.get_clipped_weights(param_data)
    out = special_decay(out)
    out = quantizer.quantize(out)
    return out

def test_mnist_patched_perceptron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mnist_input_shape = (1, 28, 28)
    mnist_output_size = 10
    Tq = 4
    simulation_length = 32
    batch_size = 2000

    pretrain_epochs = 5
    max_epochs = pretrain_epochs
    Tq_start = 64

    perceptron_flow = PatchedPerceptronFlow(
        mnist_input_shape, mnist_output_size,
        240, 240, 7, 7, fc_depth=3)
    
    train_loader = get_mnist_data(batch_size)[0]
    test_loader = get_mnist_data(50000)[1]

    trainer = WeightTransformTrainer(
        perceptron_flow, device, train_loader, test_loader, clip_param)
    
    print("Pretraining model...")
    lr = 0.001
    perceptron_flow.set_activation(ClampedShiftReLU(0.5/Tq))
    prev_acc = trainer.train_n_epochs(lr, ppf_loss, pretrain_epochs)

    print("Fusing normalization...")
    perceptron_flow.fuse_normalization()

    print("Tuning model with CQ and weight quantization...")
    perceptron_flow.set_activation(CQ_Activation(Tq))
    trainer.weight_transformation = clip_and_quantize_param
    trainer.train_until_target_accuracy(lr, ppf_loss, max_epochs, prev_acc)

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
    core_flow = CoreFlow(mnist_input_shape, hard_core_mapping)
    core_flow.set_activation(CQ_Activation(Tq))

    print("Testing with core flow...")
    correct, total = test_on_mnist(core_flow, device)
    print("  Correct:", correct, "Total:", total)

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

    print("Evaluating simulator output...")
    _, test_loader = get_mnist_data(1)
    accuracy = evaluate_chip_output(predictions, test_loader, verbose=True)
    print("SNN accuracy on MNIST is:", accuracy*100, "%")

    print("MNIST perceptron test done.")