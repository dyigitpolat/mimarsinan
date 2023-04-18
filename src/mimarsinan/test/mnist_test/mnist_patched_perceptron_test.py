from mimarsinan.models.perceptron_flow import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *
from mimarsinan.transformations.chip_quantization import *
from mimarsinan.visualization.hardcore_visualization import *

from mimarsinan.common.wandb_utils import *
from mimarsinan.model_training.weight_transform_trainer import *
from mimarsinan.chip_simulation.nevresim_driver import *
from mimarsinan.tuning.basic_smooth_adaptation import *


import random

mnist_patched_perceptron_test_clipping_rate = 0.01

def special_decay(w):
    return torch.clamp(w, -1, 1)

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

def decay_param(param_data):
    out = special_decay(param_data)
    return out

def decay_and_quantize_param(param_data):
    quantizer = TensorQuantization(bits=4)

    out = special_decay(param_data)
    out = quantizer.quantize(out)
    return out

def clip_decay_and_quantize_param(param_data):
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

    pretrain_epochs = 10
    max_epochs = pretrain_epochs

    perceptron_flow = PatchedPerceptronFlow(
        mnist_input_shape, mnist_output_size,
        240, 240, 7, 7, fc_depth=3)
    
    train_loader = get_mnist_data(batch_size)[0]
    validation_loader = get_mnist_data(1000)[2]

    reporter = WandB_Reporter("mnist_patched_perceptron_test", "experiment")

    trainer = WeightTransformTrainer(
        perceptron_flow, device, train_loader, validation_loader, decay_param)
    trainer.report_function = reporter.report
    
    print("Pretraining model...")
    lr = 0.001
    perceptron_flow.set_activation(ClampedReLU())
    prev_acc = trainer.train_n_epochs(lr, ppf_loss, pretrain_epochs)
    print(trainer.validate())

    print("Fusing normalization...")
    perceptron_flow.fuse_normalization()

    print("Sanity check...")
    trainer.weight_transformation = lambda x: x
    trainer.train_n_epochs(lr=0.0, loss_function=ppf_loss, epochs=1)

    def alpha_and_Tq_adaptation(alpha, Tq):
        print("  Tuning model with soft CQ with alpha = {} and Tq = {}...".format(alpha, Tq))
        reporter.report("alpha", alpha)
        reporter.report("Tq", Tq)
        perceptron_flow.set_activation(CQ_Activation_Soft(Tq, alpha))
        trainer.weight_transformation = decay_and_quantize_param
        trainer.train_until_target_accuracy(lr, ppf_loss, 10, prev_acc)
    
    alpha_interpolator = BasicInterpolation(0, 15, curve = lambda x: x ** 2)
    Tq_interpolator = BasicInterpolation(100, 4, curve = lambda x: x ** 0.5)
    BasicSmoothAdaptation(alpha_and_Tq_adaptation).adapt_smoothly([
        alpha_interpolator, Tq_interpolator], 10)

    print("Tuning model with CQ and weight quantization...")
    perceptron_flow.set_activation(CQ_Activation(Tq))
    trainer.weight_transformation = clip_decay_and_quantize_param
    trainer.train_until_target_accuracy(lr, ppf_loss, max_epochs, prev_acc)

    ######
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
    correct, total = trainer.validate(core_flow, device)
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
    test_loader = get_mnist_data(1)[1]

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
    accuracy = evaluate_chip_output(predictions, test_loader, verbose=True)
    print("SNN accuracy on MNIST is:", accuracy*100, "%")

    print("MNIST perceptron test done.")