from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.pipeline import BasicClassificationPipeline

import torch

def test_mnist_patched_perceptron():
    training_dataloader = get_mnist_data(2000)[0]
    validation_dataloader = get_mnist_data(1000)[2]
    test_dataloader = get_mnist_data(10000)[1]

    reporter = WandB_Reporter("mnist_pipeline_test", "experiment")

    pipeline = BasicClassificationPipeline(
        training_dataloader,
        validation_dataloader,
        test_dataloader,
        10,
        {"max_axons": 256, "max_neurons": 256, "target_tq": 4, "simulation_steps": 16},
        reporter,
        "../generated/mnist2/"
    )

    pipeline.run()