from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.basic_classification_pipeline import BasicClassificationPipeline
from mimarsinan.data_handling.data_providers.mnist_data_provider import MNIST_DataProvider

import torch

def test_mnist_patched_perceptron():
    mnist_data_provider = MNIST_DataProvider()

    reporter = WandB_Reporter("mnist_pipeline_test", "experiment")

    pipeline = BasicClassificationPipeline(
        mnist_data_provider,
        10,
        {
            "max_axons": 256, 
            "max_neurons": 256, 
            "target_tq": 128, 
            "simulation_steps": 128
        },
        {
            "lr": 0.001, 
            "pretraining_epochs": 5, 
            "aq_cycle_epochs": 10, 
            "wq_cycle_epochs": 10,
            "aq_cycles": 10,
            "wq_cycles": 10
        },
        reporter,
        "../generated/mnist2/"
    )

    pipeline.run()