from mimarsinan.test.cifar10_test.cifar10_test_utils import *
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.basic_classification_pipeline import BasicClassificationPipeline

from mimarsinan.data_handling.data_providers.cifar10_data_provider import CIFAR10_DataProvider
import torch

def test_cifar10_patched_perceptron():
    data_provider = CIFAR10_DataProvider()

    reporter = WandB_Reporter("cifar10_pipeline_test", "experiment")

    pipeline = BasicClassificationPipeline(
        data_provider,
        10,
        {
            "max_axons": 256, 
            "max_neurons": 256, 
            "target_tq": 64, 
            "simulation_steps": 64,
            "weight_bits": 4
        },
        {
            "lr": 0.001, 
            "pretraining_epochs": 300, 
            "aq_cycle_epochs": 15, 
            "wq_cycle_epochs": 15,
            "aq_cycles": 15,
            "wq_cycles": 15
        },
        reporter,
        "../generated/cifar10_2/",
        model_complexity=2
    )

    pipeline.run()