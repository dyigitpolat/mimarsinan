from mimarsinan.test.cifar10_test.cifar10_test_utils import *
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.basic_classification_pipeline import BasicClassificationPipeline

import torch

def test_cifar10_patched_perceptron():
    training_dataloader = get_cifar10_data(2000)[0]
    validation_dataloader = get_cifar10_data(1000)[2]
    test_dataloader = get_cifar10_data(10000)[1]

    reporter = WandB_Reporter("cifar10_pipeline_test", "experiment")

    pipeline = BasicClassificationPipeline(
        training_dataloader,
        validation_dataloader,
        test_dataloader,
        10,
        {
            "max_axons": 256, 
            "max_neurons": 256, 
            "target_tq": 16, 
            "simulation_steps": 64
        },
        {
            "lr": 0.001, 
            "pretraining_epochs": 20, 
            "aq_cycle_epochs": 10, 
            "wq_cycle_epochs": 10,
            "aq_cycles": 10,
            "wq_cycles": 10
        },
        reporter,
        "../generated/cifar10_2/"
    )

    pipeline.run()