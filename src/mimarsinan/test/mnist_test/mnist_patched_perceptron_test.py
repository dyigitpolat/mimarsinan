from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.basic_classification_pipeline import BasicClassificationPipeline
from mimarsinan.pipelining.deployment_pipeline import DeploymentPipeline

from mimarsinan.data_handling.data_providers.mnist_data_provider import MNIST_DataProvider

import torch

def test_mnist_patched_perceptron():
    mnist_data_provider = MNIST_DataProvider()

    reporter = WandB_Reporter("mnist_pipeline_test", "experiment")

    pipeline = DeploymentPipeline(
        mnist_data_provider,
        deployment_parameters = {
            "lr": 0.001,
            "pt_epochs": 10,
            "aq_epochs": 10,
            "wq_epochs": 10,
            "nas_cycles": 2,
            "nas_batch_size": 2
        },
        platform_constraints = {
            "max_axons": 256,
            "max_neurons": 256,
            "target_tq": 32,
            "simulation_steps": 32,
            "weight_bits": 4
        },
        reporter = reporter,
        working_directory = "../generated/mnist3/"
    )

    pipeline.run()
    # pipeline.load_cache()
    # pipeline.run_from("Pretraining")