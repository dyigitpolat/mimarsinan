from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.pipeline import Pipeline

import torch

def test_mnist_patched_perceptron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    training_dataloader = get_mnist_data(2000)[0]
    validation_dataloader = get_mnist_data(1000)[2]
    test_dataloader = get_mnist_data(10000)[1]

    reporter = WandB_Reporter("mnist_pipeline_test", "experiment")

    pipeline = Pipeline(
        training_dataloader,
        validation_dataloader,
        test_dataloader,
        10,
        {"max_axons": 256, "max_neurons": 256, "target_tq": 2},
        reporter,
        "../generated/mnist2/"
    )

    pipeline.run()