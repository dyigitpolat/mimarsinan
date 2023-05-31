from mimarsinan.data_handling.data_providers.ecg_data_provider import ECG_DataProvider
from mimarsinan.common.wandb_utils import *
from mimarsinan.pipelining.basic_classification_pipeline import BasicClassificationPipeline

import torch

def test_ecg_patched_perceptron():
    data_provider = ECG_DataProvider()

    reporter = WandB_Reporter("ecg_pipeline_test", "experiment")

    pipeline = BasicClassificationPipeline(
        data_provider,
        2,
        {
            "max_axons": 256, 
            "max_neurons": 256, 
            "target_tq": 128, 
            "simulation_steps": 128
        },
        {
            "lr": 0.001, 
            "pretraining_epochs": 30, 
            "aq_cycle_epochs": 20, 
            "wq_cycle_epochs": 20,
            "aq_cycles": 15,
            "wq_cycles": 15
        },
        reporter,
        "../generated/ecg2/",
        model_complexity=10
    )

    pipeline.run()