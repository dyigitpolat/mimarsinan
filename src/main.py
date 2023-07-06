from init import *

from mimarsinan.data_handling.data_providers.mnist_data_provider import MNIST_DataProvider
from mimarsinan.data_handling.data_providers.cifar10_data_provider import CIFAR10_DataProvider
from mimarsinan.data_handling.data_providers.ecg_data_provider import ECG_DataProvider
from mimarsinan.data_handling.data_providers.ecg_2_data_provider import ECG_2_DataProvider

from mimarsinan.common.wandb_utils import WandB_Reporter

from mimarsinan.pipelining.deployment_pipeline import DeploymentPipeline

def main():
    platform_constraints = {
        "max_axons": 256,
        "max_neurons": 256,
        "target_tq": 32,
        "simulation_steps": 32,
        "weight_bits": 4
    }

    deployment_name, data_provider, deployment_parameters = select_deployment_configuration()

    working_directory = f"../generated/{deployment_name}/"

    run_pipeline(
        data_provider=data_provider,
        deployment_name=deployment_name,
        platform_constraints=platform_constraints,
        deployment_parameters=deployment_parameters,
        working_directory=working_directory,
        start_step=None)
    
def select_deployment_configuration():
    dataset_selection = input("Select dataset (mnist, cifar10, ecg): ")

    if dataset_selection[0] == 'mnist'[0]:
        deployment_name = "mnist_deployment"
        data_provider = MNIST_DataProvider()
        deployment_parameters = {
            "lr": 0.001,
            "pt_epochs": 10,
            "aq_epochs": 10,
            "wq_epochs": 10,
            "nas_cycles": 1,
            "nas_batch_size": 20
        }

    elif dataset_selection[0] == 'cifar10'[0]:
        deployment_name = "cifar10_deployment"
        data_provider = CIFAR10_DataProvider()
        deployment_parameters = {
            "lr": 0.001,
            "pt_epochs": 50,
            "aq_epochs": 10,
            "wq_epochs": 10,
            "nas_cycles": 1,
            "nas_batch_size": 20
        }

    elif dataset_selection[0] == 'ecg'[0]:
        deployment_name = "ecg_deployment"
        data_provider = ECG_2_DataProvider()
        deployment_parameters = {
            "lr": 0.001,
            "pt_epochs": 10,
            "aq_epochs": 10,
            "wq_epochs": 10,
            "nas_cycles": 1,
            "nas_batch_size": 20
        }

    return deployment_name, data_provider, deployment_parameters

def run_pipeline(
    data_provider, 
    deployment_name, 
    platform_constraints, 
    deployment_parameters, 
    working_directory,
    start_step = None):

    reporter = WandB_Reporter(deployment_name, "deployment")

    pipeline = DeploymentPipeline(
        data_provider=data_provider,
        deployment_parameters=deployment_parameters,
        platform_constraints=platform_constraints,
        reporter=reporter,
        working_directory=working_directory
    )
    
    if start_step is None:
        pipeline.run()
    else:
        pipeline.load_cache()
        pipeline.run_from(step_name=start_step)

if __name__ == "__main__":
    init()
    main()