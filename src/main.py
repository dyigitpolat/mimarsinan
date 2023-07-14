from init import *

from mimarsinan.common.wandb_utils import WandB_Reporter
from mimarsinan.pipelining.deployment_pipeline import DeploymentPipeline

import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <deployment_config_json>")
        exit(1)
    
    deployment_config_path = sys.argv[1]
    with open(deployment_config_path, 'r') as f:
        deployment_config = json.load(f)
    
    data_provider_path = deployment_config['data_provider_path']
    data_provider_name = deployment_config['data_provider_name']

    DataProvider = import_class_from_path(
        data_provider_path, data_provider_name)
    
    data_provider = DataProvider()
    deployment_name = deployment_config['experiment_name']
    platform_constraints = deployment_config['platform_constraints']
    deployment_parameters = deployment_config['deployment_parameters']
    working_directory = deployment_config['generated_files_path']
    start_step = deployment_config['start_step']

    run_pipeline(
        data_provider=data_provider,
        deployment_name=deployment_name,
        platform_constraints=platform_constraints,
        deployment_parameters=deployment_parameters,
        working_directory=working_directory,
        start_step=start_step)

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