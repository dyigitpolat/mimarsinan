from init import *

from mimarsinan.common.wandb_utils import WandB_Reporter
from mimarsinan.pipelining.pipelines.nas_deployment_pipeline import NASDeploymentPipeline
from mimarsinan.pipelining.pipelines.vanilla_deployment_pipeline import VanillaDeploymentPipeline
from mimarsinan.pipelining.pipelines.cq_pipeline import CQPipeline
from mimarsinan.data_handling.data_provider_factory import ImportedDataProviderFactory

import sys
import json
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <deployment_config_json>")
        exit(1)
    
    deployment_config_path = sys.argv[1]
    with open(deployment_config_path, 'r') as f:
        deployment_config = json.load(f)
    
    data_provider_path = deployment_config['data_provider_path']
    data_provider_name = deployment_config['data_provider_name']
    data_provider_factory = ImportedDataProviderFactory(data_provider_path, data_provider_name, "./datasets")

    deployment_name = deployment_config['experiment_name']
    platform_constraints = deployment_config['platform_constraints']
    deployment_parameters = deployment_config['deployment_parameters']
    start_step = deployment_config['start_step']
    target_metric_override = deployment_config.get('target_metric_override')

    if 'pipeline_mode' in deployment_config:
        pipeline_mode = deployment_config['pipeline_mode']
    else:
        pipeline_mode = "phased"
    
    working_directory = \
        deployment_config['generated_files_path'] + "/" + deployment_name + "_" + pipeline_mode + "_deployment_run"
    
    # save deployment config to working directory /_RUN_CONFIG/config.json
    os.makedirs(working_directory + "/_RUN_CONFIG", exist_ok=True)
    with open(working_directory + "/_RUN_CONFIG/config.json", 'w') as f:
        json.dump(deployment_config, f, indent=4)


    run_pipeline(
        pipeline_mode=pipeline_mode,
        data_provider_factory=data_provider_factory,
        deployment_name=deployment_name,
        platform_constraints=platform_constraints,
        deployment_parameters=deployment_parameters,
        working_directory=working_directory,
        start_step=start_step,
        target_metric_override=target_metric_override)

def run_pipeline(
    pipeline_mode,
    data_provider_factory, 
    deployment_name, 
    platform_constraints, 
    deployment_parameters, 
    working_directory,
    start_step = None,
    target_metric_override = None):

    deployment_mode_map = {
        "phased": NASDeploymentPipeline,
        "vanilla": VanillaDeploymentPipeline,
        "cq": CQPipeline
    }

    reporter = WandB_Reporter(deployment_name, "deployment")

    pipeline = deployment_mode_map[pipeline_mode] (
        data_provider_factory=data_provider_factory,
        deployment_parameters=deployment_parameters,
        platform_constraints=platform_constraints,
        reporter=reporter,
        working_directory=working_directory
    )

    if target_metric_override is not None:
        pipeline.set_target_metric(target_metric_override)
    
    if start_step is None:
        pipeline.run()
    else:
        pipeline.run_from(step_name=start_step)

if __name__ == "__main__":
    init()
    main()