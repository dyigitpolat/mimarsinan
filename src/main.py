from init import *

from mimarsinan.common.wandb_utils import WandB_Reporter
from mimarsinan.pipelining.pipelines.deployment_pipeline import (
    DeploymentPipeline,
)
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
import mimarsinan.data_handling.data_providers

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
    
    data_provider_name = deployment_config['data_provider_name']
    seed = deployment_config.get("seed", 0)
    data_provider_factory = BasicDataProviderFactory(data_provider_name, "./datasets", seed=seed)

    deployment_name = deployment_config['experiment_name']
    deployment_parameters = deployment_config['deployment_parameters']

    # Backward compatible platform constraints protocol:
    # - Legacy: platform_constraints is the flat dict used by pipelines
    # - New: {"mode": "user"|"auto", "user": {...}, "auto": {"fixed": {...}, "search_space": {...}}}
    platform_constraints_raw = deployment_config["platform_constraints"]
    if isinstance(platform_constraints_raw, dict) and "mode" in platform_constraints_raw:
        mode = platform_constraints_raw.get("mode", "user")
        if mode == "user":
            # Prefer explicit user block; otherwise, treat remaining keys as the user dict.
            platform_constraints = platform_constraints_raw.get(
                "user",
                {k: v for k, v in platform_constraints_raw.items() if k != "mode"},
            )
        elif mode == "auto":
            auto = platform_constraints_raw.get("auto", {}) or {}
            fixed = auto.get("fixed", {}) or {}
            search_space = auto.get("search_space", {}) or {}

            # Merge hardware search-space hints into deployment_parameters.arch_search (if not already provided).
            arch_cfg = deployment_parameters.setdefault("arch_search", {})
            for k, v in search_space.items():
                arch_cfg.setdefault(k, v)

            platform_constraints = fixed
        else:
            raise ValueError(f"Invalid platform_constraints.mode: {mode}")
    else:
        platform_constraints = platform_constraints_raw
    start_step = deployment_config['start_step']
    stop_step = deployment_config.get("stop_step")
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
        stop_step=stop_step,
        target_metric_override=target_metric_override)

def run_pipeline(
    pipeline_mode,
    data_provider_factory, 
    deployment_name, 
    platform_constraints, 
    deployment_parameters, 
    working_directory,
    start_step = None,
    stop_step = None,
    target_metric_override = None):

    # Merge pipeline_mode preset into a copy of deployment_parameters
    # so that explicit user values always win over preset defaults.
    merged_params = dict(deployment_parameters)
    DeploymentPipeline.apply_preset(pipeline_mode, merged_params)

    reporter = WandB_Reporter(deployment_name, "deployment")

    pipeline = DeploymentPipeline(
        data_provider_factory=data_provider_factory,
        deployment_parameters=merged_params,
        platform_constraints=platform_constraints,
        reporter=reporter,
        working_directory=working_directory,
    )

    if target_metric_override is not None:
        pipeline.set_target_metric(target_metric_override)
    
    if start_step is None:
        pipeline.run(stop_step=stop_step)
    else:
        pipeline.run_from(step_name=start_step, stop_step=stop_step)

if __name__ == "__main__":
    init()
    main()
