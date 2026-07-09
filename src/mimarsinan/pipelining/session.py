"""Composition root: one place that turns a deployment config into a running pipeline."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import mimarsinan.data_handling.data_providers  # noqa: F401  # pyright: ignore[reportUnusedImport] — registers built-in providers
from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.reporter import DefaultReporter
from mimarsinan.config_schema.deployment_derivation import (
    enforce_quantization_assembly_contract,
)
from mimarsinan.data_handling.data_loader_factory import close_pipeline_loaders
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.gui.runtime.composite_reporter import CompositeReporter
from mimarsinan.pipelining.core.pipelines.deployment_pipeline import DeploymentPipeline


@dataclass(frozen=True)
class ParsedDeploymentConfig:
    """A deployment-config JSON resolved into pipeline constructor arguments."""

    pipeline_mode: str
    data_provider_factory: Any
    deployment_name: str
    platform_constraints: dict
    deployment_parameters: dict
    working_directory: str
    start_step: Optional[str]
    stop_step: Optional[str]
    target_metric_override: Optional[float]


def parse_deployment_config(
    deployment_config: dict, *, data_provider_factory: Any = None
) -> ParsedDeploymentConfig:
    """Parse a deployment-config dict and persist it under ``_RUN_CONFIG/``."""
    deployment_name = deployment_config["experiment_name"]
    deployment_parameters = dict(deployment_config["deployment_parameters"])

    if data_provider_factory is None:
        data_provider_factory = BasicDataProviderFactory(
            deployment_config["data_provider_name"],
            deployment_config.get("datasets_path", "./datasets"),
            seed=deployment_config.get("seed", 0),
            preprocessing=deployment_parameters.get("preprocessing"),
            batch_size=deployment_parameters.get("batch_size"),
        )

    platform_constraints = _resolve_platform_constraints(
        deployment_config, deployment_parameters
    )

    enforce_quantization_assembly_contract(
        deployment_parameters,
        platform_constraints if isinstance(platform_constraints, dict) else {},
        pipeline_mode=deployment_config.get(
            "pipeline_mode", deployment_parameters.get("pipeline_mode")
        ),
    )

    pipeline_mode = deployment_config.get("pipeline_mode", "phased")
    working_directory = deployment_config.get(
        "_working_directory",
        f"{deployment_config['generated_files_path']}/"
        f"{deployment_name}_{pipeline_mode}_deployment_run",
    )
    _persist_run_config(deployment_config, working_directory)

    return ParsedDeploymentConfig(
        pipeline_mode=pipeline_mode,
        data_provider_factory=data_provider_factory,
        deployment_name=deployment_name,
        platform_constraints=platform_constraints,
        deployment_parameters=deployment_parameters,
        working_directory=working_directory,
        start_step=deployment_config.get("start_step"),
        stop_step=deployment_config.get("stop_step"),
        target_metric_override=deployment_config.get("target_metric_override"),
    )


def _resolve_platform_constraints(
    deployment_config: dict, deployment_parameters: dict
) -> dict:
    """In hw search mode, merge search_space into arch_search without mutating the input."""
    raw = deployment_config.get("platform_constraints", {})
    if deployment_parameters.get("hw_config_mode", "fixed") != "search":
        return raw
    if not isinstance(raw, dict):
        return raw
    platform_constraints = copy.deepcopy(raw)
    search_space = platform_constraints.pop("search_space", {}) or {}
    arch_cfg = deployment_parameters.setdefault("arch_search", {})
    for key, value in search_space.items():
        arch_cfg.setdefault(key, value)
    return platform_constraints


def _persist_run_config(deployment_config: dict, working_directory: str) -> None:
    os.makedirs(f"{working_directory}/_RUN_CONFIG", exist_ok=True)
    saveable = {k: v for k, v in deployment_config.items() if not k.startswith("_")}
    with open(f"{working_directory}/_RUN_CONFIG/config.json", "w") as f:
        json.dump(saveable, f, indent=4)


class PipelineSession:
    """Owns one configured DeploymentPipeline and its run lifecycle."""

    def __init__(self, parsed: ParsedDeploymentConfig, *, reporter: Any = None):
        self.parsed = parsed
        self.reporter = reporter if reporter is not None else DefaultReporter()

        deployment_parameters = dict(parsed.deployment_parameters)
        DeploymentPipeline.apply_preset(parsed.pipeline_mode, deployment_parameters)

        self.pipeline = DeploymentPipeline(
            data_provider_factory=parsed.data_provider_factory,
            deployment_parameters=deployment_parameters,
            platform_constraints=parsed.platform_constraints,
            reporter=self.reporter,
            working_directory=parsed.working_directory,
        )
        if parsed.target_metric_override is not None:
            self.pipeline.set_target_metric(parsed.target_metric_override)

    @classmethod
    def from_config(
        cls,
        deployment_config: dict,
        *,
        data_provider_factory: Any = None,
        reporter: Any = None,
    ) -> "PipelineSession":
        parsed = parse_deployment_config(
            deployment_config, data_provider_factory=data_provider_factory
        )
        return cls(parsed, reporter=reporter)

    def attach_gui(self, gui: Any) -> None:
        """Wire a GUI handle (``.reporter``, ``.on_step_start``, ``.on_step_end``)."""
        self.pipeline.reporter = CompositeReporter([self.reporter, gui.reporter])
        self.pipeline.register_pre_step_hook(gui.on_step_start)
        self.pipeline.register_post_step_hook(gui.on_step_end)

    def resolved_start_step(self) -> Optional[str]:
        if self.parsed.start_step is None:
            return None
        return self.pipeline.get_resolved_start_step(self.parsed.start_step)

    def run(self) -> None:
        start_step = self.resolved_start_step()
        if start_step is None:
            self.pipeline.run(stop_step=self.parsed.stop_step)
        else:
            self.pipeline.run_from(step_name=start_step, stop_step=self.parsed.stop_step)

    def finish(self) -> None:
        with best_effort("reporter finish"):
            self.reporter.finish()
        close_pipeline_loaders(self.pipeline)
