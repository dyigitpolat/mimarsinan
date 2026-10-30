"""Composition root: one place that turns a deployment config into a running pipeline."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import mimarsinan.data_handling.data_providers  # noqa: F401  # pyright: ignore[reportUnusedImport] — registers built-in providers
from mimarsinan.advisories import (
    evaluate_config_advisories,
    evaluate_post_pretrain_advisories,
    surface_advisories,
)
from mimarsinan.common.best_effort import best_effort
from mimarsinan.common.reporter import DefaultReporter
from mimarsinan.config_schema.runtime import (
    enforce_run_contract, merge_search_space, run_config, run_sections, top_declaration,
)
from mimarsinan.data_handling.data_loader_factory import close_pipeline_loaders
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory
from mimarsinan.gui.runtime.composite_reporter import CompositeReporter
from mimarsinan.config_schema.registry import declaration_judge
from mimarsinan.config_schema.validation import missing_top_level_errors, section_shape_errors
from mimarsinan.config_schema.registry.value_schema import DeclarationValueError
import mimarsinan.mapping.platform.target_expansion as target_expansion
# The session APPLIES determinism, before any step runs (below); the contract
# itself lives beside its counterpart `isolated_rng_stream`, so the families one
# seeds and the other restores cannot drift apart.
from mimarsinan.pipelining.determinism import (
    apply_determinism as apply_determinism,  # noqa: F401 — re-export
)
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
    """Parse a deployment-config dict and persist it under ``_RUN_CONFIG/``.

    A declared target expands into both sections first (``expand_target_document``, the document door's
    reading, no labels: the merge writes them). Then, before anything reads a declaration, the session refuses by
    name a top-level key it reads that the document leaves absent or null (``missing_top_level_errors``: the
    deployment parameters, the experiment name, the generated-files root and the provider name, which a document
    declares whichever factory a caller hands in), a section that is no object, and every declaration the one
    judge refuses
    (``document_declaration_errors``, G4 interface note 2). Every read then takes the run's one layout
    (``run_sections``: the top-level seed and pipeline mode laid beneath the deployment parameters, where the
    derivation reads them, a null declaration unset): the data provider's seed, preprocessing and batch size, the
    hardware-search branch and the quantization assembly contract (``enforce_run_contract``). The pipeline mode,
    which names the working directory and the preset, is the one the run derives (``run_config``, the merge's own
    derivation, which refuses a declared mode it would replace). The persisted document is the one declared."""
    document = target_expansion.expand_target_document(deployment_config, write_derived=False)
    required = ("deployment_parameters", "experiment_name", "generated_files_path", "data_provider_name")
    errors = missing_top_level_errors(document, required)
    errors += [*section_shape_errors(document), *declaration_judge.document_declaration_errors(document)]
    if errors:
        raise DeclarationValueError("; ".join(errors))
    deployment_name = document["experiment_name"]
    deployment_parameters, platform_constraints, layout = run_sections(document)
    enforce_run_contract(deployment_parameters, platform_constraints)
    pipeline_mode = str(run_config(copy.deepcopy(deployment_parameters),
                                   copy.deepcopy(platform_constraints))["pipeline_mode"])

    if data_provider_factory is None:
        data_provider_factory = BasicDataProviderFactory(
            document["data_provider_name"],
            top_declaration(document, "datasets_path", "./datasets"),
            seed=layout["seed"],
            preprocessing=layout.get("preprocessing"),
            batch_size=layout.get("batch_size"),
        )

    working_directory = deployment_config.get(
        "_working_directory",
        f"{document['generated_files_path']}/"
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


def _resolve_platform_constraints(deployment_config: dict, deployment_parameters: dict) -> Any:
    """In hw search mode, merge search_space into arch_search without mutating the input (``merge_search_space``,
    the run's reading in ``run_sections``)."""
    return merge_search_space(deployment_config.get("platform_constraints", {}), deployment_parameters)


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
        # [U-H] the run's name, for the records that name the run that made them (a deployment bundle's provenance).
        self.pipeline.deployment_name = parsed.deployment_name
        apply_determinism(self.pipeline.plan.seed)
        if parsed.target_metric_override is not None:
            self.pipeline.set_target_metric(parsed.target_metric_override)
        self.pipeline.register_post_step_hook(self._post_pretraining_advisory_hook)

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
        """Wire a GUI handle (``.reporter``, ``.on_step_start``, ``.on_step_end``, ``.on_step_failed``)."""
        self.pipeline.reporter = CompositeReporter([self.reporter, gui.reporter])
        self.pipeline.register_pre_step_hook(gui.on_step_start)
        self.pipeline.register_post_step_hook(gui.on_step_end)
        self.pipeline.register_step_failed_hook(gui.on_step_failed)

    def resolved_start_step(self) -> Optional[str]:
        if self.parsed.start_step is None:
            return None
        return self.pipeline.get_resolved_start_step(self.parsed.start_step)

    def surface_config_advisories(self) -> None:
        """Config-time deployment advisories at session start: loud [ADVISORY]
        prints + reporter events. Warnings only — they never gate a run."""
        with best_effort("deployment advisories (config)"):
            surface_advisories(
                self.pipeline.reporter,
                evaluate_config_advisories(self.pipeline.plan),
                context="config",
            )

    def _post_pretraining_advisory_hook(self, name: str, step: Any) -> None:
        """Envelope-gate advisory on the recorded pretrain metric (post-step
        hooks run after the engine records the step metric)."""
        if name != "Pretraining":
            return
        with best_effort("deployment advisories (post-pretrain)"):
            surface_advisories(
                self.pipeline.reporter,
                evaluate_post_pretrain_advisories(
                    float(self.pipeline.get_target_metric()),
                    self.pipeline.config,
                    acceptance_target=self.parsed.target_metric_override,
                ),
                context="post_pretrain",
            )

    def run(self) -> None:
        self.surface_config_advisories()
        start_step = self.resolved_start_step()
        if start_step is None:
            self.pipeline.run(stop_step=self.parsed.stop_step)
        else:
            self.pipeline.run_from(step_name=start_step, stop_step=self.parsed.stop_step)

    def finish(self) -> None:
        with best_effort("reporter finish"):
            self.reporter.finish()
        close_pipeline_loaders(self.pipeline)
