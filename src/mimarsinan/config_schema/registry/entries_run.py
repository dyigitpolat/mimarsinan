"""Registry entries: run identity, run control, and runtime-resolved keys."""

from __future__ import annotations

from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)


def _why_pipeline_mode(cfg: dict) -> str:
    if cfg.get("pipeline_mode") == "vanilla":
        return "vanilla — float-weight assembly (no quantization steps)"
    return "phased — weight or activation quantization is active"


ENTRIES = (
    _E("data_provider_name", section="top", group="workload", owner="PipelineSession/data_provider",
       type=T.STR, category=Category.BASIC, exposure="user", label="Data Provider",
       doc="Registered dataset provider id (see /api/data_providers)."),
    _E("experiment_name", section="top", group="run", owner="PipelineSession",
       type=T.STR, category=Category.BASIC, exposure="user", label="Experiment Name", important=True,
       doc="Run identifier; names the working directory under the generated files path."),
    _E("generated_files_path", section="top", group="run", owner="PipelineSession",
       type=T.PATH, category=Category.ADVANCED, exposure="user", label="Generated Files Path",
       doc="Root directory for run working directories (configs, caches, artifacts)."),
    _E("datasets_path", section="top", group="workload", owner="DataProviderFactory",
       type=T.PATH, category=Category.BASIC, exposure="user", label="Datasets Path",
       doc="Directory datasets are downloaded to / loaded from.",
       empty_means="./datasets"),
    _E("seed", section="top", group="run", owner="PipelineSession/determinism",
       type=T.INT, category=Category.BASIC, exposure="user", label="Random Seed",
       doc="Global torch/numpy seed; identical config + seed reproduces the step trajectory.",
       bounds=(0, None)),
    _E("start_step", section="top", group="run", owner="Pipeline/run_from",
       type=T.STR, category=Category.ADVANCED, exposure="user", label="Start Step",
       doc="Resume the pipeline from this step (cache permitting); null runs from the start.",
       empty_means="run from the first step"),
    _E("stop_step", section="top", group="run", owner="Pipeline/run",
       type=T.STR, category=Category.ADVANCED, exposure="user", label="Stop Step",
       doc="Stop the pipeline after this step; null runs to the end.",
       empty_means="run to the end"),
    _E("target_metric_override", section="top", group="run", owner="PipelineSession",
       type=T.FLOAT, category=Category.ADVANCED, exposure="user", label="Target Metric Override",
       doc="Seed the pipeline target metric with this value instead of 0 (resume workflows).",
       empty_means="the target metric seeds at 0"),
    _E("pipeline_mode", section="top", group="run", owner="deployment_derivation",
       type=T.ENUM, options=("vanilla", "phased"), category=Category.DERIVED,
       derivation="derived", exposure="derived", label="Pipeline Mode", important=True,
       doc="Assembly mode: phased enables the quantization/adaptation step chain; "
           "vanilla is the float-weight assembly.",
       effect="Phased enables weight/activation quantization steps",
       derived_from=("weight_quantization", "activation_quantization"),
       why=_why_pipeline_mode, declarable=True),
    _E("generate_visualizations", group="run", owner="visualization",
       type=T.BOOL, category=Category.ADVANCED, label="Generate Visualizations",
       doc="Write Graphviz/matplotlib artifact renderings during the run."),
    _E("num_workers", group="workload", owner="DataLoaderFactory",
       type=T.INT, category=Category.ADVANCED, label="DataLoader Workers",
       doc="torch DataLoader worker process count.", bounds=(0, None)),
    # Runtime-resolved keys: never declared in a config document; schema-known
    # so run views can label them instead of dropping them.
    _E("device", section="top", group="run", owner="DeploymentPipeline/select_device",
       type=T.STR, category=Category.RUNTIME, derivation="runtime", exposure="runtime",
       label="Device", doc="Resolved torch device (cuda/cpu).", declarable=False),
    _E("input_shape", section="top", group="run", owner="DeploymentPipeline/data_provider",
       type=T.SHAPE, category=Category.RUNTIME, derivation="runtime", exposure="runtime",
       label="Input Shape", doc="Resolved dataset input shape.", declarable=False),
    _E("input_size", section="top", group="run", owner="DeploymentPipeline/data_provider",
       type=T.INT, category=Category.RUNTIME, derivation="runtime", exposure="runtime",
       label="Input Size", doc="Flattened input size.", declarable=False),
    _E("num_classes", section="top", group="run", owner="DeploymentPipeline/data_provider",
       type=T.INT, category=Category.RUNTIME, derivation="runtime", exposure="runtime",
       label="Num Classes", doc="Resolved class count.", declarable=False),
)
