from mimarsinan.pipelining.model_config_emit import emit_model_config_entries
from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.pipelining.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)


class ModelConfigurationStep(PipelineStep):
    """
    Resolve model configuration and platform constraints from pipeline config.

    Used when ``search_mode == "fixed"`` (no architecture search).
    Takes model_config directly from ``pipeline.config['model_config']``.

    Emits ``platform_constraints_resolved``. Simulation length is read
    directly from ``pipeline.config['simulation_steps']`` by every
    downstream consumer — there is no separate "scaled" version.
    """

    def __init__(self, pipeline):
        requires = []
        promises = [
            "model_config",
            "model_builder",
            "platform_constraints_resolved",
        ]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        emit_model_config_entries(self, self.pipeline.config)
        pcfg = build_platform_constraints_resolved(self.pipeline.config)
        self.add_entry("platform_constraints_resolved", pcfg)
