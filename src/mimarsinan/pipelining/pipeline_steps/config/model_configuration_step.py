from mimarsinan.pipelining.core.model_config_emit import emit_model_config_entries
from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    PipelineStep,
)
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    build_platform_constraints_resolved,
)


class ModelConfigurationStep(PipelineStep):
    """Resolve model_config and platform_constraints from pipeline config (search_mode == "fixed", no architecture search)."""

    PROMISES = (
        "model_config",
        "model_builder",
        "platform_constraints_resolved",
    )

    @classmethod
    def applies_to(cls, plan):
        return plan.search_mode == "fixed"

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_CARRIED

    def process(self):
        emit_model_config_entries(self, self.pipeline.config)
        pcfg = build_platform_constraints_resolved(self.pipeline.config)
        self.add_entry("platform_constraints_resolved", pcfg)
