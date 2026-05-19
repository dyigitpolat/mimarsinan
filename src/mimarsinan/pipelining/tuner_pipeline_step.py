"""Base pipeline step for SmoothAdaptation / rate tuners."""

from __future__ import annotations

from mimarsinan.pipelining.pipeline_step import PipelineStep


class TunerPipelineStep(PipelineStep):
    """Shared validate/process/update pattern for tuner-backed steps."""

    def __init__(self, requires, promises, updates, clears, pipeline):
        super().__init__(requires, promises, updates, clears, pipeline)
        self.tuner = None

    def validate(self):
        if self.tuner is not None:
            return self.tuner.validate()
        return self.pipeline.get_target_metric()

    def _commit_tuner_entries(self, model, adaptation_manager):
        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, "torch_model")

    def run_tuner(self, tuner_cls, model, adaptation_manager, **tuner_kwargs):
        """Construct tuner, run, and commit cache entries."""
        self.tuner = tuner_cls(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"],
            adaptation_manager=adaptation_manager,
            **tuner_kwargs,
        )
        self.tuner.run()
        self._commit_tuner_entries(model, adaptation_manager)
