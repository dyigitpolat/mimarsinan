"""Base pipeline step for SmoothAdaptation / rate tuners."""

from __future__ import annotations

from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.tuning.orchestration.conversion_draws import (
    configured_draws,
    run_conversion_draws,
)


class TunerPipelineStep(PipelineStep):
    """Shared validate/process/update pattern for tuner-backed steps."""

    DRAW_SELECTED = False
    """Opt-in for the [MBH-DRAWS] best-of-N conversion harness: the mode's
    variance-carrying ramp/endpoint stages select it; every other tuner step
    stays single-draw (``conversion_draws`` is then inert)."""

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
        """Construct tuner (best-of-N draws when selected), run, and commit the
        winning draw's cache entries."""
        def build(draw_model, draw_manager):
            return tuner_cls(
                self.pipeline,
                model=draw_model,
                target_accuracy=self.pipeline.get_target_metric(),
                lr=self.pipeline.config["lr"],
                adaptation_manager=draw_manager,
                **tuner_kwargs,
            )

        draws = configured_draws(self.pipeline) if self.DRAW_SELECTED else 1
        self.tuner, model, adaptation_manager = run_conversion_draws(
            self.pipeline, build, model, adaptation_manager, draws=draws,
        )
        self._report_ft_pass_wall()
        self._commit_tuner_entries(model, adaptation_manager)

    def _report_ft_pass_wall(self):
        """Surface the worst single fine-tuning-pass wall into the reported metrics; no-op when absent."""
        if self.tuner is None:
            return
        wall = getattr(self.tuner, "max_ft_pass_wall_s", None)
        if wall is not None:
            self.pipeline.reporter.report("max_ft_pass_wall_s", float(wall))
