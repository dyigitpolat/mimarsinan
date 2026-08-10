"""Base pipeline step for SmoothAdaptation / rate tuners."""

from __future__ import annotations

from mimarsinan.pipelining.core.steps.pipeline_step import (
    METRIC_CARRIED,
    METRIC_MEASURED,
    PipelineStep,
)
from mimarsinan.tuning.orchestration import endpoint_steps, run_instrumentation
from mimarsinan.tuning.orchestration.conversion_draws import (
    configured_draws,
    run_conversion_draws,
)
from mimarsinan.tuning.orchestration.retention_envelope import resolve_step_anchor


class TunerPipelineStep(PipelineStep):
    """Shared validate/process/update pattern for tuner-backed steps."""

    DRAW_SELECTED = False
    """Opt-in for the best-of-N conversion harness: the mode's variance-carrying
    ramp/endpoint stages select it; every other tuner step stays single-draw
    (``conversion_draws`` is then inert)."""

    def __init__(self, requires, promises, updates, clears, pipeline):
        super().__init__(requires, promises, updates, clears, pipeline)
        self.tuner = None

    def run(self):
        # Entry snapshot for the retention ledger's endpoint-step accounting
        # (a read-only ledger peek; the tuner consumes during process()).
        self._endpoint_steps_consumed_before = endpoint_steps.consumed(self.pipeline)
        super().run()

    def validate(self):
        if self.tuner is not None:
            return self.tuner.validate()
        return self.pipeline.get_target_metric()

    def validate_metric_kind(self) -> str:
        return METRIC_MEASURED if self.tuner is not None else METRIC_CARRIED

    def _commit_tuner_entries(self, model, adaptation_manager):
        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, "torch_model")
        self._persist_adaptation_instrumentation()

    def _persist_adaptation_instrumentation(self):
        """Persist the run-dir adaptation artifacts at commit time (W3-S1):
        the ``ft_pass_walls.json`` accumulator + one ``retention_ledger.json``
        entry. Artifacts only — nothing in the training path reads them."""
        tuner = self.tuner
        if tuner is None:
            return
        working_directory = getattr(self.pipeline, "working_directory", None)
        if working_directory is None:
            return
        wall_metrics = getattr(tuner, "ft_pass_wall_metrics", None)
        if wall_metrics is not None:
            run_instrumentation.merge_ft_pass_walls(
                working_directory, self.name, wall_metrics()
            )
        if getattr(tuner, "validate", None) is None:
            return  # not a TunerBase family: no exit read to account
        entry = run_instrumentation.retention_entry(
            step_name=self.name,
            entry_metric=getattr(self, "pipeline_previous_metric", None),
            exit_metric=float(self.validate()),
            pipeline=self.pipeline,
            consumed_before=getattr(self, "_endpoint_steps_consumed_before", None),
        )
        run_instrumentation.append_retention_entry(working_directory, entry)

    def run_tuner(self, tuner_cls, model, adaptation_manager, **tuner_kwargs):
        """Construct tuner (best-of-N draws when selected), run, and commit the
        winning draw's cache entries."""
        # The origin-anchored compact (calculus §13.2 L-B) swaps the rolling
        # previous-step anchor for the ORIGIN metric when armed.
        target = resolve_step_anchor(self.pipeline)

        def build(draw_model, draw_manager):
            return tuner_cls(
                self.pipeline,
                model=draw_model,
                target_accuracy=target,
                lr=self.pipeline.config["lr"],
                adaptation_manager=draw_manager,
                **tuner_kwargs,
            )

        draws = configured_draws(self.pipeline) if self.DRAW_SELECTED else 1
        self.tuner, model, adaptation_manager = run_conversion_draws(
            self.pipeline, build, model, adaptation_manager, draws=draws,
            target=target,
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
