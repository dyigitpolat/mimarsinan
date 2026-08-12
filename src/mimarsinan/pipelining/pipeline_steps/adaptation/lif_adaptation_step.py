"""LIF Adaptation pipeline step."""

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.spiking.scale_aware_boundaries import (
    establish_gauge_for_mixed_domain_seams,
)
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner


class LIFAdaptationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    # The LIF ramp+endpoint is a variance-carrying conversion stage.
    DRAW_SELECTED = True

    @classmethod
    def applies_to(cls, plan):
        return plan.spiking_mode == "lif"

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        self._establish_gauge(model)
        self.run_tuner(
            LIFAdaptationTuner, model, self.get_entry("adaptation_manager"),
        )

    def _establish_gauge(self, model) -> None:
        """The chip-aligned LIF forward this step finalizes on is a wire-currency
        twin, so a graph with a heterogeneous fan-in (a host residual re-joining
        a branch that crossed a core) must have its gauge established first —
        otherwise the twin has no domain there at all."""
        repaired = establish_gauge_for_mixed_domain_seams(
            model,
            input_data_scale=ResolvedWorkloadProfile.from_config(
                self.pipeline.config
            ).input_data_scale,
        )
        if repaired:
            print(
                f"[LIFAdaptationStep] established the wire gauge for {repaired} "
                "mixed-domain seam(s); the twin now decodes each source at its "
                "producer's gauge, as the deployed op does."
            )
