"""LIF Adaptation pipeline step."""

from mimarsinan.common.workload_profile import ResolvedWorkloadProfile
from mimarsinan.pipelining.core.registry.trainer_factory import make_basic_trainer
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.spiking.wire_gauge_seam import establish_deployed_wire_gauge
from mimarsinan.tuning.orchestration.lif_exact_qat import deployed_negative_boundary_seam
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
        self._establish_gauge(model, trainer=lambda: make_basic_trainer(self.pipeline, model))
        self.run_tuner(
            LIFAdaptationTuner, model, self.get_entry("adaptation_manager"),
        )

    def _establish_gauge(self, model, trainer=None) -> None:
        """The chip-aligned LIF forward this step finalizes on is a wire-currency
        twin, so EVERY graph has its gauge established first (U-SC S-1): a host
        op that computes on values decodes each source at its producer's gauge
        only once the wrap slots are armed (a classifiable graph read unarmed is
        the C5 retention-gate defect, SD-10a), and a heterogeneous fan-in (a host
        residual re-joining a branch that crossed a core) has no domain at all
        until its join is repaired."""
        if getattr(model, "get_mapper_repr", lambda: None)() is None:
            # No mapper graph, no seams — the twin falls back to the plain
            # cycle-accurate forward (``chip_aligned_segment_forward``).
            return
        # [U-SC2 A-2] with the negative-value boundary policy (calibrated from ``trainer``, a factory called only when
        # the graph has a boundary to lift), so the twin and the retention metric read the boundary SCM deploys.
        repaired = establish_deployed_wire_gauge(
            model,
            input_data_scale=ResolvedWorkloadProfile.from_config(
                self.pipeline.config
            ).input_data_scale,
            negative_boundary=deployed_negative_boundary_seam(model, trainer, self.pipeline.config),
        )
        if repaired:
            print(
                f"[LIFAdaptationStep] established the wire gauge for {repaired} "
                "mixed-domain seam(s); the twin now decodes each source at its "
                "producer's gauge, as the deployed op does."
            )
