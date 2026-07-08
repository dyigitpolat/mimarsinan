"""TTFS-Cycle Fine-Tuning pipeline step."""

from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner


class TTFSCycleAdaptationStep(TunerPipelineStep):
    # activation_scales is an instance-specific opt-in requirement added in __init__, so the class REQUIRES stays a sound assembly-time lower bound (Activation Analysis always produces it earlier).
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    # The cascade FT ramp+endpoint is a variance-carrying stage.
    DRAW_SELECTED = True

    @classmethod
    def applies_to(cls, plan):
        return plan.is_cascaded_ttfs

    def __init__(self, pipeline):
        requires = list(self.REQUIRES)
        self._scale_aware_boundaries = bool(
            pipeline.config.get("ttfs_scale_aware_boundaries", False)
        )
        if self._scale_aware_boundaries:
            requires = requires + ["activation_scales"]
        super().__init__(requires, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        model = self.get_entry("model")
        if self._scale_aware_boundaries:
            calibrate_scale_aware_boundaries(
                model,
                self.get_entry("activation_scales"),
                input_data_scale=DeploymentPlan.of(
                    self.pipeline
                ).workload.input_data_scale,
            )
        self.run_tuner(
            TTFSCycleAdaptationTuner,
            model,
            self.get_entry("adaptation_manager"),
        )
