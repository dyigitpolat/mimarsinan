"""TTFS-Cycle Fine-Tuning pipeline step."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner


class TTFSCycleAdaptationStep(TunerPipelineStep):
    @classmethod
    def applies_to(cls, plan):
        return plan.is_ttfs_cycle_based

    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        self._scale_aware_boundaries = bool(
            pipeline.config.get("ttfs_scale_aware_boundaries", False)
        )
        # Opt-in: scale-aware boundaries need the Activation-Analysis theta_out.
        # Adding it to ``requires`` only when the flag is on keeps the flag-off
        # path byte-identical (and never depends on Activation Analysis output).
        if self._scale_aware_boundaries:
            requires = requires + ["activation_scales"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        if self._scale_aware_boundaries:
            calibrate_scale_aware_boundaries(
                model, self.get_entry("activation_scales")
            )
        self.run_tuner(
            TTFSCycleAdaptationTuner,
            model,
            self.get_entry("adaptation_manager"),
        )
