"""LIF Adaptation pipeline step."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner


class LIFAdaptationStep(TunerPipelineStep):
    REQUIRES = ("model", "adaptation_manager")
    UPDATES = ("model", "adaptation_manager")

    # [MBH-DRAWS] the LIF ramp+endpoint is a variance-carrying conversion stage.
    DRAW_SELECTED = True

    @classmethod
    def applies_to(cls, plan):
        return plan.spiking_mode == "lif"

    def __init__(self, pipeline):
        super().__init__(self.REQUIRES, self.PROMISES, self.UPDATES, self.CLEARS, pipeline)

    def process(self):
        self.run_tuner(
            LIFAdaptationTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
        )
