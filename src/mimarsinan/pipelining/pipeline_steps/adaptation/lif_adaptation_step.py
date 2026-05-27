"""LIF Adaptation pipeline step."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner


class LIFAdaptationStep(TunerPipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        self.run_tuner(
            LIFAdaptationTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
        )
