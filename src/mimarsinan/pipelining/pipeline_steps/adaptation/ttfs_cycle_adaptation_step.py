"""TTFS-Cycle Fine-Tuning pipeline step."""

from mimarsinan.pipelining.core.steps.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import TTFSCycleAdaptationTuner


class TTFSCycleAdaptationStep(TunerPipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        self.run_tuner(
            TTFSCycleAdaptationTuner,
            self.get_entry("model"),
            self.get_entry("adaptation_manager"),
        )
