"""PruningAdaptationStep: pipeline step for progressive weight pruning."""

from mimarsinan.pipelining.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner


class PruningAdaptationStep(TunerPipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self.tuner = PruningTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"],
            adaptation_manager=adaptation_manager,
            pruning_fraction=self.pipeline.config.get("pruning_fraction", 0.0),
        )
        self.tuner.run()
        self._commit_tuner_entries(model, adaptation_manager)
