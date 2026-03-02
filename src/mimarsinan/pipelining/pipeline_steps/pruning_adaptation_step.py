"""PruningAdaptationStep: pipeline step for progressive weight pruning.

Placed before weight quantization. Uses PruningTuner to gradually zero
the least significant rows and columns of each perceptron's weight matrix.
"""

from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.tuning.tuners.pruning_tuner import PruningTuner


class PruningAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.tuner = None

    def validate(self):
        return self.tuner.validate()

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        pruning_fraction = self.pipeline.config.get("pruning_fraction", 0.0)

        self.tuner = PruningTuner(
            self.pipeline,
            model=model,
            target_accuracy=self.pipeline.get_target_metric(),
            lr=self.pipeline.config["lr"] * 1e-3,
            adaptation_manager=adaptation_manager,
            pruning_fraction=pruning_fraction,
        )
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, "pickle")
        self.update_entry("model", model, "torch_model")
