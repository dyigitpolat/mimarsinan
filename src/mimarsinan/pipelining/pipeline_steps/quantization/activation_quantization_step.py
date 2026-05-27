from mimarsinan.pipelining.tuner_pipeline_step import TunerPipelineStep
from mimarsinan.tuning.tuners.activation_quantization_tuner import ActivationQuantizationTuner


class ActivationQuantizationStep(TunerPipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")
        self.run_tuner(
            ActivationQuantizationTuner,
            model,
            adaptation_manager,
            target_tq=self.pipeline.config["target_tq"],
        )
