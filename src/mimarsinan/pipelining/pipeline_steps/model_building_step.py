from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.models.builders import PerceptronMixerBuilder

class ModelBuildingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model_config", "model_builder"]
        promises = ["model"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        builder = self.get_entry('model_builder')

        init_model = builder.build(self.get_entry("model_config"))
        self.add_entry("model", (init_model), "torch_model")