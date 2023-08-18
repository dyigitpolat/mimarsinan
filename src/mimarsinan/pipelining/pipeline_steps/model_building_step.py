from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.models.builders import PerceptronMixerBuilder

class ModelBuildingStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model_config", "model_builder"]
        promises = ["init_model"]
        clears = []
        super().__init__(requires, promises, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        builder = self.pipeline.cache.get('model_builder')

        init_model = builder.build(self.pipeline.cache.get("model_config"))
        self.pipeline.cache.add("init_model", (init_model), "torch_model")