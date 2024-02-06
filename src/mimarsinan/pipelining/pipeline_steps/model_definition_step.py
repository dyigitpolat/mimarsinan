from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.models.builders import PerceptronMixerBuilder

class ModelDefinitionStep(PipelineStep):

    def __init__(self, pipeline):
        requires = []
        promises = ["model_config", "model_builder"]
        updates = []
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        builder = PerceptronMixerBuilder(
            self.pipeline.config['device'],
            self.pipeline.config['input_shape'], 
            self.pipeline.config['num_classes'], 
            self.pipeline.config['max_axons'], 
            self.pipeline.config['max_neurons'],
            self.pipeline.config)

        self.add_entry("model_builder", builder, 'pickle')
        self.add_entry("model_config", self.pipeline.config["model_definition"]["configuration"])