from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.search.mlp_mixer_searcher import MLP_Mixer_Searcher
from mimarsinan.search.small_step_evaluator import SmallStepEvaluator
from mimarsinan.models.builders import PerceptronMixerBuilder

class ArchitectureSearchStep(PipelineStep):

    def __init__(self, pipeline):
        requires = []
        promises = ["model_config"]
        clears = []
        super().__init__(requires, promises, clears, pipeline)

    def validate(self):
        return self.pipeline.get_target_metric()

    def process(self):
        searcher = MLP_Mixer_Searcher(
            SmallStepEvaluator(
                self.pipeline.data_provider,
                self.pipeline.loss,
                self.pipeline.config['lr'],
                self.pipeline.config['device'],
                PerceptronMixerBuilder(
                    self.pipeline.config['input_shape'], 
                    self.pipeline.config['num_classes'], 
                    self.pipeline.config['max_axons'], 
                    self.pipeline.config['max_neurons'])
            )
        )
    
        model_config = searcher.get_optimized_configuration(
            self.pipeline.config['nas_cycles'],
            self.pipeline.config['nas_batch_size']
        )

        self.pipeline.cache.add("model_config", model_config)