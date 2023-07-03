from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.search.mlp_mixer_searcher import MLP_Mixer_Searcher
from mimarsinan.search.small_step_evaluator import SmallStepEvaluator

class ArchitectureSearchStep(PipelineStep):

    def __init__(self, pipeline):
        requires = []
        promises = ["init_model", "model_config"]
        clears = []
        super().__init__(requires, promises, clears, pipeline)

    def process(self):
        searcher = MLP_Mixer_Searcher(
            self.pipeline.config['input_shape'], 
            self.pipeline.config['num_classes'], 
            self.pipeline.config['max_axons'], 
            self.pipeline.config['max_neurons'], 
            SmallStepEvaluator(
                self.pipeline.data_provider,
                self.pipeline.loss,
                self.pipeline.config['lr'],
                self.pipeline.config['device']))
        
        model_config = searcher.get_optimized_configuration(
            self.pipeline.config['nas_cycles'],
            self.pipeline.config['nas_batch_size']
        )

        model = searcher._create_model(model_config)
        
        self.pipeline.cache.add("init_model", model, 'torch_model')
        self.pipeline.cache.add("model_config", model_config)