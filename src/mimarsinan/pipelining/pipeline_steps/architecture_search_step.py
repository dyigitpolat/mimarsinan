from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.search.mlp_mixer_searcher import MLP_Mixer_Searcher
from mimarsinan.search.small_step_evaluator import SmallStepEvaluator
from mimarsinan.models.builders import PerceptronMixerBuilder

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.adaptation_manager import AdaptationManager
class ArchitectureSearchStep(PipelineStep):

    def __init__(self, pipeline):
        requires = []
        promises = ["model_config", "model_builder", "adaptation_manager"]
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
        
        searcher = MLP_Mixer_Searcher(
            SmallStepEvaluator(
                self.pipeline.data_provider_factory,
                self.pipeline.loss,
                self.pipeline.config['lr'],
                self.pipeline.config['device'],
                builder),
            self.pipeline.config['nas_workers']
        )
    
        model_config = searcher.get_optimized_configuration(
            self.pipeline.config['nas_cycles'],
            self.pipeline.config['nas_batch_size']
        )

        self.add_entry("adaptation_manager", AdaptationManager(), 'pickle')
        self.add_entry("model_builder", builder, 'pickle')
        self.add_entry("model_config", model_config)