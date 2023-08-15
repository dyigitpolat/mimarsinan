from mimarsinan.search.basic_architecture_searcher import BasicArchitectureSearcher
from mimarsinan.search.mlp_mixer_configuration_sampler import MLP_Mixer_ConfigurationSampler
from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer
from mimarsinan.models.perceptron_mixer.skip_perceptron_mixer import SkipPerceptronMixer

import json
class MLP_Mixer_Searcher(BasicArchitectureSearcher):
    def __init__(self, evaluator):
        super().__init__()
        self.configuration_sampler = MLP_Mixer_ConfigurationSampler()
        self.evaluator = evaluator
    
    def _get_evaluator(self):
        return self.evaluator
    
    def _sample_configurations(self, n):
        return self.configuration_sampler.sample(n)
    
    def _update_sampler(self, metrics):
        self.configuration_sampler.update(metrics)