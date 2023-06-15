from mimarsinan.search.basic_architecture_searcher import BasicArchitectureSearcher
from mimarsinan.search.mlp_mixer_configuration_sampler import MLP_Mixer_ConfigurationSampler
from mimarsinan.models.perceptron_mixer.perceptron_mixer import PerceptronMixer

import json
class MLP_Mixer_Searcher(BasicArchitectureSearcher):
    def __init__(self, input_shape, num_classes, max_axons, max_neurons, evaluator):
        super().__init__()
        self.configuration_sampler = MLP_Mixer_ConfigurationSampler()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.evaluator = evaluator

        self.cache = {}

    def _configuration_to_json(self, configuration):
        return json.dumps(configuration)

    def _evaluate_architecture(self, configuration):
        key = self._configuration_to_json(configuration)
        if key in self.cache:
            return self.cache[key]
        
        model = self._create_model(configuration)
        self.cache[key] = self.evaluator.evaluate(model)

        return self.cache[key]

    def _create_model(self, configuration):
        div = configuration["input_patch_division"]
        fc_count = configuration["fc_count"]

        assert self.input_shape[-2] % div == 0, \
            "mod div != 0"
        assert self.input_shape[-1] % div == 0, \
            "mod div != 0"

        patch_rows = max(self.input_shape[-2] // div, 1)
        patch_cols = max(self.input_shape[-1] // div, 1)

        perceptron_flow = PerceptronMixer(
            self.input_shape, self.num_classes,
            self.max_axons - 1, self.max_neurons,
            patch_cols, 
            patch_rows, fc_depth=fc_count)
    
        return perceptron_flow

    def _validate_configuration(self, configuration):
        try:
            self._create_model(configuration)
            return True
        except Exception as e:
            return False
    
    def _sample_configurations(self, n):
        return self.configuration_sampler.sample(n)
    
    def _update_sampler(self, metrics):
        self.configuration_sampler.update(metrics)