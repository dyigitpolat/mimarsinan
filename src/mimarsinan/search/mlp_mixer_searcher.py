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
        patch_n_1 = configuration["patch_n_1"]
        patch_m_1 = configuration["patch_m_1"]
        patch_c_1 = configuration["patch_c_1"]
        fc_k_1 = configuration["fc_k_1"]
        fc_w_1 = configuration["fc_w_1"]

        patch_n_2 = configuration["patch_n_2"]
        patch_c_2 = configuration["patch_c_2"]
        fc_k_2 = configuration["fc_k_2"]
        fc_w_2 = configuration["fc_w_2"]

        assert self.input_shape[-2] % patch_n_1 == 0, \
            "mod div != 0"
        assert self.input_shape[-1] % patch_m_1 == 0, \
            "mod div != 0"
        
        fc_in = patch_n_1 * patch_m_1 * patch_c_1
        fc_width = fc_w_1
        patch_height = self.input_shape[-2] // patch_n_1
        patch_width = self.input_shape[-1] // patch_m_1
        input_channels = self.input_shape[-3]
        patch_size = patch_height * patch_width * input_channels

        assert fc_width <= self.max_neurons, f"not enough neurons ({fc_width} > {self.max_neurons})"
        assert fc_width <= self.max_axons - 1, f"not enough axons ({fc_width} > {self.max_axons})"
        assert fc_in <= self.max_axons - 1, f"not enough axons ({fc_in} > {self.max_axons})"
        assert patch_size <= self.max_axons - 1, f"not enough axons ({patch_size} > {self.max_axons})"

        fc_width_2 = fc_w_2
        fc_in_2 = patch_n_2 * patch_c_2
        patch_size_2 = fc_width // patch_n_2

        assert fc_width % patch_n_2 == 0, \
            "mod div != 0"

        assert fc_width_2 <= self.max_neurons, f"not enough neurons ({fc_width_2} > {self.max_neurons})"
        assert fc_width_2 <= self.max_axons - 1, f"not enough axons ({fc_width_2} > {self.max_axons})"
        assert fc_in_2 <= self.max_axons - 1, f"not enough axons ({fc_in_2} > {self.max_axons})"
        assert patch_size_2 <= self.max_axons - 1, f"not enough axons ({patch_size_2} > {self.max_axons})"

        perceptron_flow = PerceptronMixer(
            self.input_shape, self.num_classes,
            patch_n_1, patch_m_1, patch_c_1, fc_w_1, fc_k_1,
            patch_n_2, patch_c_2, fc_w_2, fc_k_2)
    
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