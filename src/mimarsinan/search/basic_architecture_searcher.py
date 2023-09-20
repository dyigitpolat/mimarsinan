import torch.multiprocessing as mp

import copy

class BasicArchitectureSearcher:
    def __init__(self, nas_workers):
        self._nas_workers = nas_workers

    def _get_evaluator(self, configuration):
        raise NotImplementedError

    def _create_model(self, configuration):
        raise NotImplementedError

    def _sample_configurations(self, n):
        raise NotImplementedError
    
    def _update_sampler(self, metrics):
        raise NotImplementedError
    
    def _evaluate_configurations(self, configurations):
        for configuration in configurations:
            assert self._get_evaluator().validate(configuration), \
                f"unexpected error occured, invalid configurations may have been sampled"

        print(f"evaluating {len(configurations)} configurations")
        with mp.Pool(processes = self._nas_workers) as pool:
            params = [(self._get_evaluator(), configuration) for configuration in configurations]
            metrics = pool.map(
                evaluate_configuration_runner, params)

        return metrics
    
    def _sample_valid_configurations(self, sample_size_max):
        configurations = []

        total_sample_size = 0
        while len(configurations) < sample_size_max:
            sampled_configurations = self._sample_configurations(sample_size_max)

            for configuration in sampled_configurations:
                if self._get_evaluator().validate(configuration):
                    configurations.append(copy.deepcopy(configuration))

            total_sample_size += sample_size_max
            assert total_sample_size < 10000 * sample_size_max, \
                f"unexpected error occured, cannot sample valid configurations"

        return configurations
    
    def get_optimized_configuration(self, cycles = 5, configuration_batch_size = 50):
        best_configuration_metric_pair = (None, 0)
        for i in range(cycles):
            print("Cycle", i, ":")

            configurations = self._sample_valid_configurations(configuration_batch_size)
            metrics = self._evaluate_configurations(configurations)
        
            self._update_sampler(metrics)
            best_configuration_metric_pair = max(
                copy.deepcopy(metrics) + [best_configuration_metric_pair], key = lambda x: x[1])
        
        return best_configuration_metric_pair[0]
    
    def get_optimized_model(self, cycles = 5, configuration_batch_size = 50):
        config = self.get_optimized_configuration(cycles, configuration_batch_size)
        return self._create_model(config)
    
def evaluate_configuration_runner(param_tuple):
    evaluator, configuration = param_tuple
    return (configuration, evaluator.evaluate(configuration))