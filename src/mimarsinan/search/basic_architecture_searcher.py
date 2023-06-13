class BasicArchitectureSearcher:
    def __init__(self):
        pass

    def _evaluate_architecture(self, configuration):
        raise NotImplementedError

    def _create_model(self, configuration):
        raise NotImplementedError

    def _sample_configurations(self, n):
        raise NotImplementedError

    def _validate_configuration(self, configuration):
        raise NotImplementedError
    
    def _update_sampler(self, metrics):
        raise NotImplementedError
    
    def _evaluate_configurations(self, configurations):
        metrics = []
        for configuration in configurations:
            if self._validate_configuration(configuration):
                metrics.append(
                    (configuration, self._evaluate_architecture(configuration)))
        return metrics

    def get_optimized_model(self):
        cycles = 10
        configuration_batch_size = 100
        for _ in range(cycles):
            configurations = self._sample_configurations(configuration_batch_size)

            metrics = self._evaluate_configurations(configurations)
            
            self._update_sampler(metrics)
            best_configuration = max(metrics, key = lambda x: x[1])
        
        return self._create_model(best_configuration)
