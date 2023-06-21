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
                print("evaluating", configuration)
                score = self._evaluate_architecture(configuration)
                metrics.append(
                    (configuration, self._evaluate_architecture(configuration)))
                print("score", score)
        return metrics

    def get_optimized_model(self):
        cycles = 5
        configuration_batch_size = 50
        for i in range(cycles):
            print(i)

            metrics = []
            while len(metrics) < configuration_batch_size:
                configurations = self._sample_configurations(configuration_batch_size)
                metrics += self._evaluate_configurations(configurations)
            
            self._update_sampler(metrics)
            best_configuration = max(metrics, key = lambda x: x[1])[0]
        
        print(best_configuration)
        return self._create_model(best_configuration)
