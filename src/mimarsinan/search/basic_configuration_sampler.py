import random 

class BasicConfigurationSampler:
    def __init__(self, configuration_space):
        self.previous_metrics = None
        self.refined_space = None

    def _get_configuration_space(self):
        raise NotImplementedError

    def _sample_from_space(self, space):
        sample = []
        for key in space:
            sample.append(random.choice(space[key]))

        return sample

    def update(self, metrics):
        if self.previous_metrics is not None:
            self.previous_metrics += metrics
            self.previous_metrics.sort(key = lambda x: x[1])
            length = len(self.previous_metrics)
            self.previous_metrics = self.previous_metrics[:length//2]
        else:
            self.previous_metrics = metrics

        top_p = 0.1
        top_metric_count = int(len(self.previous_metrics) * top_p)
        top_metrics = self.previous_metrics[:top_metric_count]

        self.refined_space = {}
        for metric in top_metrics:
            configuration = metric[0]
            for key in configuration:
                if key not in self.refined_space:
                    self.refined_space[key] = []
                self.refined_space[key].append(configuration[key])

    def sample(self, n):
        samples = []
        refined_samples_p = 0.5
        refined_samples_count = int(n * refined_samples_p)
        for _ in range(refined_samples_count):
            samples.append(self._sample_from_space(self.refined_space))
        
        for _ in range(n - refined_samples_count):
            samples.append(self._sample_from_space(self._get_configuration_space()))

        return samples
