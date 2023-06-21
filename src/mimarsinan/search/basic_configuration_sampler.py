import random 

class BasicConfigurationSampler:
    def __init__(self):
        self.previous_metrics = None
        self.refined_space = self._get_configuration_space()

    def _get_configuration_space(self):
        raise NotImplementedError

    def _sample_from_space(self, space):
        sample = {}
        for key in space:
            sample[key] = random.choice(space[key])

        return sample
    
    def _mutate(self, configuration):
        mutator = self._sample_from_space(self._get_configuration_space())
        for key in mutator:
            if random.random() < 0.2:
                configuration[key] = mutator[key]

    def update(self, metrics):
        if self.previous_metrics is not None:
            self.previous_metrics += metrics
            self.previous_metrics.sort(key = lambda x: x[1])
            length = len(self.previous_metrics)
            self.previous_metrics = self.previous_metrics[-length//2:]
        else:
            self.previous_metrics = metrics
            self.previous_metrics.sort(key = lambda x: x[1])

        top_p = 0.2
        top_metric_count = int(len(self.previous_metrics) * top_p)
        top_metrics = self.previous_metrics[-top_metric_count:]

        print("top metrics", top_metrics)

        self.refined_space = {}
        for metric in top_metrics:
            configuration = metric[0]
            for key in configuration:
                if key not in self.refined_space:
                    self.refined_space[key] = []
                if configuration[key] not in self.refined_space[key]:
                    self.refined_space[key].append(configuration[key])

    def sample(self, n):
        samples = []
        refined_samples_p = 0.8
        refined_samples_count = int(n * refined_samples_p)

        for _ in range(refined_samples_count):
            refined_sample = self._sample_from_space(self.refined_space)
            self._mutate(refined_sample)
            samples.append(refined_sample)
        
        for _ in range(n - refined_samples_count):
            samples.append(self._sample_from_space(self._get_configuration_space()))

        return samples
