import random 
import copy

class BasicConfigurationSampler:
    def __init__(self):
        self.previous_metrics = None
        self.refined_space = self._get_configuration_space()
        self.top_metrics = None

    def _get_configuration_space(self):
        raise NotImplementedError

    def _sample_from_space(self, space):
        sample = {}
        for key in space:
            sample[key] = random.choice(space[key])

        return sample
    
    def _mutate(self, configuration, rate):
        mutator = self._sample_from_space(self._get_configuration_space())
        for key in mutator:
            if random.random() < rate:
                configuration[key] = mutator[key]

    def update(self, metrics):
        if self.previous_metrics is not None:
            self.previous_metrics += metrics
            self.previous_metrics.sort(key = lambda x: x[1])
            length = len(self.previous_metrics)
            half_length = max(1, length // 2)
            self.previous_metrics = copy.deepcopy((self.previous_metrics[-half_length:]))
        else:
            self.previous_metrics = copy.deepcopy((metrics))
            self.previous_metrics.sort(key = lambda x: x[1])

        top_p = 0.1
        top_metric_count = max(1, int(len(self.previous_metrics) * top_p))
        self.top_metrics = copy.deepcopy((self.previous_metrics[-top_metric_count:]))

        print("top metrics", self.top_metrics)

        self.refined_space = {}
        for metric in self.top_metrics:
            configuration = metric[0]
            for key in configuration:
                if key not in self.refined_space:
                    self.refined_space[key] = []
                if configuration[key] not in self.refined_space[key]:
                    self.refined_space[key].append(configuration[key])

    def sample(self, n):
        assert n > 2
        
        if self.top_metrics:
            mutated_top_candidates_p = 0.6
            samples_from_refined_space_p = 0.3
        else:
            mutated_top_candidates_p = 0.0
            samples_from_refined_space_p = 0.9

        mutated_top_candidates_count = int(n * mutated_top_candidates_p)
        samples_from_refined_space_count = int(n * samples_from_refined_space_p)

        assert mutated_top_candidates_count + samples_from_refined_space_count < n, f"{mutated_top_candidates_count} + {samples_from_refined_space_count} >= {n}"
        vanilla_samples_count = n - (mutated_top_candidates_count + samples_from_refined_space_count)

        samples = []
        for _ in range(mutated_top_candidates_count):
            sample = random.choice(self.top_metrics)[0]
            self._mutate(sample, 0.5)
            samples.append(sample)

        for _ in range(samples_from_refined_space_count):
            refined_sample = self._sample_from_space(self.refined_space)
            self._mutate(refined_sample, 0.5)
            samples.append(refined_sample)
        
        for _ in range(n - vanilla_samples_count):
            samples.append(self._sample_from_space(self._get_configuration_space()))

        return samples
