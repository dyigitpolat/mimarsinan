import random

class MultiMetricSearcher:
    def __init__(self, top_n=10, generations=20, population_size=50, selection_ratio=0.2):
        self.top_n = top_n
        self.generations = generations
        self.population_size = population_size
        self.selection_ratio = selection_ratio

    def search(self, configuration_sampler, configuration_mutator, evaluator):
        population = [configuration_sampler() for _ in range(self.population_size)]

        for generation in range(self.generations):
            scores = [(candidate, evaluator(candidate)) for candidate in population]

            current_selection_ratio = self.selection_ratio
            while True:
                metric1_scores = sorted([score[1][0] for score in scores], reverse=True)
                metric2_scores = sorted([score[1][1] for score in scores], reverse=True)

                threshold1 = metric1_scores[int(len(metric1_scores) * current_selection_ratio)]
                threshold2 = metric2_scores[int(len(metric2_scores) * current_selection_ratio)]

                top_candidates = [
                    candidate for candidate, (m1, m2) in scores
                    if m1 >= threshold1 and m2 >= threshold2
                ]

                if top_candidates or current_selection_ratio >= 1.0:
                    break

                current_selection_ratio = min(current_selection_ratio + self.selection_ratio / 2.0, 1.0)

            if not top_candidates:
                top_candidates = [candidate for candidate, _ in scores]

            top_candidates = sorted(
                top_candidates,
                key=lambda c: sum(evaluator(c)),
                reverse=True
            )[:self.top_n]

            population = [
                configuration_mutator(random.choice(top_candidates))
                for _ in range(self.population_size)
            ]

        best_candidate = max(population, key=lambda c: sum(evaluator(c)))
        return best_candidate, evaluator(best_candidate)


def configuration_sampler():
    return [random.uniform(0, 10) for _ in range(2)]

def configuration_mutator(candidate):
    return [
        min(max(x + random.uniform(-1, 1), 0), 10) for x in candidate
    ]

def evaluator(candidate):
    x1, x2 = candidate
    metric1 = -(x1 - 5)**2 - (x2 - 5)**2
    metric2 = -(x1 - 0)**2 - (x2 - 0)**2
    return metric1, metric2

random_samples = [configuration_sampler() for _ in range(5)]
print("Randomly Sampled Configurations and Metrics (Before Search):")
for i, sample in enumerate(random_samples, 1):
    metrics = evaluator(sample)
    print(f"Sample {i}: Configuration = {sample}, Metrics = {metrics}")

searcher = MultiMetricSearcher(top_n=10, generations=500, population_size=1000, selection_ratio=0.2)

best_candidate, best_scores = searcher.search(
    configuration_sampler,
    configuration_mutator,
    evaluator
)

print("Best Candidate:", best_candidate)
print("Best Scores (Metric 1, Metric 2):", best_scores)
