"""
Legacy compatibility shim: MLP_Mixer_Searcher exposing get_optimized_configuration().

New code should prefer:
  - SearchProblem implementations under mimarsinan.search.problems
  - SearchOptimizer implementations under mimarsinan.search.optimizers
"""

from mimarsinan.search.mlp_mixer_configuration_sampler import MLP_Mixer_ConfigurationSampler
from mimarsinan.search.optimizers.sampler_optimizer import SamplerOptimizer
from mimarsinan.search.problems.evaluator_problem import EvaluatorProblem
from mimarsinan.search.results import ObjectiveSpec


class MLP_Mixer_Searcher:
    def __init__(self, evaluator, nas_workers: int):
        self._nas_workers = int(nas_workers)
        self.configuration_sampler = MLP_Mixer_ConfigurationSampler()
        self.evaluator = evaluator

    def get_optimized_configuration(self, cycles: int = 5, configuration_batch_size: int = 50):
        problem = EvaluatorProblem(
            evaluator=self.evaluator,
            objective=ObjectiveSpec(name="accuracy", goal="max"),
        )
        optimizer = SamplerOptimizer(
            sampler=self.configuration_sampler,
            cycles=int(cycles),
            batch_size=int(configuration_batch_size),
            workers=self._nas_workers,
        )
        return optimizer.optimize(problem).best.configuration