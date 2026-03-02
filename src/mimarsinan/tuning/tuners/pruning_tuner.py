"""PruningTuner: gradually prunes perceptron weight rows and columns.

Extends PerceptronTuner to use SmartSmoothAdaptation for progressive
weight pruning. At each adaptation step, the tuner computes significance
masks and applies rate-adaptive scaling to the least significant rows
and columns.
"""

from mimarsinan.tuning.tuners.perceptron_tuner import PerceptronTuner
from mimarsinan.transformations.pruning import compute_pruning_masks, apply_pruning_masks


class PruningTuner(PerceptronTuner):
    def __init__(
        self,
        pipeline,
        model,
        target_accuracy,
        lr,
        adaptation_manager,
        pruning_fraction,
    ):
        super().__init__(pipeline, model, target_accuracy, lr)

        self.adaptation_manager = adaptation_manager
        self.pruning_fraction = pruning_fraction

    def _get_target_decay(self):
        return 0.99

    def _update_and_evaluate(self, rate):
        self.adaptation_manager.pruning_rate = rate

        for perceptron in self.model.get_perceptrons():
            row_mask, col_mask = compute_pruning_masks(
                perceptron, self.pruning_fraction
            )
            apply_pruning_masks(perceptron, row_mask, col_mask, rate)

        self.trainer.train_one_step(0)
        return self.trainer.validate()

    def run(self):
        self.trainer.validate()
        super().run()
        return self.trainer.validate()
