"""Base for tuners using PerceptronTransformTrainer with stochastic mixing.

Extends SmoothAdaptationTuner, overriding _create_trainer to use
PerceptronTransformTrainer. Adds stochastic parameter mixing between
"previous" and "new" perceptron transforms.
"""

import copy

import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.perceptron_transform_trainer import (
    PerceptronTransformTrainer,
)
from mimarsinan.tuning.unified_tuner import SmoothAdaptationTuner


class PerceptronTransformTuner(SmoothAdaptationTuner):
    def __init__(self, pipeline, model, target_accuracy, lr):
        self._device = pipeline.config["device"]
        self._data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        self._pipeline_loss = pipeline.loss
        super().__init__(pipeline, model, target_accuracy, lr)

    def _create_trainer(self):
        trainer = PerceptronTransformTrainer(
            self.model,
            self._device,
            self._data_loader_factory,
            self._pipeline_loss,
            self._mixed_transform(1.0),
        )
        return trainer

    def _update_and_evaluate(self, rate):
        self.trainer.perceptron_transformation = self._mixed_transform(rate)
        raise NotImplementedError

    def _get_previous_perceptron_transform(self, rate):
        raise NotImplementedError

    def _get_new_perceptron_transform(self, rate):
        raise NotImplementedError

    def _mixed_transform(self, rate):
        return lambda perceptron: self._mixed_perceptron_transform(perceptron, rate)

    def _mix_params(self, prev_param, new_param, rate):
        random_mask = torch.rand(prev_param.shape, device=prev_param.device)
        random_mask = (random_mask < rate).float()
        return random_mask * new_param + (1 - random_mask) * prev_param

    def _mixed_perceptron_transform(self, perceptron, rate):
        temp_prev_perceptron = copy.deepcopy(perceptron).to(self._device)

        self._get_previous_perceptron_transform(rate)(temp_prev_perceptron)
        self._get_new_perceptron_transform(rate)(perceptron)

        for param, prev_param in zip(
            perceptron.parameters(), temp_prev_perceptron.parameters()
        ):
            if len(param.shape) == 0:
                param.data.copy_(
                    self._mix_params(
                        prev_param.data, param.data.clone().detach(), rate
                    )
                )
            else:
                param.data.copy_(
                    self._mix_params(
                        prev_param.data[:], param.data[:].clone().detach(), rate
                    )
                )

    def _adaptation(self, rate):
        self.pipeline.reporter.report(self.name, rate)
        self.pipeline.reporter.report("Adaptation target", self._get_target())
        self.trainer.perceptron_transformation = self._mixed_transform(rate)

        self._update_and_evaluate(rate)

        lr = self._find_lr()
        self.trainer.train_steps_until_target(
            lr,
            self._budget.max_training_steps,
            self._get_target(),
            0,
            validation_n_batches=self._budget.validation_steps,
            check_interval=self._budget.check_interval,
            patience=3,
        )

        acc = self.trainer.validate_n_batches(self._budget.eval_n_batches)
        self.target_adjuster.update_target(acc)
