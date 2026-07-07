"""Base for tuners using PerceptronTransformTrainer with stochastic mixing."""

import copy

import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.perceptron_transform_trainer import (
    PerceptronTransformTrainer,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


class PerceptronTransformTuner(SmoothAdaptationTuner):
    # A persisted optimizer would step transform-produced tensors that receive no gradients (GradScaler crash).
    _supports_persistent_optimizer = False

    # Subclasses whose previous-transform is a declared no-op (NAPQ: projection
    # rungs) set True: the per-step mixing then clones parameter TENSORS instead
    # of deep-copying the whole perceptron module (bit-identical — same values,
    # same RNG draws; the deepcopy was 78% of the WQ endpoint's step wall).
    _prev_transform_is_identity = False

    trainer: PerceptronTransformTrainer

    def __init__(self, pipeline, model, target_accuracy, lr):
        self._device = pipeline.config["device"]
        self._data_loader_factory = DataLoaderFactory.for_pipeline(pipeline)
        self._pipeline_loss = pipeline.loss
        super().__init__(pipeline, model, target_accuracy, lr)

    def _create_trainer(self) -> PerceptronTransformTrainer:
        trainer = PerceptronTransformTrainer(
            self.model,
            self._device,
            self._data_loader_factory,
            self._pipeline_loss,
            self._mixed_transform(1.0),
            recipe=self._tuning_recipe(),
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
        """Build the transformation closure for one probe cycle.

        A fresh per-closure ``mask_cache`` freezes the prev/new Bernoulli realisation
        for this rate, so training and evaluation within one probe see the same mask.
        """
        mask_cache: dict = {}
        return lambda perceptron: self._mixed_perceptron_transform(
            perceptron, rate, mask_cache
        )

    def _mix_params(self, prev_param, new_param, rate, cache_key=None, cache=None):
        """Return ``mask * new + (1 - mask) * prev`` with a possibly cached mask.

        With ``cache``, the Bernoulli mask is drawn once per ``cache_key`` and reused,
        keeping the mixture stable across steps within one probe (see :meth:`_mixed_transform`).
        """
        if cache is not None and cache_key is not None:
            mask = cache.get(cache_key)
            if mask is None or tuple(mask.shape) != tuple(prev_param.shape):
                mask = (
                    torch.rand(prev_param.shape, device=prev_param.device) < rate
                ).to(prev_param.dtype)
                cache[cache_key] = mask
        else:
            mask = (
                torch.rand(prev_param.shape, device=prev_param.device) < rate
            ).to(prev_param.dtype)
        return mask * new_param + (1 - mask) * prev_param

    def _mixed_perceptron_transform(self, perceptron, rate, mask_cache=None):
        if self._prev_transform_is_identity:
            prev_named = [
                (name, param.detach().clone())
                for name, param in perceptron.named_parameters()
            ]
            self._get_new_perceptron_transform(rate)(perceptron)
        else:
            temp_prev_perceptron = copy.deepcopy(perceptron).to(self._device)
            self._get_previous_perceptron_transform(rate)(temp_prev_perceptron)
            self._get_new_perceptron_transform(rate)(perceptron)
            prev_named = list(temp_prev_perceptron.named_parameters())

        perceptron_key = id(perceptron)
        for (name, param), (_, prev_param) in zip(
            perceptron.named_parameters(),
            prev_named,
        ):
            cache_key = (
                (perceptron_key, name) if mask_cache is not None else None
            )
            if len(param.shape) == 0:
                param.data.copy_(
                    self._mix_params(
                        prev_param.data,
                        param.data.clone().detach(),
                        rate,
                        cache_key=cache_key,
                        cache=mask_cache,
                    )
                )
            else:
                param.data.copy_(
                    self._mix_params(
                        prev_param.data[:],
                        param.data[:].clone().detach(),
                        rate,
                        cache_key=cache_key,
                        cache=mask_cache,
                    )
                )

    def _after_run(self):
        self._continue_to_full_rate()

        self.trainer.perceptron_transformation = self._mixed_transform(1.0)
        with torch.no_grad():
            self.trainer._update_and_transform_model()

        final_acc = self._ensure_pipeline_threshold()
        self._committed_rate = 1.0
        return final_acc
