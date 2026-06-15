"""Base for tuners using PerceptronTransformTrainer with stochastic mixing.

Extends SmoothAdaptationTuner, overriding _create_trainer to use
PerceptronTransformTrainer. Adds stochastic parameter mixing between
"previous" and "new" perceptron transforms. Every perceptron sees the scalar
``rate`` delivered by the orchestration loop (uniform application).
"""

import copy

import torch

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.perceptron_transform_trainer import (
    PerceptronTransformTrainer,
)
from mimarsinan.tuning.orchestration.smooth_adaptation_tuner import SmoothAdaptationTuner


class PerceptronTransformTuner(SmoothAdaptationTuner):
    # The forward applies a per-step parameter transform, so a persisted
    # optimizer would step tensors that receive no gradients (GradScaler crash).
    _supports_persistent_optimizer = False

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

        A fresh per-closure ``mask_cache`` is created here and captured by
        the returned callable. Every invocation of the callable
        (``PerceptronTransformTrainer`` calls it once per training step
        and once again on each validation after ``_update_and_evaluate``)
        will draw masks lazily on first use and reuse them thereafter,
        so training and evaluation within a single probe see a
        **deterministic, frozen** stochastic realisation of the
        prev/new mix. When the orchestration loop probes a new rate it
        assigns a fresh closure (with a fresh cache) to
        ``trainer.perceptron_transformation``, so the mask is regenerated
        per probe -- preserving the intended stochastic-regularisation
        behaviour across probes while eliminating the moving-target loss
        surface that the legacy per-step redraw produced.
        """
        mask_cache: dict = {}
        return lambda perceptron: self._mixed_perceptron_transform(
            perceptron, rate, mask_cache
        )

    def _mix_params(self, prev_param, new_param, rate, cache_key=None, cache=None):
        """Return ``mask * new + (1 - mask) * prev`` with a possibly cached mask.

        When ``cache`` is provided, the Bernoulli mask at probability
        ``rate`` is drawn once per ``cache_key`` and then reused across
        every subsequent call with the same key. This makes the
        prev/new mixture stable across training steps and validation
        within one probe cycle; see :meth:`_mixed_transform`.
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
        temp_prev_perceptron = copy.deepcopy(perceptron).to(self._device)

        self._get_previous_perceptron_transform(rate)(temp_prev_perceptron)
        self._get_new_perceptron_transform(rate)(perceptron)

        perceptron_key = id(perceptron)
        for (name, param), (_, prev_param) in zip(
            perceptron.named_parameters(),
            temp_prev_perceptron.named_parameters(),
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
