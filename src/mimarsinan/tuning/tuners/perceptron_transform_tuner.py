"""Base for tuners using PerceptronTransformTrainer.

Phase C1 cleanup
----------------
Historically this class implemented "rate-based stochastic mixing":
each perceptron's parameters were sampled element-wise between the
previous (FP) and new (quantized) tensors with probability ``rate``.
That schedule existed because the legacy quantizer was non-differentiable
and there was no way to interpolate via gradients.

With the LSQ + STE quantizer in place, the quantization is fully
differentiable and the rate parameter no longer has any quantitative
meaning -- the only sensible value is ``rate = 1.0`` (apply the
quantization).  ``_mixed_perceptron_transform`` therefore now simply
runs the "new" transform without any random masking.  The ``rate``
argument is still threaded through for API compatibility with the
``SmoothAdaptationTuner`` adaptation loop (which still varies the
schedule for its own bookkeeping) but is treated as an indicator of
whether the transform should be applied at all (any positive rate
applies it; ``rate == 0`` skips it).

The aux/main shadow-copy machinery in ``PerceptronTransformTrainer`` is
retained because it serves a different purpose: it keeps the FP master
copy that the optimizer steps separate from the (possibly transformed)
forward-pass copy.  That is still needed when ``apply_effective_*``
writes hard-quantized values back into ``layer.weight.data``.
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
        return lambda perceptron: self._mixed_perceptron_transform(perceptron, rate)

    def _mixed_perceptron_transform(self, perceptron, rate):
        # Phase C1: no more rate-based stochastic mixing.  LSQ + STE
        # already provides a smooth, fully differentiable transition
        # between FP and quantized weights via the learnable step size,
        # so any positive ``rate`` simply means "apply the new transform".
        # ``rate <= 0`` skips the transform entirely (used by tests and
        # by the early stages of the SmoothAdaptationTuner loop where
        # the transform should be a no-op).
        if float(rate) <= 0.0:
            return
        self._get_new_perceptron_transform(rate)(perceptron)

    def _after_run(self):
        self._continue_to_full_rate()

        self.trainer.perceptron_transformation = self._mixed_transform(1.0)
        with torch.no_grad():
            self.trainer._update_and_transform_model()

        recovered_val = self._attempt_recovery_if_below_floor()
        if recovered_val >= self._validation_floor_for_commit():
            self._committed_rate = 1.0
        self._final_metric = recovered_val
        self._flush_enforcement_hooks()
        return recovered_val
