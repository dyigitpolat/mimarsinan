from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

import torch.nn as nn


# Base activation types that produce non-negative outputs and are
# compatible with the chip's hardcoded ReLU.  Models using only these
# activations do not need clamp adaptation.
_RELU_COMPATIBLE_TYPES = (
    "LeakyGradReLU",  # forward is pure ReLU; leaky only in backward
    "ReLU",
)


# Activations that become host-side ComputeOps — no adaptation needed.
_HOST_SIDE_TYPES = ("Identity",)


def _needs_clamp_adaptation(model) -> bool:
    """Check whether any chip-targeted perceptron needs clamp adaptation.

    Identity/GELU perceptrons become ComputeOps (host-side) and need no
    adaptation. Only chip-targeted perceptrons with non-ReLU activations
    (unlikely after correct packaging) would need clamping.
    """
    for p in model.get_perceptrons():
        base = p.base_activation
        name = type(base).__name__
        if name in _HOST_SIDE_TYPES:
            continue  # runs on host, no clamping needed
        if name not in _RELU_COMPATIBLE_TYPES:
            return True  # chip-targeted but not ReLU-compatible
    return False


class ClampAdaptationStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager", "activation_scales"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None
        self.tuner = None

    def validate(self):
        if self.tuner is not None:
            return self.tuner.validate()
        return self.pipeline.get_target_metric()

    def process(self):
        model = self.get_entry('model')
        adaptation_manager = self.get_entry("adaptation_manager")

        if not _needs_clamp_adaptation(model):
            # All activations are already ReLU-compatible — just set
            # clamp_rate=1 and apply scales (no tuning needed).
            adaptation_manager.clamp_rate = 1.0
            scales = self.get_entry("activation_scales")
            for idx, perceptron in enumerate(model.get_perceptrons()):
                perceptron.set_activation_scale(scales[idx])
                adaptation_manager.update_activation(self.pipeline.config, perceptron)
            print("[ClampAdaptationStep] All activations are ReLU-compatible — "
                  "skipping tuner, applying scales directly.")
            self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
            self.update_entry("model", model, 'torch_model')
            return

        self.tuner = ClampTuner(
            self.pipeline,
            model = self.get_entry('model'),
            target_accuracy = self.pipeline.get_target_metric(),
            lr = self.pipeline.config['lr'] * 1e-3,
            adaptation_manager = adaptation_manager,
            activation_scales = self.get_entry("activation_scales"))
        self.tuner.run()

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", model, 'torch_model')
        