from mimarsinan.pipelining.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.activation_utils import needs_clamp_adaptation

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

import torch.nn as nn


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

        if not needs_clamp_adaptation(model):
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
        