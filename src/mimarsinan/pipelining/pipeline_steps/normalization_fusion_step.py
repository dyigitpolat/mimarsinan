from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.mapping.mapping_utils import *

import torch.nn as nn

class NormalizationFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["aq_model"]
        promises = ["nf_model"]
        clears = ["aq_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.pipeline.cache["aq_model"]

        # Trainer
        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            self.pipeline.data_provider,
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        for perceptron in model.get_perceptrons():
            self._fuse_normalization(perceptron)

        print(self.validate())

        self.trainer.train_until_target_accuracy(
            LearningRateExplorer(
                self.trainer,
                model,
                self.pipeline.config['lr'] / 2,
                self.pipeline.config['lr'] / 1000,
                0.01
            ).find_lr_for_tuning(),
            self.pipeline.config['tuner_epochs'],
            self.pipeline.get_target_metric(),
        )

        self.pipeline.cache.add("nf_model", model, 'torch_model')
        self.pipeline.cache.remove("aq_model")

    def _fuse_normalization(self, perceptron):
        if isinstance(perceptron.normalization, nn.Identity):
            return

        assert isinstance(perceptron.normalization, nn.BatchNorm1d)
        assert perceptron.normalization.affine

        w, b = get_fused_weights(
            linear_layer=perceptron.layer, bn_layer=perceptron.normalization)
        
        scaled_activation = perceptron.activation
        assert isinstance(scaled_activation, ScaleActivation)
        scale = scaled_activation.scale

        w = w / (scale ** 0.5)
        b = b / (scale ** 0.5)

        perceptron.layer = nn.Linear(
            perceptron.input_features, 
            perceptron.output_channels, bias=True)
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b

        perceptron.normalization = nn.Identity()