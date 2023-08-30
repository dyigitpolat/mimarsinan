from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.tuning.learning_rate_explorer import LearningRateExplorer
from mimarsinan.mapping.mapping_utils import *

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.pipelining.pipeline_steps.perceptron_fusion_step import FusedLinear

import torch.nn as nn

class NormalizationFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model"]
        promises = []
        updates = ["model"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("model")

        # Trainer
        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        for perceptron in model.get_perceptrons():
            self._fuse_normalization(perceptron)

        print(self.validate())

        self.update_entry("model", model, 'torch_model')

    def bring_back_bias(self, fused_linear_layer):
        assert isinstance(fused_linear_layer, FusedLinear), 'Input layer must be an instance of LinearWithoutBias'
        
        # Get the weights from the existing layer
        weights = fused_linear_layer.linear.weight.data
        
        # Split the weights back into the main weights and the bias
        main_weights, bias = weights[:, :-1], weights[:, -1]

        # Create a new layer with the main weights and bias
        out_features, in_features = main_weights.shape
        new_layer = nn.Linear(in_features, out_features)
        new_layer.weight.data = main_weights
        new_layer.bias.data = bias

        return new_layer

    def _fuse_normalization(self, perceptron):
        if isinstance(perceptron.layer, FusedLinear):
            perceptron.layer = self.bring_back_bias(perceptron.layer)

        if isinstance(perceptron.normalization, nn.Identity):
            return

        assert isinstance(perceptron.normalization, nn.BatchNorm1d)
        assert perceptron.normalization.affine

        perceptron.to(self.pipeline.config['device'])
        w, b = get_fused_weights(
            linear_layer=perceptron.layer, bn_layer=perceptron.normalization)
        
        # scaled_activation = perceptron.activation
        # assert isinstance(scaled_activation, ScaleActivation)
        # scale = scaled_activation.scale

        # w = w / (scale ** 0.5)
        # b = b / (scale ** 0.5)

        perceptron.layer = nn.Linear(
            perceptron.input_features, 
            perceptron.output_channels, bias=True)
        
        perceptron.layer.weight.data = w
        perceptron.layer.bias.data = b

        perceptron.normalization = nn.Identity()