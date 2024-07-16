from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer

from mimarsinan.models.layers import SavedTensorDecorator, TransformedActivation

from mimarsinan.tuning.tuners.scale_tuner import ScaleTuner

import torch.nn as nn
import torch

class ScaleFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "adaptation_manager"]
        promises = []
        updates = ["model", "adaptation_manager"]
        clears = []
        super().__init__(requires, promises, updates, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()
    
    def _adjust_normalization_stats(self, perceptron, scale):
        bn = perceptron.normalization
        
        # Adjust running mean
        bn.running_mean.data[:] *= scale
        
        # Adjust running variance
        bn.running_var.data[:] *= scale**2

    def process(self):
        model = self.get_entry("model")
        adaptation_manager = self.get_entry("adaptation_manager")

        # Trainer
        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report
        
        print("??")
        print(self.validate())

        in_scales = [1.0]
        scales = []
        for g_idx, perceptron_group in enumerate(model.perceptron_flow.get_perceptron_groups()):
            total_scale = 0.0
            for perceptron in perceptron_group:
                total_scale += perceptron.activation_scale.item()

            s = total_scale / len(perceptron_group)
            in_scales.append(s)
            scales.append(s)

        print(in_scales)
        print(scales)
        
        for g_idx, perceptron_group in enumerate(model.perceptron_flow.get_perceptron_groups()):
            for perceptron in perceptron_group:
                scale = self.out_scales[g_idx]
                in_scale = self.in_scales[g_idx]

                perceptron.set_scale_factor(1.0)
                perceptron.set_activation_scale(1.0)

                adaptation_manager.update_activation(self.pipeline.config, perceptron)

                PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda p: (p / scale))
                PerceptronTransformer().apply_effective_weight_transform(perceptron, lambda p: (p * in_scale / scale))

        print("?")
        print(self.validate())

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", model, 'torch_model')