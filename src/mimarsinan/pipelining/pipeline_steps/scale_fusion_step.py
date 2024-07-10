from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.layers import TransformedActivation, ClampDecorator, QuantizeDecorator, ScaleDecorator

from mimarsinan.visualization.activation_function_visualization import ActivationFunctionVisualizer


import torch.nn as nn
import torch

class ScaleFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["model", "activation_scales", "input_activation_scales", "output_activation_scales", "adaptation_manager"]
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
        
        in_scales = self.get_entry('input_activation_scales')
        scales = self.get_entry('activation_scales')
        out_scales = self.get_entry('output_activation_scales')

        # model.in_act = TransformedActivation(
        #     base_activation = nn.Identity(),
        #     decorators = [
        #         ScaleDecorator(torch.tensor(1.0/scales[4])),
        #         ClampDecorator(torch.tensor(0.0), torch.tensor(1.0/scales[4])),
        #         #QuantizeDecorator(torch.tensor(self.pipeline.config['target_tq']), torch.tensor(1.0/scales[4])),
        #     ])
        
        print("??")
        print(self.validate())

        for idx, perceptron in enumerate(model.get_perceptrons()):
            in_scale = in_scales[idx] * 1.0
            #scale = perceptron.activation_scale * 1.0
            scale = out_scales[idx] * 1.0

            perceptron.set_scale_factor(1.0)
            perceptron.set_activation_scale(1.0)
            adaptation_manager.update_activation(self.pipeline.config, perceptron)

            inp = torch.randn(perceptron.layer.in_features).to(self.pipeline.config['device']).abs()
            perceptron.layer.eval()
            #old = perceptron.layer(inp).mean()

            # PerceptronTransformer().apply_effective_bias_transform(perceptron, lambda p: (p / scale))
            # PerceptronTransformer().apply_effective_weight_transform(perceptron, lambda p: (p * in_scale / scale))
            perceptron.layer.weight.data *= in_scale / scale
            if perceptron.layer.bias is not None:
                perceptron.layer.bias.data /= scale

            perceptron.layer.eval()
            #new = perceptron.layer(inp / in_scale).mean()

            if not isinstance(perceptron.normalization, nn.Identity):
                perceptron.eval()
                perceptron.normalization.eval()
                self._adjust_normalization_stats(perceptron, in_scale / scale)
                perceptron.normalization.eval()
                perceptron.eval()

        # self.trainer.train_n_epochs(
        #     self.pipeline.config['lr'] * 0.0, 
        #     self.pipeline.config['tuner_epochs'],
        #     warmup_epochs=0)
        
        # self.trainer.train_until_target_accuracy(
        #     self.pipeline.config['lr'] * 0.1, 
        #     self.pipeline.config['tuner_epochs'],
        #     target_accuracy=self.pipeline.get_target_metric(),
        #     warmup_epochs=0)


        print("?")
        print(self.validate())

        self.update_entry("adaptation_manager", adaptation_manager, 'pickle')
        self.update_entry("model", model, 'torch_model')