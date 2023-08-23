from mimarsinan.pipelining.pipeline_step import PipelineStep

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory

import torch.nn as nn
import torch

class PerceptronFusionStep(PipelineStep):
    def __init__(self, pipeline):
        requires = ["aq_model"]
        promises = ["pf_model"]
        clears = ["aq_model"]
        super().__init__(requires, promises, clears, pipeline)

        self.trainer = None

    def validate(self):
        return self.trainer.validate()

    def process(self):
        model = self.get_entry("aq_model")
        
        for perceptron in model.get_perceptrons():
            if perceptron.layer.bias is not None:
                perceptron.layer = self.fuse_linear_layer_bias(perceptron.layer)

        # Trainer
        self.trainer = BasicTrainer(
            model, 
            self.pipeline.config['device'], 
            DataLoaderFactory(self.pipeline.data_provider_factory),
            self.pipeline.loss)
        self.trainer.report_function = self.pipeline.reporter.report

        self.add_entry("pf_model", model, 'torch_model')
        
    def fuse_linear_layer_bias(self, layer):
        assert isinstance(layer, nn.Linear), 'Input layer must be an instance of nn.Linear'
    
        # Get the weights and bias from the existing layer
        weights = layer.weight.data
        bias = layer.bias.data.unsqueeze(1)
        
        # Append the bias to the weights
        new_weights = torch.cat((weights, bias), dim=1)
        
        # Create a new layer with the new weights
        out_features, in_features = new_weights.shape
        fused_layer = FusedLinear(in_features - 1, out_features).to(self.pipeline.config['device'])
        fused_layer.linear.weight.data = new_weights

        return fused_layer


class FusedLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(FusedLinear, self).__init__()
        self.linear = nn.Linear(input_features + 1, output_features, bias=False)
        self.weight = self.linear.weight
        self.bias = self.linear.bias

    def forward(self, x):
        # Add an extra dimension for 2D inputs (i.e., make it 3D)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        bias_feature = torch.ones((batch_size, seq_len, 1), device=x.device)
        x = torch.cat([x, bias_feature], dim=-1)
        output = self.linear(x)

        # If the original input was 2D, remove the added dimension
        if len(output.shape) == 3 and output.shape[1] == 1:
            output = output.squeeze(1)
        return output