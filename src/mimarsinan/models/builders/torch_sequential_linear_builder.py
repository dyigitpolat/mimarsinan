"""Builder that produces a sequential n-linear-layer PyTorch model (torch repr flow)."""

import torch
import torch.nn as nn

from mimarsinan.pipelining.model_registry import ModelRegistry


@ModelRegistry.register("torch_sequential_linear", label="Torch Seq. Linear", category="torch")
class TorchSequentialLinearBuilder:
    """Builds a plain nn.Module: Sequential(Flatten, Linear, ReLU, ..., Linear).

    The model is a stack of linear layers with ReLU between them; the last
    layer is the logits (no activation). Compatible with TorchMappingStep
    (torch 2 repr flow). Configuration must provide "hidden_dims": list of
    int (hidden layer sizes). Input size and num_classes come from pipeline.
    """

    def __init__(
        self,
        device,
        input_shape,
        num_classes,
        max_axons,
        max_neurons,
        pipeline_config,
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        if "hidden_dims" not in configuration:
            raise ValueError(
                "TorchSequentialLinearBuilder requires configuration['hidden_dims'] "
                "(a non-empty list of hidden layer sizes)."
            )
        hidden_dims = configuration["hidden_dims"]
        if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
            raise ValueError(
                "TorchSequentialLinearBuilder requires configuration['hidden_dims'] "
                "to be a non-empty list of hidden layer sizes."
            )
        input_size = int(torch.Size(self.input_shape).numel())
        dims = [input_size] + list(hidden_dims) + [self.num_classes]
        layers = []
        # Flatten
        layers.append(nn.Flatten())
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "hidden_dims", "type": "text", "label": "Hidden Dims (comma-sep)", "default": "512, 256, 128"},
        ]
