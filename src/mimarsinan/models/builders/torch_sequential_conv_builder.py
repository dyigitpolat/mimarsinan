"""Builder that produces a sequential convnet PyTorch model with two segments (conv + compute op + FC)."""

import torch
import torch.nn as nn

from mimarsinan.pipelining.model_registry import ModelRegistry


def _conv_output_size(size: int, kernel_size: int, stride: int, padding: int) -> int:
    """Output size after Conv2d/MaxPool2d: (size + 2*padding - kernel_size) // stride + 1."""
    return (size + 2 * padding - kernel_size) // stride + 1


@ModelRegistry.register("torch_sequential_conv", label="Torch Seq. Conv", category="torch")
class TorchSequentialConvBuilder:
    """Builds a plain nn.Module: Sequential(Conv2d, ReLU, MaxPool2d, Flatten, Linear, ReLU, ..., Linear).

    The model has a single conv block, one non-neural compute op (MaxPool2d), then a stack of
    linear layers. After conversion to IR this yields two neural segments separated by one
    ComputeOp. Compatible with TorchMappingStep (torch 2 repr flow).

    Configuration must provide:
    - "conv_out_channels": int
    - "hidden_dims": non-empty list of int (FC hidden layer sizes; last layer is num_classes)

    Optional: conv_kernel_size, conv_stride, conv_padding, pool_kernel_size, pool_stride, pool_padding.
    """

    def __init__(
        self,
        device,
        input_shape,
        num_classes,
        pipeline_config,
    ):
        self.device = device
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.pipeline_config = pipeline_config

    def build(self, configuration):
        if "conv_out_channels" not in configuration:
            raise ValueError(
                "TorchSequentialConvBuilder requires configuration['conv_out_channels'] (int)."
            )
        if "hidden_dims" not in configuration:
            raise ValueError(
                "TorchSequentialConvBuilder requires configuration['hidden_dims'] "
                "(a non-empty list of hidden layer sizes)."
            )
        conv_out_channels = int(configuration["conv_out_channels"])
        hidden_dims = configuration["hidden_dims"]
        if not isinstance(hidden_dims, (list, tuple)) or len(hidden_dims) == 0:
            raise ValueError(
                "TorchSequentialConvBuilder requires configuration['hidden_dims'] "
                "to be a non-empty list of hidden layer sizes."
            )

        # Conv/pool params with defaults
        conv_kernel_size = configuration.get("conv_kernel_size", 3)
        conv_stride = configuration.get("conv_stride", 1)
        conv_padding = configuration.get("conv_padding", 1)
        pool_kernel_size = configuration.get("pool_kernel_size", 2)
        pool_stride = configuration.get("pool_stride", 2)
        pool_padding = configuration.get("pool_padding", 0)

        if isinstance(conv_kernel_size, (list, tuple)):
            conv_kernel_size = conv_kernel_size[0]
        if isinstance(conv_stride, (list, tuple)):
            conv_stride = conv_stride[0]
        if isinstance(conv_padding, (list, tuple)):
            conv_padding = conv_padding[0]
        if isinstance(pool_kernel_size, (list, tuple)):
            pool_kernel_size = pool_kernel_size[0]
        if isinstance(pool_stride, (list, tuple)):
            pool_stride = pool_stride[0]
        if isinstance(pool_padding, (list, tuple)):
            pool_padding = pool_padding[0]

        conv_kernel_size = int(conv_kernel_size)
        conv_stride = int(conv_stride)
        conv_padding = int(conv_padding)
        pool_kernel_size = int(pool_kernel_size)
        pool_stride = int(pool_stride)
        pool_padding = int(pool_padding)

        # input_shape is (C, H, W)
        shape = tuple(self.input_shape)
        if len(shape) != 3:
            raise ValueError(
                f"TorchSequentialConvBuilder expects input_shape (C, H, W), got {shape}"
            )
        in_channels, h_in, w_in = shape[0], shape[1], shape[2]

        h_conv = _conv_output_size(h_in, conv_kernel_size, conv_stride, conv_padding)
        w_conv = _conv_output_size(w_in, conv_kernel_size, conv_stride, conv_padding)
        h_pool = _conv_output_size(h_conv, pool_kernel_size, pool_stride, pool_padding)
        w_pool = _conv_output_size(w_conv, pool_kernel_size, pool_stride, pool_padding)

        if h_pool <= 0 or w_pool <= 0:
            raise ValueError(
                f"TorchSequentialConvBuilder: conv+pool output spatial size would be "
                f"({h_pool}, {w_pool}); check input_shape and conv/pool params."
            )
        flat_size = conv_out_channels * h_pool * w_pool

        dims = [flat_size] + list(hidden_dims) + [self.num_classes]
        layers = []

        layers.append(
            nn.Conv2d(
                in_channels,
                conv_out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.MaxPool2d(
                kernel_size=pool_kernel_size,
                stride=pool_stride,
                padding=pool_padding,
            )
        )
        layers.append(nn.Flatten())
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    @classmethod
    def get_config_schema(cls):
        return [
            {"key": "conv_out_channels", "type": "number", "label": "Conv Channels", "default": 16},
            {"key": "hidden_dims", "type": "text", "label": "FC Hidden Dims", "default": "128, 64"},
            {"key": "conv_kernel_size", "type": "number", "label": "Kernel Size", "default": 3},
            {"key": "pool_kernel_size", "type": "number", "label": "Pool Kernel", "default": 2},
        ]
