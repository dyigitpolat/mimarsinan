from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.mapping_utils import (
    Conv2DPerceptronMapper,
    Ensure2DMapper,
    EinopsRearrangeMapper,
    InputMapper,
    Mapper,
    MaxPool2DMapper,
    ModelRepresentation,
    ModuleMapper,
    PerceptronMapper,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow


def _to_2tuple(x):
    if isinstance(x, tuple):
        return (int(x[0]), int(x[1]))
    v = int(x)
    return (v, v)


class SimpleConvMapper(PerceptronFlow):
    """
    Simple convolutional model:
      Input -> Conv2d -> Conv2d -> (optional Pool) -> Flatten -> FC -> FC

    Notes:
    - Two convolution layers for feature extraction
    - Two FC layers for classification
    - Optional pooling layer (creates ComputeOp in IR)
    """

    def __init__(
        self,
        device,
        input_shape,
        num_classes: int,
        *,
        conv_out_channels: int = 3,
        conv_kernel_size: int | tuple[int, int] = 3,
        conv_stride: int | tuple[int, int] = 4,
        conv_padding: int | tuple[int, int] = 1,
        conv_dilation: int | tuple[int, int] = 1,
        conv_bias: bool = True,
        # Second conv layer parameters (uses same kernel but stride=1 by default)
        conv2_out_channels: int | None = None,  # defaults to conv_out_channels
        conv2_kernel_size: int | tuple[int, int] = 3,
        conv2_stride: int | tuple[int, int] = 1,
        conv2_padding: int | tuple[int, int] = 1,
        conv2_dilation: int | tuple[int, int] = 1,
        use_pool: bool = False,
        pool_kernel_size: int | tuple[int, int] = 2,
        pool_stride: int | tuple[int, int] = 2,
        pool_padding: int | tuple[int, int] = 0,
        use_batchnorm: bool = True,
        # Hidden FC layer size (defaults to fc_in_features)
        fc_hidden_features: int | None = None,
        max_axons: int | None = None,
        max_neurons: int | None = None,
        name: str = "simple_conv",
    ):
        super().__init__(device)

        self.name = str(name)
        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)

        if len(self.input_shape) != 3:
            raise ValueError(f"{self.name}: expected input_shape=(C,H,W), got {self.input_shape}")

        c_in, h_in, w_in = (int(self.input_shape[0]), int(self.input_shape[1]), int(self.input_shape[2]))
        
        # First conv layer params
        k_h, k_w = _to_2tuple(conv_kernel_size)
        s_h, s_w = _to_2tuple(conv_stride)
        p_h, p_w = _to_2tuple(conv_padding)
        d_h, d_w = _to_2tuple(conv_dilation)
        conv_out_channels = int(conv_out_channels)

        # Output spatial dims after first conv
        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
        if h_out <= 0 or w_out <= 0:
            raise ValueError(
                f"{self.name}: conv1 output spatial dims invalid: (h_out,w_out)=({h_out},{w_out}). "
                f"Check kernel/stride/padding/dilation."
            )

        # Second conv layer params
        conv2_out_channels = int(conv2_out_channels) if conv2_out_channels is not None else conv_out_channels
        k2_h, k2_w = _to_2tuple(conv2_kernel_size)
        s2_h, s2_w = _to_2tuple(conv2_stride)
        p2_h, p2_w = _to_2tuple(conv2_padding)
        d2_h, d2_w = _to_2tuple(conv2_dilation)

        # Output spatial dims after second conv
        h_out2 = (h_out + 2 * p2_h - d2_h * (k2_h - 1) - 1) // s2_h + 1
        w_out2 = (w_out + 2 * p2_w - d2_w * (k2_w - 1) - 1) // s2_w + 1
        if h_out2 <= 0 or w_out2 <= 0:
            raise ValueError(
                f"{self.name}: conv2 output spatial dims invalid: (h_out,w_out)=({h_out2},{w_out2}). "
                f"Check conv2 kernel/stride/padding/dilation."
            )

        use_pool = bool(use_pool)
        pk_h, pk_w = _to_2tuple(pool_kernel_size)
        ps_h, ps_w = _to_2tuple(pool_stride)
        pp_h, pp_w = _to_2tuple(pool_padding)

        # Save pre-pool spatial dims for pooling mapper
        h_before_pool, w_before_pool = h_out2, w_out2

        if use_pool:
            # MaxPool2d output dims (dilation is assumed 1 for simplicity)
            h_out2 = (h_out2 + 2 * pp_h - (pk_h - 1) - 1) // ps_h + 1
            w_out2 = (w_out2 + 2 * pp_w - (pk_w - 1) - 1) // ps_w + 1
            if h_out2 <= 0 or w_out2 <= 0:
                raise ValueError(
                    f"{self.name}: pool output spatial dims invalid: (h_out,w_out)=({h_out2},{w_out2}). "
                    f"Check pool kernel/stride/padding."
                )

        fc_in_features = int(conv2_out_channels * h_out2 * w_out2)
        fc_hidden = int(fc_hidden_features) if fc_hidden_features is not None else fc_in_features

        self.input_activation = nn.Identity()

        # Register parameter-owning nodes so they appear in state_dict traversal.
        self.feature_mappers = nn.ModuleList()
        self.classifier_perceptrons = nn.ModuleList()

        out: Mapper = InputMapper(self.input_shape)
        self._input_activation_mapper = ModuleMapper(out, self.input_activation)
        out = self._input_activation_mapper

        # First conv layer
        conv1 = Conv2DPerceptronMapper(
            out,
            in_channels=c_in,
            out_channels=conv_out_channels,
            kernel_size=(k_h, k_w),
            stride=(s_h, s_w),
            padding=(p_h, p_w),
            dilation=(d_h, d_w),
            bias=bool(conv_bias),
            max_neurons=max_neurons,
            max_axons=max_axons,
            use_batchnorm=bool(use_batchnorm),
            name=f"{self.name}_conv1",
        )
        self.feature_mappers.append(conv1)
        out = conv1

        # Second conv layer
        conv2 = Conv2DPerceptronMapper(
            out,
            in_channels=conv_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=(k2_h, k2_w),
            stride=(s2_h, s2_w),
            padding=(p2_h, p2_w),
            dilation=(d2_h, d2_w),
            bias=bool(conv_bias),
            max_neurons=max_neurons,
            max_axons=max_axons,
            use_batchnorm=bool(use_batchnorm),
            name=f"{self.name}_conv2",
        )
        self.feature_mappers.append(conv2)
        out = conv2

        if use_pool:
            # Use the new MaxPool2DMapper that supports both old and IR mapping
            pool_mapper = MaxPool2DMapper(
                out,
                kernel_size=(pk_h, pk_w),
                stride=(ps_h, ps_w),
                padding=(pp_h, pp_w),
                input_spatial_shape=(h_before_pool, w_before_pool),
                input_channels=conv2_out_channels,
                name=f"{self.name}_pool",
            )
            self.feature_mappers.append(pool_mapper)
            out = pool_mapper

        out = EinopsRearrangeMapper(out, "... c h w -> ... (c h w)")
        out = Ensure2DMapper(out)

        # First FC layer (hidden)
        # Perceptron args: (output_channels, input_features, ...)
        fc1 = Perceptron(fc_hidden, fc_in_features, normalization=nn.Identity(), name=f"{self.name}_fc1")
        self.classifier_perceptrons.append(fc1)
        out = PerceptronMapper(out, fc1)

        # Second FC layer (output)
        fc2 = Perceptron(num_classes, fc_hidden, normalization=nn.Identity(), name=f"{self.name}_fc2")
        self.classifier_perceptrons.append(fc2)
        out = PerceptronMapper(out, fc2)

        self._mapper_repr = ModelRepresentation(out)

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        self._input_activation_mapper.module = activation

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


