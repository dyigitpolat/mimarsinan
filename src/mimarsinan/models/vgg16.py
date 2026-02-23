from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.mapping_utils import (
    Conv2DPerceptronMapper,
    Ensure2DMapper,
    EinopsRearrangeMapper,
    InputMapper,
    Mapper,
    ModelRepresentation,
    ModuleMapper,
    PerceptronMapper,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow


VGG16_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]


class MaxPool2DOpMapper(Mapper):
    """
    Non-neural op placeholder.
    - Forward: real nn.MaxPool2d
    - Mapping: NOT supported yet (requires hybrid execution with sync + spike regeneration)
    """

    def __init__(self, source_mapper: Mapper, pool: nn.MaxPool2d, name: str = "MaxPool2d"):
        super().__init__(source_mapper)
        self.pool = pool
        self.name = str(name)

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: pooling is not supported in SoftCoreMapping yet. "
            f"This requires interleaving non-neural ops with perceptron cores (sync points + spike regeneration)."
        )

    def _forward_impl(self, x):
        return self.pool(x)


class AdaptiveAvgPool2DOpMapper(Mapper):
    """
    Non-neural op placeholder.
    - Forward: real nn.AdaptiveAvgPool2d
    - Mapping: NOT supported yet (requires hybrid execution with sync + spike regeneration)
    """

    def __init__(
        self, source_mapper: Mapper, pool: nn.AdaptiveAvgPool2d, name: str = "AdaptiveAvgPool2d"
    ):
        super().__init__(source_mapper)
        self.pool = pool
        self.name = str(name)

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: pooling is not supported in SoftCoreMapping yet. "
            f"This requires interleaving non-neural ops with perceptron cores (sync points + spike regeneration)."
        )

    def _forward_impl(self, x):
        return self.pool(x)


class VGG16Mapper(PerceptronFlow):
    """
    Crossbar-first VGG16:
    - Conv layers are shared-weight perceptron mappers (im2col + Perceptron).
    - FC layers are Perceptrons (via PerceptronMapper).
    - Pooling ops are explicit non-neural placeholders and will fail mapping for now.
    """

    def __init__(
        self,
        device,
        input_shape,
        num_classes: int,
        *,
        max_axons: int | None = None,
        max_neurons: int | None = None,
    ):
        super().__init__(device)

        self.input_shape = tuple(input_shape)
        self.num_classes = int(num_classes)

        self.input_activation = nn.Identity()

        # IMPORTANT: Mapper graph nodes that own parameters (conv mappers) must be registered
        # under this nn.Module, because Mapper source edges are intentionally hidden from
        # PyTorch's submodule traversal.
        self.feature_mappers = nn.ModuleList()
        self.feature_ops = nn.ModuleList()  # pooling ops (no params), kept for state_dict parity
        self.classifier_perceptrons = nn.ModuleList()

        out: Mapper = InputMapper(self.input_shape)
        self._input_activation_mapper = ModuleMapper(out, self.input_activation)
        out = self._input_activation_mapper

        in_c = int(self.input_shape[-3])
        conv_idx = 0
        for v in VGG16_CFG:
            if v == "M":
                pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.feature_ops.append(pool)
                out = MaxPool2DOpMapper(out, pool, name=f"vgg_pool_{len(self.feature_ops)-1}")
                continue

            conv = Conv2DPerceptronMapper(
                out,
                in_channels=in_c,
                out_channels=int(v),
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=1,
                bias=True,
                max_neurons=max_neurons,
                max_axons=max_axons,
                use_batchnorm=True,
                name=f"vgg_conv_{conv_idx}",
            )
            self.feature_mappers.append(conv)
            out = conv
            in_c = int(v)
            conv_idx += 1

        avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.feature_ops.append(avgpool)
        out = AdaptiveAvgPool2DOpMapper(out, avgpool, name="vgg_adaptive_avgpool_7x7")

        out = EinopsRearrangeMapper(out, "... c h w -> ... (c h w)")
        out = Ensure2DMapper(out)

        fc1 = Perceptron(4096, 512 * 7 * 7, normalization=nn.LazyBatchNorm1d(), name="vgg_fc1")
        fc1.regularization = nn.Dropout()
        fc2 = Perceptron(4096, 4096, normalization=nn.LazyBatchNorm1d(), name="vgg_fc2")
        fc2.regularization = nn.Dropout()
        fc3 = Perceptron(self.num_classes, 4096, normalization=nn.Identity(), name="vgg_fc3")

        self.classifier_perceptrons.extend([fc1, fc2, fc3])

        out = PerceptronMapper(out, fc1)
        out = PerceptronMapper(out, fc2)
        out = PerceptronMapper(out, fc3)

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


