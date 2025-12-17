from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch.nn as nn
import einops


class SimpleMLP(PerceptronFlow):
    def __init__(
        self,
        device,
        input_shape,
        num_classes,
        mlp_width_1,
        mlp_width_2,
    ):
        super(SimpleMLP, self).__init__(device)

        self.input_activation = nn.Identity()

        self.input_shape = input_shape
        self.input_width = (
            input_shape[-3] * input_shape[-2] * input_shape[-1]
        )

        w1 = mlp_width_1
        w2 = mlp_width_2
        network_shape = [self.input_width, w1, w2, w1, num_classes]
        has_norm = [True, False, True, False]

        self.perceptrons = nn.ModuleList()
        for i in range(len(network_shape) - 1):
            if has_norm[i]:
                norm = nn.LazyBatchNorm1d()
            else:
                norm = nn.Identity()

            self.perceptrons.append(
                Perceptron(
                    output_channels=network_shape[i + 1],
                    input_features=network_shape[i],
                    normalization=norm,
                )
            )

        # Single source of truth: mapper graph (also used for forward execution)
        inp = InputMapper(self.input_shape)
        self._input_activation_mapper = ModuleMapper(inp, self.input_activation)
        out = EinopsRearrangeMapper(
            self._input_activation_mapper, "... c h w -> ... (c h w)"
        )
        out = Ensure2DMapper(out)
        for perceptron in self.perceptrons:
            out = PerceptronMapper(out, perceptron)
        self._mapper_repr = ModelRepresentation(out)

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation
        # Keep mapper graph consistent
        self._input_activation_mapper.module = activation

    def get_perceptrons(self):
        return self.perceptrons

    def get_perceptron_groups(self):
        groups = []
        for p in self.perceptrons:
            groups.append([p])

        return groups

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)

