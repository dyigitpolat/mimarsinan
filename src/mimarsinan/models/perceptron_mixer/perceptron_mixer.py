from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow

import torch.nn as nn
import einops


class PerceptronMixer(PerceptronFlow):
    def __init__(
        self,
        device,
        input_shape,
        num_classes,
        patch_n_1,
        patch_m_1,
        patch_c_1,
        fc_w_1,
        fc_w_2,
    ):
        super(PerceptronMixer, self).__init__(device)

        self.input_activation = nn.Identity()

        self.input_shape = input_shape
        self.input_channels = input_shape[-3]

        self.patch_rows = patch_n_1
        self.patch_cols = patch_m_1
        self.patch_channels = patch_c_1

        self.patch_height = input_shape[-2] // self.patch_rows
        self.patch_width = input_shape[-1] // self.patch_cols

        self.patch_size = (
            self.patch_height * self.patch_width * self.input_channels
        )
        self.patch_count = self.patch_rows * self.patch_cols

        self.patch_layer = Perceptron(
            self.patch_channels,
            self.patch_size,
            normalization=nn.LazyBatchNorm1d(),
        )
        self.patch_layer_CONV = nn.Conv2d(
            self.input_channels,
            self.patch_channels,
            kernel_size=self.patch_height,
            stride=self.patch_height,
        )

        self.patch_layers_list = nn.ModuleList()
        self.patch_layers_list_2 = nn.ModuleList()
        self.fc_layers_list = nn.ModuleList()
        self.fc_layers_list_2 = nn.ModuleList()

        self.mixer_count = 2

        XX = fc_w_1
        YY = fc_w_2
        for mixer_idx in range(self.mixer_count):
            self.patch_layers_list.append(
                Perceptron(
                    XX,
                    self.patch_count,
                    normalization=nn.LazyBatchNorm1d(),
                    name=f"tok_mixer_{mixer_idx}",
                )
            )
            self.fc_layers_list.append(Perceptron(self.patch_count, XX))

            self.patch_layers_list_2.append(
                Perceptron(
                    YY,
                    self.patch_channels,
                    normalization=nn.LazyBatchNorm1d(),
                    name=f"ch_mixer_{mixer_idx}",
                )
            )
            self.fc_layers_list_2.append(Perceptron(self.patch_channels, YY))

        self.output_layer = Perceptron(
            num_classes, self.patch_count * self.patch_channels
        )

        self.out = None

        # Single source of truth: mapper graph (also used for forward execution)
        inp = InputMapper(self.input_shape)
        self._input_activation_mapper = ModuleMapper(inp, self.input_activation)

        out = EinopsRearrangeMapper(
            self._input_activation_mapper,
            "... c (h p1) (w p2) -> ... (h w) (p1 p2 c)",
            p1=self.patch_height,
            p2=self.patch_width,
        )
        # (B, NP, PS) -> (B*NP, PS) for eval; identity for mapping (NP, PS)
        out = MergeLeadingDimsMapper(out)
        out = PerceptronMapper(out, self.patch_layer)
        # (B*NP, CP) -> (B, NP, CP) for eval; identity for mapping (NP, CP)
        out = SplitLeadingDimMapper(out, second_dim_size=self.patch_count)

        for mixer_idx in range(self.mixer_count):
            # Token mixer
            out = EinopsRearrangeMapper(out, "... np cp -> ... cp np")
            out = MergeLeadingDimsMapper(out)  # (B, CP, NP) -> (B*CP, NP)
            out = PerceptronMapper(out, self.patch_layers_list[mixer_idx])
            out = PerceptronMapper(out, self.fc_layers_list[mixer_idx])
            out = SplitLeadingDimMapper(out, second_dim_size=self.patch_channels)  # (B*CP, NP) -> (B, CP, NP)
            out = EinopsRearrangeMapper(out, "... cp np -> ... np cp")

            # Channel mixer
            out = MergeLeadingDimsMapper(out)  # (B, NP, CP) -> (B*NP, CP)
            out = PerceptronMapper(out, self.patch_layers_list_2[mixer_idx])
            out = PerceptronMapper(out, self.fc_layers_list_2[mixer_idx])
            out = SplitLeadingDimMapper(out, second_dim_size=self.patch_count)  # (B*NP, CP) -> (B, NP, CP)

        out = EinopsRearrangeMapper(out, "... np cp -> ... (np cp)")
        out = Ensure2DMapper(out)
        out = PerceptronMapper(out, self.output_layer)
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
