"""Convolution mappers: perceptron-style (shared-weight)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.conv_helpers import map_shared_weight_conv
from mimarsinan.mapping.mappers.conv_unfold import (
    ConvPatchGather,
    conv_patch_gather,
    conv_slot_unfold,
)
from mimarsinan.mapping.mappers.flowchart import FlowchartFCSpec, FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    perceptron_boundary_scale,
    perceptron_per_source_scale,
)
from mimarsinan.mapping.mappers.slot_unfold import SerialSlotUnfold
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.pruning.committed_masks import commit_layer_pruning


class Conv1DPerceptronMapper(Mapper):
    """1D convolution: efficient nn.Conv1d forward; shared-weight Perceptron (unfold+matmul) in IR, tiled as needed."""

    def __init__(
        self,
        source_mapper,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        max_neurons: int | None = None,
        max_axons: int | None = None,
        use_batchnorm: bool = True,
        name: str = "Conv1DPerceptron",
        base_activation_name=None,
    ):
        super().__init__(source_mapper)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.bias = bool(bias)
        self.name = str(name)
        self.use_batchnorm = bool(use_batchnorm)
        self.max_neurons = max_neurons

        patch_size = self.in_channels * self.kernel_size

        self.perceptron = Perceptron(
            output_channels=self.out_channels,
            input_features=patch_size,
            normalization=(nn.LazyBatchNorm1d() if self.use_batchnorm else nn.Identity()),
            bias=self.bias,
            base_activation_name=base_activation_name,
            name=f"{self.name}_full",
        )
        # _forward_impl drives the shared perceptron through F.conv1d, so its
        # activation output is [B, C, L] — channels-first.
        self.perceptron.output_channel_axis = 1

    def __setstate__(self, state):
        super().__setstate__(state)
        # Caches saved before the layout declaration: the owner re-declares it.
        self.perceptron.output_channel_axis = 1

    def owned_perceptron_groups(self):
        return [[self.perceptron]]

    def patch_gather(self, input_shape) -> ConvPatchGather:
        """This convolution's unfold over a ``(C, L)`` input: the slot table the NF event-serial twin folds
        through, pinned equal to the IR mapping's gather (``conv_helpers.gather_conv_patches``) by test."""
        c_in, l_in = (int(v) for v in input_shape)
        if c_in != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )
        return conv_patch_gather(
            in_channels=self.in_channels,
            source_grid=(l_in,),
            kernel=(self.kernel_size,),
            stride=(self.stride,),
            padding=(self.padding,),
            dilation=(self.dilation,),
        )

    def serial_slot_unfold(self, *, input_shape, n_slots: int) -> SerialSlotUnfold:
        del n_slots  # the twin checks the table against its own weight
        return conv_slot_unfold(self.patch_gather(input_shape))

    def propagate_source_scale(self, deps, out_scales):
        return perceptron_per_source_scale(self, deps, out_scales)

    def propagate_boundary_scale(self, deps, out_scales, default):
        return perceptron_boundary_scale(self, deps, out_scales, default)

    def flowchart_node_estimate(self, out_shape):
        in_f = int(self.in_channels * self.kernel_size)
        out_f = int(self.out_channels)
        sw_text = (
            f"SW perceptrons=1 (shared conv weights, patch={in_f}, "
            f"out_channels={out_f})"
        )
        if out_shape is not None and len(out_shape) == 2:
            _, l = out_shape
            inst = int(l)
        else:
            inst = 1
        return FlowchartNodeEstimate(
            sw_text=sw_text,
            fc_spec=FlowchartFCSpec(
                in_features=in_f,
                out_features=out_f,
                instances=inst,
                has_bias=bool(self.bias),
            ),
        )

    def _forward_impl(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"{self.name}: expected input (B,C,L), got shape {tuple(x.shape)}"
            )

        x = self.perceptron.input_activation(x)

        # F.conv1d reads raw weights without calling layer.__call__, so the
        # pruning pre-hooks never fire here; commit the masks directly.
        commit_layer_pruning(self.perceptron.layer)
        w = self.perceptron.layer.weight.view(
            self.out_channels, self.in_channels, self.kernel_size
        )
        # Live additive term, not the construction-time ``self.bias`` flag: see
        # Conv2DPerceptronMapper._forward_impl (Normalization Fusion installs a
        # bias on a previously bias-free layer).
        b = self.perceptron.layer.bias

        y = F.conv1d(
            x, w, b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        if self.use_batchnorm or not isinstance(self.perceptron.normalization, nn.Identity):
            y = self.perceptron.normalization(y)

        y = self.perceptron.scaler(y)
        y = self.perceptron.activation(y)
        if self.training:
            y = self.perceptron.regularization(y)
        return y

    def _map_to_ir(self, ir_mapping):
        input_sources = self.require_source_mapper().map_to_ir(ir_mapping)
        return map_shared_weight_conv(
            self, ir_mapping, input_sources,
            kernel_size=(self.kernel_size,), stride=(self.stride,),
            padding=(self.padding,), dilation=(self.dilation,), layout="(C,L)",
        )
