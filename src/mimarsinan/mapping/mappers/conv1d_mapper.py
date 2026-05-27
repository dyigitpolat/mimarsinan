"""Convolution mappers: perceptron-style (shared-weight)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer


from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.conv_helpers import _chunk_sizes

class Conv1DPerceptronMapper(Mapper):
    """
    1D Convolution implemented as:
    - Forward: efficient nn.Conv1d
    - Mapping: shared-weight Perceptron (unfold + matmul), tiled as needed.
    """

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

    def owned_perceptron_groups(self):
        return [[self.perceptron]]

    def _forward_impl(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"{self.name}: expected input (B,C,L), got shape {tuple(x.shape)}"
            )

        x = self.perceptron.input_activation(x)

        w = self.perceptron.layer.weight.view(
            self.out_channels, self.in_channels, self.kernel_size
        )
        b = self.perceptron.layer.bias if self.bias else None

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
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        if len(input_sources.shape) != 2:
            raise ValueError(
                f"{self.name}: expects 2D input sources (C,L), got {input_sources.shape}"
            )

        c_in, l_in = input_sources.shape
        if int(c_in) != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )

        k = self.kernel_size
        s = self.stride
        p = self.padding
        d = self.dilation
        l_out = (l_in + 2 * p - d * (k - 1) - 1) // s + 1

        if getattr(self.perceptron, "is_encoding_layer", False):
            flat_in = np.array(input_sources, dtype=object).flatten()
            out = ir_mapping.add_compute_op(
                input_sources=flat_in,
                op_type="module",
                params={"module": self, "input_shape": (c_in, l_in)},
                input_shape=(c_in, l_in),
                output_shape=(self.out_channels, l_out),
                name=self.name,
            )
            return out.reshape(self.out_channels, l_out)

        off_source = IRSource(node_id=-1, index=0)
        if p > 0:
            pad_width = ((0, 0), (p, p))
            input_sources = np.pad(
                input_sources,
                pad_width,
                mode="constant",
                constant_values=off_source,
            )

        full_w = PerceptronTransformer().get_effective_weight(self.perceptron)
        full_b = PerceptronTransformer().get_effective_bias(self.perceptron)
        has_bias = full_b is not None
        activation_type = resolve_activation_type(self.perceptron)

        if self.max_neurons is None:
            group_sizes = [self.out_channels]
        else:
            group_sizes = _chunk_sizes(self.out_channels, int(self.max_neurons))

        bank_ids: list[int] = []
        start_idx = 0
        for g in group_sizes:
            end_idx = start_idx + g
            w_slice = full_w[start_idx:end_idx, :]
            b_slice = full_b[start_idx:end_idx] if has_bias else None
            bank_id = ir_mapping.register_weight_bank(
                weights=w_slice,
                biases=b_slice,
                activation_scale=self.perceptron.activation_scale,
                parameter_scale=self.perceptron.parameter_scale,
                input_activation_scale=self.perceptron.input_activation_scale,
                perceptron_index=getattr(self, "perceptron_index", None),
            )
            bank_ids.append(bank_id)
            start_idx = end_idx

        l_base = np.arange(l_out) * s
        k_off = np.arange(k) * d
        l_idx = l_base[:, None] + k_off[None, :]
        c_idx = np.arange(self.in_channels)[:, None, None]
        l_idx_b, c_idx_b = np.broadcast_arrays(l_idx[None, :, :], c_idx)
        patches = input_sources[c_idx_b, l_idx_b]
        patch_size = self.in_channels * k
        patches_flat = patches.reshape(l_out, patch_size)

        all_output_sources = []
        for pos in range(l_out):
            patch_sources = patches_flat[pos]
            position_outputs = []
            for g_idx, bank_id in enumerate(bank_ids):
                core_outputs = ir_mapping.add_shared_neural_core(
                    input_sources=patch_sources,
                    weight_bank_id=bank_id,
                    has_bias=has_bias,
                    name=f"{self.name}_pos{pos}_g{g_idx}",
                    activation_type=activation_type,
                    perceptron_index=getattr(self, "perceptron_index", None),
                )
                position_outputs.append(core_outputs)
            all_output_sources.append(np.concatenate(position_outputs))

        output_array = np.stack(all_output_sources, axis=0).T
        return output_array.reshape(self.out_channels, l_out)
