"""Convolution mappers: perceptron-style (shared-weight) and legacy layer-wrap."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.mappers.pooling import _chunk_sizes
from mimarsinan.mapping.soft_core_mapper import map_mm
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer


class Conv2DPerceptronMapper(Mapper):
    """
    Convolution implemented as:
    - Forward: efficient nn.Conv2d
    - Mapping: shared-weight Perceptron (im2col + matmul), tiled as needed.
    - owned_perceptron_groups(): only chip-targeted perceptrons (not Identity).
    """

    def __init__(
        self,
        source_mapper,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        max_neurons: int | None = None,
        max_axons: int | None = None,
        use_batchnorm: bool = True,
        name: str = "Conv2DPerceptron",
        base_activation_name=None,
    ):
        super().__init__(source_mapper)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        if isinstance(kernel_size, tuple):
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            k = int(kernel_size)
            self.kernel_size = (k, k)

        if isinstance(stride, tuple):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            s = int(stride)
            self.stride = (s, s)

        if isinstance(padding, tuple):
            self.padding = (int(padding[0]), int(padding[1]))
        else:
            p = int(padding)
            self.padding = (p, p)

        if isinstance(dilation, tuple):
            self.dilation = (int(dilation[0]), int(dilation[1]))
        else:
            d = int(dilation)
            self.dilation = (d, d)

        self.bias = bool(bias)
        self.name = str(name)
        self.use_batchnorm = bool(use_batchnorm)
        self.max_neurons = max_neurons

        k_h, k_w = self.kernel_size
        patch_size = self.in_channels * k_h * k_w

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
        if x.dim() != 4:
            raise ValueError(
                f"{self.name}: expected input (B,C,H,W), got shape {tuple(x.shape)}"
            )

        x = self.perceptron.input_activation(x)

        w = self.perceptron.layer.weight.view(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )
        b = self.perceptron.layer.bias if self.bias else None

        y = F.conv2d(
            x, w, b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        if self.use_batchnorm or not isinstance(self.perceptron.normalization, nn.Identity):
            b_sz, c_sz, h_sz, w_sz = y.shape
            y = y.view(b_sz, c_sz, -1)
            y = self.perceptron.normalization(y)
            y = y.view(b_sz, c_sz, h_sz, w_sz)

        y = self.perceptron.scaler(y)
        y = self.perceptron.activation(y)
        if self.training:
            y = self.perceptron.regularization(y)

        return y

    def _map(self, mapping):
        input_sources = self.source_mapper.map(mapping)

        if len(input_sources.shape) != 3:
            raise ValueError(
                f"{self.name}: expects 3D input sources (C,H,W), got {input_sources.shape}"
            )

        c_in, h_in, w_in = input_sources.shape
        if int(c_in) != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        d_h, d_w = self.dilation

        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        zero_source = SpikeSource(-1, 0, False, True)
        if p_h > 0 or p_w > 0:
            pad_width = ((0, 0), (p_h, p_h), (p_w, p_w))
            input_sources = np.pad(
                input_sources,
                pad_width,
                mode="constant",
                constant_values=zero_source,
            )

        unfolded_sources_list = []
        for oh in range(h_out):
            for ow in range(w_out):
                h_start = oh * s_h
                w_start = ow * s_w

                patch = []
                for c in range(self.in_channels):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            r = h_start + kh * d_h
                            c_idx = w_start + kw * d_w
                            patch.append(input_sources[c, r, c_idx])

                unfolded_sources_list.append(patch)

        unfolded_sources = np.array(unfolded_sources_list, dtype=object).transpose()

        full_w = PerceptronTransformer().get_effective_weight(self.perceptron)
        full_b = PerceptronTransformer().get_effective_bias(self.perceptron)

        if self.max_neurons is None:
            group_sizes = [self.out_channels]
        else:
            group_sizes = _chunk_sizes(self.out_channels, int(self.max_neurons))

        mapped_groups = []
        start_idx = 0
        for g in group_sizes:
            end_idx = start_idx + g

            w_slice = full_w[start_idx:end_idx, :]
            b_slice = full_b[start_idx:end_idx] if full_b is not None else None

            mapped = map_mm(
                mapping,
                unfolded_sources,
                w_slice,
                b_slice,
                self.perceptron.activation_scale,
                self.perceptron.parameter_scale,
                self.perceptron.input_activation_scale,
            )
            mapped_groups.append(mapped)
            start_idx = end_idx

        mapped_sources = np.concatenate(mapped_groups, axis=0)
        return mapped_sources.reshape(self.out_channels, h_out, w_out)

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)

        if len(input_sources.shape) != 3:
            raise ValueError(
                f"{self.name}: expects 3D input sources (C,H,W), got {input_sources.shape}"
            )

        c_in, h_in, w_in = input_sources.shape
        if int(c_in) != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        d_h, d_w = self.dilation

        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        off_source = IRSource(node_id=-1, index=0)
        if p_h > 0 or p_w > 0:
            pad_width = ((0, 0), (p_h, p_h), (p_w, p_w))
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

        all_output_sources = []

        for oh in range(h_out):
            for ow in range(w_out):
                h_start = oh * s_h
                w_start = ow * s_w

                patch_sources = []
                for c in range(self.in_channels):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            r = h_start + kh * d_h
                            c_idx = w_start + kw * d_w
                            patch_sources.append(input_sources[c, r, c_idx])

                patch_sources = np.array(patch_sources)

                position_outputs = []
                for g_idx, bank_id in enumerate(bank_ids):
                    core_outputs = ir_mapping.add_shared_neural_core(
                        input_sources=patch_sources,
                        weight_bank_id=bank_id,
                        has_bias=has_bias,
                        name=f"{self.name}_pos{oh}_{ow}_g{g_idx}",
                        activation_type=activation_type,
                        perceptron_index=getattr(self, "perceptron_index", None),
                    )
                    position_outputs.append(core_outputs)
                all_output_sources.append(np.concatenate(position_outputs))

        output_array = np.stack(all_output_sources, axis=0)
        output_array = output_array.T
        return output_array.reshape(self.out_channels, h_out, w_out)


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

    def _map(self, mapping):
        input_sources = self.source_mapper.map(mapping)
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

        if p > 0:
            zero_source = SpikeSource(-1, 0, False, True)
            zeros = np.full((self.in_channels, p), zero_source, dtype=object)
            input_sources = np.concatenate([zeros, input_sources, zeros], axis=-1)

        l_out = (l_in + 2 * p - d * (k - 1) - 1) // s + 1

        unfolded_list = []
        for i in range(l_out):
            start = i * s
            end = start + k * d
            window_indices = np.arange(start, end, d)
            window = input_sources[:, window_indices]
            unfolded_list.append(window.flatten())

        unfolded_sources = np.stack(unfolded_list, axis=1)

        full_w = PerceptronTransformer().get_effective_weight(self.perceptron)
        full_b = PerceptronTransformer().get_effective_bias(self.perceptron)

        if self.max_neurons is None:
            group_sizes = [self.out_channels]
        else:
            group_sizes = _chunk_sizes(self.out_channels, int(self.max_neurons))

        mapped_groups = []
        start_idx = 0
        for g in group_sizes:
            end_idx = start_idx + g

            w_slice = full_w[start_idx:end_idx, :]
            b_slice = full_b[start_idx:end_idx] if full_b is not None else None

            mapped = map_mm(
                mapping,
                unfolded_sources,
                w_slice,
                b_slice,
                self.perceptron.activation_scale,
                self.perceptron.parameter_scale,
                self.perceptron.input_activation_scale,
            )
            mapped_groups.append(mapped)
            start_idx = end_idx

        mapped_sources = np.concatenate(mapped_groups, axis=0)
        return mapped_sources


class Conv1DMapper(Mapper):
    def __init__(self, source_mapper, conv_layer):
        super(Conv1DMapper, self).__init__(source_mapper)
        self.conv_layer = conv_layer

    def _map(self, mapping):
        weights = self.conv_layer.weight
        bias = self.conv_layer.bias

        input_sources = self.source_mapper.map(mapping)

        C_in = input_sources.shape[-2]
        L_in = input_sources.shape[-1]

        if self.conv_layer.padding[0] > 0:
            pad = self.conv_layer.padding[0]
            zero_source = SpikeSource(-1, 0, False, True)

            zeros = np.full((C_in, pad), zero_source, dtype=object)
            input_sources_padded = np.concatenate([zeros, input_sources, zeros], axis=-1)
        else:
            input_sources_padded = input_sources

        L_out = (L_in + 2 * self.conv_layer.padding[0] - self.conv_layer.dilation[0] * (self.conv_layer.kernel_size[0] - 1) - 1) // self.conv_layer.stride[0] + 1

        unfolded_sources_list = []
        K = self.conv_layer.kernel_size[0]
        S = self.conv_layer.stride[0]
        D = self.conv_layer.dilation[0]

        for i in range(L_out):
            start = i * S
            end = start + K * D
            window_indices = np.arange(start, end, D)
            window = input_sources_padded[:, window_indices]
            unfolded_sources_list.append(window.flatten())

        unfolded_sources = np.stack(unfolded_sources_list, axis=1)

        mm_weights = self.conv_layer.weight.view(self.conv_layer.out_channels, -1)
        mm_bias = self.conv_layer.bias

        mapped_sources = map_mm(
            mapping,
            unfolded_sources,
            mm_weights,
            mm_bias,
            torch.tensor(1.0),
            torch.tensor(mapping.q_max)
        )

        return mapped_sources

    def _forward_impl(self, x):
        return self.conv_layer(x)


class Conv2DMapper(Mapper):
    def __init__(self, source_mapper, conv_layer):
        super(Conv2DMapper, self).__init__(source_mapper)
        self.conv_layer = conv_layer

    def _map(self, mapping):
        input_sources = self.source_mapper.map(mapping)

        if len(input_sources.shape) != 3:
            raise ValueError(f"Conv2DMapper expects 3D input sources (C, H, W), got {input_sources.shape}")

        C_in, H_in, W_in = input_sources.shape

        K_h, K_w = self.conv_layer.kernel_size
        S_h, S_w = self.conv_layer.stride
        P_h, P_w = self.conv_layer.padding
        D_h, D_w = self.conv_layer.dilation

        H_out = (H_in + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W_in + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1

        zero_source = SpikeSource(-1, 0, False, True)
        if P_h > 0 or P_w > 0:
            pad_width = ((0, 0), (P_h, P_h), (P_w, P_w))
            input_sources_padded = np.pad(
                input_sources,
                pad_width,
                mode='constant',
                constant_values=zero_source
            )
        else:
            input_sources_padded = input_sources

        unfolded_sources_list = []

        for h in range(H_out):
            for w in range(W_out):
                h_start = h * S_h
                w_start = w * S_w

                patch = []
                for c in range(C_in):
                    for kh in range(K_h):
                        for kw in range(K_w):
                            r = h_start + kh * D_h
                            c_idx = w_start + kw * D_w
                            patch.append(input_sources_padded[c, r, c_idx])

                unfolded_sources_list.append(patch)

        unfolded_sources = np.array(unfolded_sources_list).transpose()

        mm_weights = self.conv_layer.weight.reshape(self.conv_layer.out_channels, -1)
        mm_bias = self.conv_layer.bias

        mapped_sources = map_mm(
            mapping,
            unfolded_sources,
            mm_weights,
            mm_bias,
            torch.tensor(1.0),
            torch.tensor(mapping.q_max)
        )

        return mapped_sources.reshape(self.conv_layer.out_channels, H_out, W_out)

    def _forward_impl(self, x):
        return self.conv_layer(x)
