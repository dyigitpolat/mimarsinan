"""Convolution mappers: perceptron-style (shared-weight)."""

from __future__ import annotations

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mimarsinan.mapping.mappers.base import Mapper, resolve_activation_type
from mimarsinan.mapping.mappers.conv_helpers import _chunk_sizes, pad_source_grid
from mimarsinan.mapping.mappers.flowchart import FlowchartFCSpec, FlowchartNodeEstimate
from mimarsinan.mapping.mappers.scale_propagation import (
    perceptron_boundary_scale,
    perceptron_per_source_scale,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.transformations.perceptron.perceptron_transformer import PerceptronTransformer
from mimarsinan.transformations.pruning.committed_masks import commit_layer_pruning


class Conv2DPerceptronMapper(Mapper):
    """2D convolution: efficient nn.Conv2d forward; shared-weight Perceptron (im2col+matmul) in IR, tiled as needed."""

    def __init__(
        self,
        source_mapper,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
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
        # _forward_impl drives the shared perceptron through F.conv2d, so its
        # activation output is [B, C, H, W] — channels-first.
        self.perceptron.output_channel_axis = 1

    def __setstate__(self, state):
        super().__setstate__(state)
        # Caches saved before the layout declaration: the owner re-declares it.
        self.perceptron.output_channel_axis = 1

    def owned_perceptron_groups(self):
        return [[self.perceptron]]

    def propagate_source_scale(self, deps, out_scales):
        return perceptron_per_source_scale(self, deps, out_scales)

    def propagate_boundary_scale(self, deps, out_scales, default):
        return perceptron_boundary_scale(self, deps, out_scales, default)

    def flowchart_node_estimate(self, out_shape):
        k_h, k_w = self.kernel_size
        in_f = int(self.in_channels * k_h * k_w)
        out_f = int(self.out_channels)
        sw_text = (
            f"SW perceptrons=1 (shared conv weights, patch={in_f}, "
            f"out_channels={out_f})"
        )
        if out_shape is not None and len(out_shape) == 3:
            _, h, w = out_shape
            inst = int(h * w)
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
        if x.dim() != 4:
            raise ValueError(
                f"{self.name}: expected input (B,C,H,W), got shape {tuple(x.shape)}"
            )

        x = self.perceptron.input_activation(x)

        # F.conv2d reads raw weights without calling layer.__call__, so the
        # pruning pre-hooks never fire here; commit the masks directly.
        commit_layer_pruning(self.perceptron.layer)
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

    def _map_to_ir(self, ir_mapping):
        input_sources = self.require_source_mapper().map_to_ir(ir_mapping)

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

        if getattr(self.perceptron, "is_encoding_layer", False):
            flat_in = np.array(input_sources, dtype=object).flatten()
            out = ir_mapping.add_compute_op(
                input_sources=flat_in,
                op_type="module",
                params={"module": self, "input_shape": (c_in, h_in, w_in)},
                input_shape=(c_in, h_in, w_in),
                output_shape=(self.out_channels, h_out, w_out),
                name=self.name,
            )
            return out.reshape(self.out_channels, h_out, w_out)

        if p_h > 0 or p_w > 0:
            input_sources = pad_source_grid(
                input_sources, ((0, 0), (p_h, p_h), (p_w, p_w))
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

        h_base = np.arange(h_out) * s_h
        w_base = np.arange(w_out) * s_w
        kh_off = np.arange(k_h) * d_h
        kw_off = np.arange(k_w) * d_w
        h_idx = (h_base[:, None, None, None, None] + kh_off[None, None, None, :, None])
        w_idx = (w_base[None, :, None, None, None] + kw_off[None, None, None, None, :])
        c_idx = np.arange(self.in_channels)[None, None, :, None, None]
        h_idx_b, w_idx_b, c_idx_b = np.broadcast_arrays(h_idx, w_idx, c_idx)
        patches = input_sources[c_idx_b, h_idx_b, w_idx_b]
        n_positions = h_out * w_out
        patch_size = self.in_channels * k_h * k_w
        patches_flat = patches.reshape(n_positions, patch_size)

        all_output_sources = []
        for pos in range(n_positions):
            oh, ow = divmod(pos, w_out)
            patch_sources = patches_flat[pos]
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

