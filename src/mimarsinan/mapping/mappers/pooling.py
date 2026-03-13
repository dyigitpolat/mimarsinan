"""Pooling mappers and chunk-size helper for tiled mapping."""

from __future__ import annotations

import torch.nn as nn

from mimarsinan.mapping.mappers.base import Mapper


def _chunk_sizes(total: int, chunk: int):
    assert chunk > 0
    sizes = []
    remaining = int(total)
    while remaining > 0:
        sizes.append(min(chunk, remaining))
        remaining -= sizes[-1]
    return sizes


class MaxPool2DMapper(Mapper):
    """
    MaxPool2d operation.

    - Forward: nn.MaxPool2d
    - Old mapping (SoftCoreMapping): raises NotImplementedError
    - IR mapping (IRMapping): produces ComputeOp node
    """

    def __init__(
        self,
        source_mapper,
        kernel_size,
        stride=None,
        padding=0,
        input_spatial_shape=None,
        input_channels=None,
        name: str = "MaxPool2d",
    ):
        super().__init__(source_mapper)
        self.name = str(name)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

        self.input_spatial_shape = input_spatial_shape
        self.input_channels = input_channels

        self.pool = nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: pooling is not supported in SoftCoreMapping. "
            f"Use IRMapping for unified IR that supports ComputeOps."
        )

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)

        if self.input_spatial_shape is not None and self.input_channels is not None:
            c, h_in, w_in = self.input_channels, self.input_spatial_shape[0], self.input_spatial_shape[1]
        else:
            if len(input_sources.shape) == 3:
                c, h_in, w_in = input_sources.shape
            else:
                raise ValueError(
                    f"{self.name}: cannot infer input shape. "
                    f"Provide input_spatial_shape and input_channels, or ensure source has 3D shape."
                )

        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        output_sources = ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="max_pool2d",
            params={
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
            input_shape=(c, h_in, w_in),
            output_shape=(c, h_out, w_out),
            name=self.name,
        )

        return output_sources

    def _forward_impl(self, x):
        return self.pool(x)


class AvgPool2DMapper(Mapper):
    """
    AvgPool2d operation.

    - Forward: nn.AvgPool2d
    - Old mapping: raises NotImplementedError
    - IR mapping: produces ComputeOp node
    """

    def __init__(
        self,
        source_mapper,
        kernel_size,
        stride=None,
        padding=0,
        input_spatial_shape=None,
        input_channels=None,
        name: str = "AvgPool2d",
    ):
        super().__init__(source_mapper)
        self.name = str(name)

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

        self.input_spatial_shape = input_spatial_shape
        self.input_channels = input_channels

        self.pool = nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: pooling is not supported in SoftCoreMapping. "
            f"Use IRMapping for unified IR that supports ComputeOps."
        )

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)

        if self.input_spatial_shape is not None and self.input_channels is not None:
            c, h_in, w_in = self.input_channels, self.input_spatial_shape[0], self.input_spatial_shape[1]
        else:
            if len(input_sources.shape) == 3:
                c, h_in, w_in = input_sources.shape
            else:
                raise ValueError(f"{self.name}: cannot infer input shape.")

        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        output_sources = ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="avg_pool2d",
            params={
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
            input_shape=(c, h_in, w_in),
            output_shape=(c, h_out, w_out),
            name=self.name,
        )

        return output_sources

    def _forward_impl(self, x):
        return self.pool(x)


class AdaptiveAvgPool2DMapper(Mapper):
    """
    AdaptiveAvgPool2d operation.

    - Forward: nn.AdaptiveAvgPool2d
    - Old mapping: raises NotImplementedError
    - IR mapping: produces ComputeOp node
    """

    def __init__(
        self,
        source_mapper,
        output_size,
        input_channels=None,
        name: str = "AdaptiveAvgPool2d",
    ):
        super().__init__(source_mapper)
        self.name = str(name)

        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)

        self.input_channels = input_channels
        self.pool = nn.AdaptiveAvgPool2d(self.output_size)

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: adaptive pooling is not supported in SoftCoreMapping. "
            f"Use IRMapping for unified IR that supports ComputeOps."
        )

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)

        if self.input_channels is not None:
            c = self.input_channels
        elif len(input_sources.shape) >= 1:
            c = input_sources.shape[0]
        else:
            raise ValueError(f"{self.name}: cannot infer input channels.")

        h_out, w_out = self.output_size

        output_sources = ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="adaptive_avg_pool2d",
            params={"output_size": self.output_size},
            input_shape=tuple(input_sources.shape) if len(input_sources.shape) >= 2 else None,
            output_shape=(c, h_out, w_out),
            name=self.name,
        )

        return output_sources

    def _forward_impl(self, x):
        return self.pool(x)
