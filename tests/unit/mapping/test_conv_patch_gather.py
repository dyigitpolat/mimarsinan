"""Shared-weight conv lowering: one im2col gather and one IR wiring for Conv1d and Conv2d.

The gather's column order must equal ``ConvNd.weight.reshape(out, -1)`` (channel
major, then kernel taps) and its rows the row-major output positions; the old
Conv1d gather reshaped a (C, L, k) patch tensor as (L, C*k), which mixed
channels and positions whenever C > 1 and L_out > 1 (D9).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mimarsinan.mapping.ir import IRSource, NeuralCore
from mimarsinan.mapping.map_model_to_ir import map_model_to_ir
from mimarsinan.mapping.mappers.conv1d_mapper import Conv1DPerceptronMapper
from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.conv_helpers import (
    conv_output_size,
    gather_conv_patches,
    pad_source_grid,
)
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation

_OFF = -1

# (kernel, stride, padding, dilation) per spatial axis.
_CONV1D_GEOMETRIES = [
    ((1,), (1,), (0,), (1,)),
    ((3,), (1,), (1,), (1,)),
    ((3,), (2,), (1,), (1,)),
    ((3,), (1,), (2,), (2,)),
    ((4,), (1,), (3,), (1,)),
    ((5,), (3,), (0,), (2,)),
]
_CONV2D_GEOMETRIES = [
    ((1, 1), (1, 1), (0, 0), (1, 1)),
    ((3, 3), (1, 1), (1, 1), (1, 1)),
    ((3, 3), (2, 2), (1, 1), (1, 1)),
    ((3, 3), (1, 1), (2, 2), (2, 2)),
    ((3, 1), (2, 1), (1, 0), (1, 1)),
    ((2, 3), (1, 2), (0, 2), (2, 1)),
]


def _index_grid(shape) -> np.ndarray:
    """Object grid of raw-input IRSources whose index is the flat position."""
    flat = np.empty(int(np.prod(shape)), dtype=object)
    for i in range(flat.size):
        flat[i] = IRSource(node_id=-2, index=i)
    return flat.reshape(shape)


def _as_indices(sources) -> np.ndarray:
    """Raw-input index per source; OFF (padding) as -1; anything else is a wiring error."""
    grid = np.asarray(sources)
    flat = []
    for s in grid.flat:
        assert s.is_off() or s.is_input(), f"unexpected source {s}"
        flat.append(_OFF if s.is_off() else int(s.index))
    return np.array(flat, dtype=np.int64).reshape(grid.shape)


def _core_patch(core) -> np.ndarray:
    """A conv core's patch axons: every input source but the trailing always-on bias axon."""
    sources = list(core.input_sources.flatten())
    assert sources[-1].is_always_on(), "a biased conv core ends with its bias axon"
    return _as_indices(np.array(sources[:-1], dtype=object))


def _gather(shape, kernel, stride, padding, dilation) -> np.ndarray:
    grid = _index_grid(shape)
    if any(p > 0 for p in padding):
        grid = pad_source_grid(grid, ((0, 0),) + tuple((p, p) for p in padding))
    out_size = conv_output_size(shape[1:], kernel, stride, padding, dilation)
    return _as_indices(gather_conv_patches(grid, kernel, stride, dilation, out_size))


def _torch_unfold_indices(shape, kernel, stride, padding, dilation) -> np.ndarray:
    """torch's own im2col of the index grid (OFF = padding), rows = output positions."""
    c = shape[0]
    spatial = shape[1:]
    if len(spatial) == 1:
        spatial = (1,) + tuple(spatial)
        kernel, stride = (1,) + tuple(kernel), (1,) + tuple(stride)
        padding, dilation = (0,) + tuple(padding), (1,) + tuple(dilation)
    ids = torch.arange(int(np.prod(shape)), dtype=torch.float64).reshape(1, c, *spatial) + 1.0
    cols = F.unfold(ids, kernel, dilation=dilation, padding=padding, stride=stride)
    return (cols[0].T - 1.0).to(torch.int64).numpy()


class TestConvOutputSize:
    @pytest.mark.parametrize("geometry", _CONV1D_GEOMETRIES + _CONV2D_GEOMETRIES)
    def test_matches_torch_conv(self, geometry):
        kernel, stride, padding, dilation = geometry
        spatial = (11,) * len(kernel)
        conv = F.conv1d if len(kernel) == 1 else F.conv2d
        y = conv(
            torch.zeros(1, 2, *spatial), torch.zeros(3, 2, *kernel),
            stride=stride, padding=padding, dilation=dilation,
        )
        assert conv_output_size(spatial, kernel, stride, padding, dilation) == tuple(y.shape[2:])

    def test_returns_python_ints(self):
        size = conv_output_size((np.int64(9), 7), (3, 3), (2, 1), (1, 0), (1, 1))
        assert size == (5, 5)
        assert all(type(v) is int for v in size)


class TestGatherConvPatches:
    """The gather is torch's unfold on the index grid, for every geometry."""

    @pytest.mark.parametrize("geometry", _CONV1D_GEOMETRIES)
    @pytest.mark.parametrize("channels", [1, 3])
    def test_conv1d_gather_equals_torch_unfold(self, geometry, channels):
        shape = (channels, 13)
        got = _gather(shape, *geometry)
        want = _torch_unfold_indices(shape, *geometry)
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("geometry", _CONV2D_GEOMETRIES)
    @pytest.mark.parametrize("channels", [1, 3])
    def test_conv2d_gather_equals_torch_unfold(self, geometry, channels):
        shape = (channels, 7, 6)
        got = _gather(shape, *geometry)
        want = _torch_unfold_indices(shape, *geometry)
        np.testing.assert_array_equal(got, want)

    @pytest.mark.parametrize("geometry", _CONV1D_GEOMETRIES + _CONV2D_GEOMETRIES)
    def test_patch_times_flat_weight_is_the_native_conv_fp64(self, geometry):
        kernel, stride, padding, dilation = geometry
        channels, out_channels = 4, 5
        spatial = (12,) * len(kernel)
        gen = torch.Generator().manual_seed(len(kernel) * 100 + sum(kernel) + sum(stride))
        x = torch.randn(channels, *spatial, dtype=torch.float64, generator=gen)
        w = torch.randn(out_channels, channels, *kernel, dtype=torch.float64, generator=gen)
        conv = F.conv1d if len(kernel) == 1 else F.conv2d
        want = conv(x[None], w, stride=stride, padding=padding, dilation=dilation)[0]

        idx = _gather((channels, *spatial), kernel, stride, padding, dilation)
        values = torch.cat([x.reshape(-1), torch.zeros(1, dtype=torch.float64)])
        patches = values[torch.from_numpy(np.where(idx == _OFF, x.numel(), idx))]
        got = (patches @ w.reshape(out_channels, -1).T).T.reshape(want.shape)
        assert float((got - want).abs().max()) <= 1e-12

    def test_rows_are_row_major_output_positions(self):
        idx = _gather((1, 4, 5), (1, 1), (1, 1), (0, 0), (1, 1))
        np.testing.assert_array_equal(idx[:, 0], np.arange(20))


def _conv1d_mapper(c_in, length, kernel, stride, padding, dilation, max_neurons=None):
    return Conv1DPerceptronMapper(
        InputMapper((c_in, length)), in_channels=c_in, out_channels=5,
        kernel_size=kernel, stride=stride, padding=padding, dilation=dilation,
        use_batchnorm=False, max_neurons=max_neurons, base_activation_name="ReLU",
    )


def _conv2d_mapper(c_in, hw, kernel, stride, padding, dilation, max_neurons=None):
    return Conv2DPerceptronMapper(
        InputMapper((c_in, *hw)), in_channels=c_in, out_channels=5,
        kernel_size=kernel, stride=stride, padding=padding, dilation=dilation,
        use_batchnorm=False, max_neurons=max_neurons, base_activation_name="ReLU",
    )


class TestSharedWeightConvIRWiring:
    """Every emitted core reads exactly torch's patch for its output position."""

    @staticmethod
    def _cores_by_position(ir):
        cores = [n for n in ir.nodes if isinstance(n, NeuralCore)]
        by_pos: dict[int, list[NeuralCore]] = {}
        for core in cores:
            by_pos.setdefault(int(core.perceptron_output_column), []).append(core)
        return by_pos

    @pytest.mark.parametrize(
        "geometry,max_neurons",
        list(itertools.product(_CONV1D_GEOMETRIES, [None, 2])),
    )
    def test_conv1d_cores_read_torch_patches(self, geometry, max_neurons):
        kernel, stride, padding, dilation = geometry
        mapper = _conv1d_mapper(3, 13, kernel[0], stride[0], padding[0], dilation[0], max_neurons)
        ir = map_model_to_ir(ModelRepresentation(mapper))
        want = _torch_unfold_indices((3, 13), *geometry)
        by_pos = self._cores_by_position(ir)
        assert sorted(by_pos) == list(range(want.shape[0]))
        for pos, cores in by_pos.items():
            assert len(cores) == (1 if max_neurons is None else 3)
            for core in cores:
                np.testing.assert_array_equal(_core_patch(core), want[pos])
                assert core.name.startswith(f"{mapper.name}_pos{pos}_g")

    @pytest.mark.parametrize("geometry", _CONV2D_GEOMETRIES)
    def test_conv2d_cores_read_torch_patches(self, geometry):
        kernel, stride, padding, dilation = geometry
        mapper = _conv2d_mapper(3, (7, 6), kernel, stride, padding, dilation)
        ir = map_model_to_ir(ModelRepresentation(mapper))
        want = _torch_unfold_indices((3, 7, 6), *geometry)
        h_out, w_out = conv_output_size((7, 6), kernel, stride, padding, dilation)
        by_pos = self._cores_by_position(ir)
        assert sorted(by_pos) == list(range(h_out * w_out))
        for pos, (core,) in by_pos.items():
            np.testing.assert_array_equal(_core_patch(core), want[pos])
            oh, ow = divmod(pos, w_out)
            assert core.name == f"{mapper.name}_pos{oh}_{ow}_g0"

    def test_output_grid_is_channels_first(self):
        mapper = _conv1d_mapper(2, 6, 3, 1, 1, 1, max_neurons=2)
        ir = map_model_to_ir(ModelRepresentation(mapper))
        out = ir.output_sources.reshape(5, 6)
        cores = {n.id: n for n in ir.nodes if isinstance(n, NeuralCore)}
        for channel in range(5):
            for pos in range(6):
                core = cores[out[channel, pos].node_id]
                assert core.perceptron_output_column == pos
                lo, hi = core.perceptron_output_slice
                assert lo <= channel < hi
                assert out[channel, pos].index == channel - lo

    @pytest.mark.parametrize("build,message", [
        (lambda: Conv1DPerceptronMapper(InputMapper((3, 7, 6)), 3, 5, 3, name="c1"),
         r"c1: expects 2D input sources \(C,L\)"),
        (lambda: Conv2DPerceptronMapper(InputMapper((3, 7)), 3, 5, 3, name="c2"),
         r"c2: expects 3D input sources \(C,H,W\)"),
        (lambda: Conv1DPerceptronMapper(InputMapper((4, 7)), 3, 5, 3, name="c1"),
         r"c1: expected in_channels=3, got 4"),
    ])
    def test_wrong_source_grid_is_refused_by_name(self, build, message):
        with pytest.raises(ValueError, match=message):
            map_model_to_ir(ModelRepresentation(build()))


class TestEncodingLayerHostOp:
    """An encoding conv is one host op over the whole grid; its IR pickles as before the shared lowering."""

    @pytest.mark.parametrize("build", [
        lambda: _conv1d_mapper(3, 13, 3, 1, 1, 1), lambda: _conv2d_mapper(3, (7, 6), (3, 3), (1, 1), (1, 1), (1, 1)),
    ], ids=["conv1d", "conv2d"])
    def test_shape_fields_are_distinct_equal_tuples(self, build):
        from mimarsinan.mapping.ir import ComputeOp

        mapper = build()
        mapper.perceptron.is_encoding_layer = True
        ir = map_model_to_ir(ModelRepresentation(mapper))
        (op,) = [n for n in ir.nodes if isinstance(n, ComputeOp)]
        assert op.params["input_shape"] == op.input_shape
        assert op.params["input_shape"] is not op.input_shape
        assert all(type(d) is int for d in op.input_shape)
