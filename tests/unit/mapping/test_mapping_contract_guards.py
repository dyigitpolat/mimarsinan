"""Contract guards and shared helpers hardened during the mapping type-grind.

Covers:
- ``Mapper.require_source_mapper``: explicit error when a source-consuming
  mapper was built without a source, and identity with ``source_mapper``
  when one is set.
- ``conv_helpers.pad_source_grid``: object-grid padding with OFF IRSources.
- ``NeuralCore.get_core_matrix``: explicit error when the core has neither
  an owned matrix nor a weight-bank reference.
- ``neural_core_to_soft_core``: explicit error on a dangling weight-bank id.
- ``quantize_ir_graph``: explicit error on a bankless, matrix-less core.
- ``ComputeOpMapper``: multi-input mapping refuses per-source ``None``
  input shapes with a clear error.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch.nn as nn

from mimarsinan.mapping.export.chip_quantize import quantize_ir_graph
from mimarsinan.mapping.ir import (
    IRGraph,
    IRSource,
    NeuralCore,
    neural_core_to_soft_core,
)
from mimarsinan.mapping.mappers.base import Mapper
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.conv_helpers import pad_source_grid
from mimarsinan.mapping.mappers.structural import InputMapper


def _src(specs):
    return np.array(
        [IRSource(node_id=nid, index=idx) for nid, idx in specs],
        dtype=object,
    )


class TestRequireSourceMapper:
    def test_returns_the_configured_source_mapper(self):
        source = InputMapper((1, 4))
        mapper = Mapper(source)
        assert mapper.require_source_mapper() is source
        assert mapper.require_source_mapper() is mapper.source_mapper

    def test_raises_value_error_when_no_source_is_set(self):
        mapper = Mapper()
        with pytest.raises(ValueError, match="requires a source mapper"):
            mapper.require_source_mapper()


class TestPadSourceGrid:
    def test_pads_2d_grid_with_off_sources(self):
        grid = _src([(-2, 0), (-2, 1)]).reshape(1, 2)
        padded = pad_source_grid(grid, ((0, 0), (1, 1)))

        assert padded.shape == (1, 4)
        assert padded[0, 0].is_off()
        assert padded[0, 3].is_off()
        assert padded[0, 1] is grid[0, 0]
        assert padded[0, 2] is grid[0, 1]

    def test_pads_3d_grid_along_spatial_axes_only(self):
        grid = np.empty((2, 1, 1), dtype=object)
        grid[0, 0, 0] = IRSource(node_id=-2, index=0)
        grid[1, 0, 0] = IRSource(node_id=-2, index=1)
        padded = pad_source_grid(grid, ((0, 0), (1, 1), (1, 1)))

        assert padded.shape == (2, 3, 3)
        assert padded[0, 1, 1] is grid[0, 0, 0]
        assert padded[1, 1, 1] is grid[1, 0, 0]
        off_flags = [
            padded[c, i, j].is_off()
            for c in range(2)
            for i in range(3)
            for j in range(3)
            if (i, j) != (1, 1)
        ]
        assert all(off_flags)


class TestGetCoreMatrixGuards:
    def test_neither_matrix_nor_bank_raises_value_error(self):
        core = NeuralCore(
            id=0, name="empty",
            input_sources=_src([(-2, 0)]),
            core_matrix=None,
        )
        graph = IRGraph(nodes=[core], output_sources=_src([(0, 0)]))
        with pytest.raises(ValueError, match="neither"):
            core.get_core_matrix(graph)


class TestNeuralCoreToSoftCoreGuards:
    def test_dangling_weight_bank_id_raises_key_error(self):
        core = NeuralCore(
            id=0, name="dangling",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]]),
            weight_bank_id=7,
        )
        graph = IRGraph(nodes=[core], output_sources=_src([(0, 0)]))
        with pytest.raises(KeyError, match="weight_bank_id=7"):
            neural_core_to_soft_core(core, graph=graph)


class TestQuantizeIrGraphGuards:
    def test_bankless_matrixless_core_raises_value_error(self):
        core = NeuralCore(
            id=0, name="empty",
            input_sources=_src([(-2, 0)]),
            core_matrix=None,
        )
        graph = IRGraph(nodes=[core], output_sources=_src([(0, 0)]))
        with pytest.raises(ValueError, match="neither"):
            quantize_ir_graph(graph, bits=8, weight_quantization=True)


class TestComputeOpMapperMultiInputShapes:
    def test_none_entry_in_input_shapes_raises_value_error(self):
        a = InputMapper((1, 3))
        b = InputMapper((1, 3))
        mapper = ComputeOpMapper(
            [a, b],
            nn.Identity(),
            input_shapes=[None, (3,)],
            output_shape=(3,),
        )
        with pytest.raises(ValueError, match="input_shapes"):
            mapper._resolved_multi_input_shapes(
                source_arrays=[np.empty((3,), dtype=object)] * 2,
                module=nn.Identity(),
            )
