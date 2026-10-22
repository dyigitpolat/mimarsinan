"""A deferred ``LayoutSourceView`` answers numpy's integer-array (advanced) indexing.

The shape-only plane hands a host op's outputs on as a deferred view. A conv
with zero padding gathers its patches from that view by integer-array indexing
(``conv_helpers.gather_conv_patches``), which the view refused with a TypeError,
so every unpadded conv fed by a host op failed the shape-only layout: the 1x1
projection shortcut after a residual add under the value-domain contract, and the
first 1x1 conv of every pre-activation block (BN-ReLU on host) under both
contracts. A padded conv never met the refusal because ``np.pad`` materializes
the view first; an advanced index now materializes it the same way.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_source_view import LayoutSourceView
from mimarsinan.mapping.mappers.conv_helpers import gather_conv_patches
from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.models.cifar_models import CifarResNet
from mimarsinan.torch_mapping.converter import convert_torch_model


def _cells(arr) -> list:
    return [(s.node_id, s.index) for s in np.asarray(arr, dtype=object).flat]


def test_advanced_index_equals_the_materialized_array():
    view = LayoutSourceView.from_producer(producer_node_id=3, shape=(2, 3, 3))
    index = np.broadcast_arrays(
        np.arange(2)[:, None, None], np.array([0, 2])[None, :, None], np.array([1, 2])[None, None, :],
    )
    got = view[tuple(index)]
    assert np.shape(got) == (2, 2, 2)
    assert _cells(got) == _cells(np.asarray(view)[tuple(index)])


def test_basic_indexing_stays_deferred():
    view = LayoutSourceView.from_producer(producer_node_id=3, shape=(2, 3, 3))
    assert isinstance(view[0], LayoutSourceView) and view[0].shape == (3, 3)
    assert isinstance(view[:, 1:], LayoutSourceView) and view[:, 1:].shape == (2, 2, 3)


@pytest.mark.parametrize("kernel,stride", [((1, 1), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (2, 2))])
def test_unpadded_patch_gather_over_a_deferred_view(kernel, stride):
    view = LayoutSourceView.from_producer(producer_node_id=5, shape=(3, 4, 4))
    out = tuple((4 - k) // s + 1 for k, s in zip(kernel, stride))
    got = gather_conv_patches(view, kernel, stride, (1, 1), out)
    expected = gather_conv_patches(np.asarray(view), kernel, stride, (1, 1), out)
    assert np.shape(got) == expected.shape
    assert _cells(got) == _cells(expected)


@pytest.mark.parametrize("packaging", [MVM_PACKAGING, SPIKING_PACKAGING], ids=["mvm", "spiking"])
def test_projection_resnet_lays_out_on_the_shape_only_plane(packaging):
    """One softcore per IR neural core, on the plane that previously raised."""
    torch.manual_seed(0)
    model = CifarResNet((3, 8, 8), 4, blocks_per_stage=1, base_width=4).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        flow = convert_torch_model(model, (3, 8, 8), 4, packaging=packaging, strict=False)
    softcores = LayoutIRMapping(max_axons=256, max_neurons=256).collect_layout_softcores(flow.get_mapper_repr())
    flow.get_mapper_repr().clear_ir_caches()
    ir = IRMapping(q_max=1.0, firing_mode="Default", max_axons=256, max_neurons=256).map(flow.get_mapper_repr())
    cores = ir.get_neural_cores()
    assert len(softcores) == len(cores) > 0
    assert sorted((sc.input_count, sc.output_count) for sc in softcores) == sorted(
        (int(c.get_input_count()), int(c.get_output_count())) for c in cores
    )
