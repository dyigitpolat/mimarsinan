"""Torchvision's Swin and Swin V2 at toy size, converted end to end (coverage W7 and W8).

On the base the shifted blocks' attention (``shifted_window_attention``) stayed opaque: its body builds the attention
mask with ``x.new_zeros`` and ``setitem`` (Swin) and zeroes a slice of a clone of ``qkv_bias`` in place (Swin V2), and
once inlined, the window partition merged the batch into the window axis (``BatchAxisLayoutError``). Now every leaf
call is inlined (the writes computed out of place, the mask a batch-free value), every merged region is rewritten
batch-leading, and both stages agree at every batch, in torchvision's own code: windows that tile the map, windows of 3
that pad it, and a last stage whose window covers the whole map (the shift set to 0 by Swin's own branch on extents).
"""

from __future__ import annotations

import pytest
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformer, SwinTransformerBlockV2

from mimarsinan.torch_mapping.converter_handlers.symbolic.declared_operators import OPAQUE_BODY
from mimarsinan.torch_mapping.converter_handlers.symbolic.purity import program_purity
from mimarsinan.torch_mapping.torch_graph_tracer import trace_model
from operand_completeness.harness import CONTRACTS, assert_converted_exact, assert_value_program_exact, native

SHAPE = (3, 16, 16)


def swin():
    return SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[2, 2], num_heads=[2, 2], window_size=[2, 2],
                           num_classes=5)


def swin_v2():
    return SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[2, 2], num_heads=[2, 2], window_size=[2, 2],
                           num_classes=5, block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2)


def swin_padded():
    return SwinTransformer(patch_size=[4, 4], embed_dim=8, depths=[2, 2], num_heads=[2, 2], window_size=[3, 3],
                           num_classes=5)


FAMILY = {"swin": swin, "swin_v2": swin_v2, "swin_padded": swin_padded}


@pytest.mark.parametrize("name", sorted(FAMILY))
def test_every_leaf_call_of_swin_is_inlined_and_the_program_is_pure(name):
    gm = trace_model(native(FAMILY[name]), SHAPE)
    kept = {node.name: node.meta[OPAQUE_BODY] for node in gm.graph.nodes if OPAQUE_BODY in node.meta}
    assert not kept, kept
    assert program_purity(gm).pure, program_purity(gm).impurity


@pytest.mark.timeout(240)
@pytest.mark.parametrize("name", sorted(FAMILY))
@pytest.mark.parametrize("contract", CONTRACTS)
def test_swin_converts_exactly_at_every_batch(name, contract):
    flow = assert_converted_exact(native(FAMILY[name]), SHAPE, contract, batches=(1, 2, 3, 5))
    assert flow.shape_impurity is None


@pytest.mark.timeout(240)
@pytest.mark.parametrize("name", sorted(FAMILY))
def test_the_value_program_of_swin_is_exact(name):
    assert_value_program_exact(native(FAMILY[name]), SHAPE, batches=(2, 3))
