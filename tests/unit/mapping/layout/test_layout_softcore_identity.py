"""``LayoutSoftCoreSpec`` carries the two identities the mapper already knows.

Before W5.2 the mapper's ``_sc_idx_to_bank_id`` / ``_sc_idx_to_perceptron_index``
side-tables died at the ``collect_layout_softcores`` boundary: every downstream
consumer (search, wizard, the agent introspection surface) was bank-blind and had
to recover layer identity by splitting core-name strings. These pin that the
identities survive the boundary on BOTH emission paths — bank-backed conv
positions (``add_shared_neural_core``) and owned-weight fc cores
(``add_neural_core``) — and that the full ``IRMapping`` path agrees.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def _conv_and_fc_repr():
    """One model exercising both emission paths: a conv (banks) then an fc (owned)."""
    from mimarsinan.mapping.mappers.conv2d_mapper import Conv2DPerceptronMapper
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.mappers.structural import (
        EinopsRearrangeMapper,
        InputMapper,
    )
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

    torch.manual_seed(7)
    conv = Conv2DPerceptronMapper(
        InputMapper((1, 4, 4)),
        in_channels=1, out_channels=2,
        kernel_size=2, stride=2, padding=0,
        bias=True, use_batchnorm=False,
        base_activation_name="Identity",
    )
    flat = EinopsRearrangeMapper(conv, "... c h w -> ... (c h w)")
    head = PerceptronMapper(
        flat,
        Perceptron(3, 8, normalization=nn.Identity(), base_activation_name="ReLU"),
    )
    repr_ = ModelRepresentation(head)
    compute_per_source_scales(repr_)
    repr_.assign_perceptron_indices()
    return repr_


def _collect(repr_, **kwargs):
    layout = LayoutIRMapping(max_axons=256, max_neurons=256, **kwargs)
    return layout.collect_layout_softcores(repr_)


class TestFieldsExist:
    def test_defaults_are_none_so_shape_only_callers_stay_valid(self):
        spec = LayoutSoftCoreSpec(input_count=4, output_count=2)
        assert spec.bank_id is None
        assert spec.perceptron_index is None


class TestBankBackedConvPath:
    def test_bank_id_survives_with_its_sharing_degree(self):
        softcores = _collect(_conv_and_fc_repr())
        assert softcores
        banked = [sc for sc in softcores if sc.bank_id is not None]
        assert banked, "conv positions are bank-backed; none carried a bank_id"

        # A conv maps ONE bank shared by every spatial position: the sharing
        # degree is exactly what the introspection surface must be able to see.
        by_bank: dict[int, int] = {}
        for sc in banked:
            key = int(sc.bank_id or 0)
            by_bank[key] = by_bank.get(key, 0) + 1
        assert max(by_bank.values()) > 1, by_bank

    def test_same_bank_means_same_source_perceptron(self):
        banked = [sc for sc in _collect(_conv_and_fc_repr()) if sc.bank_id is not None]
        assert all(sc.perceptron_index is not None for sc in banked)
        for bank_id in {sc.bank_id for sc in banked}:
            identities = {
                sc.perceptron_index for sc in banked if sc.bank_id == bank_id
            }
            assert len(identities) == 1, (bank_id, identities)


class TestOwnedWeightFcPath:
    def test_fc_cores_declare_no_bank_but_keep_their_identity(self):
        softcores = _collect(_conv_and_fc_repr())
        owned = [sc for sc in softcores if sc.bank_id is None]
        assert owned, "the head Perceptron is owned-weight, not bank-backed"
        assert all(sc.perceptron_index is not None for sc in owned)

    def test_the_two_layers_are_distinguishable_without_reading_names(self):
        softcores = _collect(_conv_and_fc_repr())
        conv_ids = {
            sc.perceptron_index for sc in softcores if sc.bank_id is not None
        }
        fc_ids = {sc.perceptron_index for sc in softcores if sc.bank_id is None}
        assert conv_ids and fc_ids
        assert conv_ids.isdisjoint(fc_ids)


class TestFullIrPathAgrees:
    def test_ir_mapping_emits_the_same_identities(self):
        """``IRMapping`` subclasses the layout mapper; the identities must not fork."""
        from mimarsinan.mapping.ir_mapping_class import IRMapping

        layout_scs = _collect(_conv_and_fc_repr())
        irm = IRMapping(
            q_max=1, firing_mode="Default", max_axons=256, max_neurons=256,
        )
        irm.map(_conv_and_fc_repr())
        full_scs = irm.layout_softcores

        assert len(layout_scs) == len(full_scs)
        assert [sc.bank_id for sc in layout_scs] == [sc.bank_id for sc in full_scs]
        assert [sc.perceptron_index for sc in layout_scs] == [
            sc.perceptron_index for sc in full_scs
        ]

    def test_bank_id_is_the_id_the_ir_core_references(self):
        from mimarsinan.mapping.ir_mapping_class import IRMapping

        irm = IRMapping(
            q_max=1, firing_mode="Default", max_axons=256, max_neurons=256,
        )
        ir_graph = irm.map(_conv_and_fc_repr())
        cores = ir_graph.get_neural_cores()
        specs = irm.layout_softcores
        assert len(cores) == len(specs)
        assert [sc.bank_id for sc in specs] == [
            core.weight_bank_id for core in cores
        ]
        assert [sc.perceptron_index for sc in specs] == [
            core.perceptron_index for core in cores
        ]
