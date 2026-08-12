"""A value-domain (MVM) flow carries an EXPLICIT not-applicable placement stamp.

``encoding_layer_placement`` is registry ``domain="event"``: a value-domain
deployment cannot even author the key, because a value core consumes values and
has no spike-train encoder to place. Before this pin, ``convert_torch_model``
skipped the marking for a value-domain packaging and left the graph with NO
stamp — indistinguishable from "nobody ever applied the configured placement",
so the placement guard refused every MVM run at soft-core mapping (t0_41,
t0_44, t1_09, t1_11, t1_12 — a whole deployment family).

The stamp must stay unambiguous: not-applicable is its own value, never a
silent ``None``.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import (
    MVM_PACKAGING,
    SPIKING_PACKAGING,
    packaging_contract_for,
)
from mimarsinan.models.lenet5 import LeNet5
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import (
    PLACEMENT_NOT_APPLICABLE,
    UnresolvedEncodingPlacementError,
    require_resolved_encoding_placement,
    resolved_encoding_placement,
)

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10


def _mvm_flow(placement: str = "subsume"):
    model = LeNet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    return convert_torch_model(
        model, INPUT_SHAPE, NUM_CLASSES, device="cpu",
        encoding_layer_placement=placement, packaging=MVM_PACKAGING,
    )


def _spiking_flow(placement: str = "subsume"):
    model = LeNet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    return convert_torch_model(
        model, INPUT_SHAPE, NUM_CLASSES, device="cpu",
        encoding_layer_placement=placement, packaging=SPIKING_PACKAGING,
    )


class TestValueDomainStamp:
    def test_mvm_flow_is_stamped_not_applicable(self):
        flow = _mvm_flow()
        resolved = resolved_encoding_placement(flow.get_mapper_repr())
        assert resolved == PLACEMENT_NOT_APPLICABLE
        assert resolved is not None, "a silent None means two different things"

    def test_not_applicable_is_distinct_from_both_placements(self):
        assert PLACEMENT_NOT_APPLICABLE not in ("subsume", "offload")

    def test_mvm_flow_marks_no_encoder(self):
        """A value core has no encoder to place, so nothing is host-marked."""
        flow = _mvm_flow()
        assert not any(
            getattr(p, "is_encoding_layer", False) for p in flow.get_perceptrons()
        )

    @pytest.mark.parametrize("asked", ["subsume", "offload"])
    def test_the_gate_accepts_a_value_domain_flow_for_either_placement(self, asked):
        """The regression: this raised UnresolvedEncodingPlacementError, and the
        SCM validity gate refused every MVM deployment."""
        flow = _mvm_flow()
        require_resolved_encoding_placement(
            flow.get_mapper_repr(), asked, context="a value-domain consumer"
        )

    def test_the_scm_gate_path_runs_on_an_mvm_flow(self):
        from mimarsinan.mapping.verification.onchip_fraction import (
            estimate_onchip_fraction,
        )

        flow = _mvm_flow()
        est = estimate_onchip_fraction(
            flow, INPUT_SHAPE, NUM_CLASSES, encoding_placement="subsume",
        )
        assert est.total > 0
        assert est.fraction > 0.0


class TestEventDomainIsUnaffected:
    @pytest.mark.parametrize("placement", ["subsume", "offload"])
    def test_a_spiking_flow_still_stamps_the_real_placement(self, placement):
        flow = _spiking_flow(placement)
        assert resolved_encoding_placement(flow.get_mapper_repr()) == placement

    def test_a_spiking_flow_is_still_refused_under_the_other_placement(self):
        flow = _spiking_flow("subsume")
        with pytest.raises(UnresolvedEncodingPlacementError, match="offload"):
            require_resolved_encoding_placement(
                flow.get_mapper_repr(), "offload", context="a consumer"
            )

    def test_the_two_placements_still_differ_on_a_spiking_flow(self):
        """Sanity: the vehicle under test really is placement-sensitive, so the
        MVM acceptance above is not vacuous."""
        subsumed = _spiking_flow("subsume")
        offloaded = _spiking_flow("offload")
        marked = lambda f: sum(  # noqa: E731
            bool(getattr(p, "is_encoding_layer", False)) for p in f.get_perceptrons()
        )
        assert marked(subsumed) > marked(offloaded) == 0


class TestPackagingDispatchIsTheSSOT:
    """The stamp follows the packaging CONTRACT, not a hardcoded MVM check."""

    def test_a_value_domain_plan_resolves_to_a_value_domain_contract(self):
        plan = _plan(is_mvm=True, activation_quantization=False)
        assert packaging_contract_for(plan).is_value_domain

    def test_a_gridded_value_domain_contract_also_stamps_not_applicable(self):
        """``activation_bits`` swaps the boundary spec but not the domain."""
        plan = _plan(is_mvm=True, activation_quantization=True)
        packaging = packaging_contract_for(plan)
        assert packaging.boundary_is_gridded
        model = LeNet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
        flow = convert_torch_model(
            model, INPUT_SHAPE, NUM_CLASSES, device="cpu", packaging=packaging,
        )
        assert (
            resolved_encoding_placement(flow.get_mapper_repr())
            == PLACEMENT_NOT_APPLICABLE
        )


class _Plan:
    def __init__(self, is_mvm: bool, activation_quantization: bool):
        self.is_mvm = is_mvm
        self.activation_quantization = activation_quantization


def _plan(*, is_mvm: bool, activation_quantization: bool) -> _Plan:
    return _Plan(is_mvm, activation_quantization)


def test_a_bare_module_is_not_a_flow():
    """Guard-rail for the fixtures above: LeNet5 is an nn.Module, not a flow."""
    assert isinstance(LeNet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES), nn.Module)
    assert not hasattr(LeNet5(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES), "get_mapper_repr")
