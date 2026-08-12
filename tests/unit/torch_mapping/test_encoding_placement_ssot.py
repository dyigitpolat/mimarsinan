"""The encoding-layer placement SSOT: one writer, applied at flow birth, honored end to end.

The reported defect: ``SimpleMLPBuilder`` baked ``is_encoding_layer=True``
unconditionally, and the only production caller that passed the configured
placement (``convert_torch_model``) is reachable only from ``TorchMappingStep``,
which ``applies_to`` the ``torch`` category alone. ``simple_mlp`` is the only
``native`` model, so ``encoding_layer_placement`` was a silent NO-OP for it: a
run configured ``offload`` deployed the SUBSUMED mapping and its static on-chip
gate measured the subsumed fraction (15.23% on the failing run) instead of the
~100% the requested placement would deploy.

The contract these tests pin:
  * ``mark_encoding_layers`` is the ONE writer of the placement decision and it
    STAMPS the resolved placement on the ``ModelRepresentation``;
  * placement is applied exactly once, at FLOW BIRTH — ``build_model`` for a
    builder that returns a flow, ``convert_torch_model`` for a torch module;
  * a consumer that is handed a flow whose placement was never resolved fails
    LOUD instead of silently measuring an unresolved marking.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.mapping.verification.onchip_fraction import estimate_onchip_fraction
from mimarsinan.models.builders import BUILDERS_REGISTRY, build_model
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.torch_mapping.encoding_layers import (
    UnresolvedEncodingPlacementError,
    mark_encoding_layers,
    resolved_encoding_placement,
)

INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10
MLP_CONFIG = {"mlp_width_1": 256, "mlp_width_2": 128, "base_activation": "ReLU"}


def _builder(model_type="simple_mlp", input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    return BUILDERS_REGISTRY[model_type](
        "cpu", input_shape, num_classes, {"target_tq": 32, "device": "cpu"}
    )


def _warm(model, input_shape=INPUT_SHAPE):
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    return model


def _marked(model):
    return [p for p in model.get_perceptrons() if p.is_encoding_layer]


class TestMarkEncodingLayersIsTheOneWriter:
    """``mark_encoding_layers`` records the decision it made, on the repr."""

    def test_marking_stamps_the_resolved_placement(self):
        model = _builder().build(MLP_CONFIG)
        repr_ = model.get_mapper_repr()
        assert resolved_encoding_placement(repr_) is None

        mark_encoding_layers(repr_, placement="offload")
        assert resolved_encoding_placement(repr_) == "offload"

        mark_encoding_layers(repr_, placement="subsume")
        assert resolved_encoding_placement(repr_) == "subsume"

    def test_a_rejected_placement_stamps_nothing(self):
        repr_ = _builder().build(MLP_CONFIG).get_mapper_repr()
        with pytest.raises(ValueError):
            mark_encoding_layers(repr_, placement="somewhere_else")
        assert resolved_encoding_placement(repr_) is None

    def test_converted_torch_flow_carries_the_stamp(self):
        from mimarsinan.models.deep_mlp import DeepMLP

        flow = convert_torch_model(
            DeepMLP(input_shape=(1, 8, 8), num_classes=4, depth=3, width=16),
            (1, 8, 8),
            4,
            encoding_layer_placement="offload",
        )
        assert resolved_encoding_placement(flow.get_mapper_repr()) == "offload"


class TestBuildersTakeNoPlacementDecision:
    """A builder returns an UNRESOLVED flow: baking a placement is the defect."""

    def test_simple_mlp_build_marks_nothing(self):
        model = _warm(_builder().build(MLP_CONFIG))
        assert _marked(model) == [], (
            "the builder baked an encoding mark — placement is the config's "
            "decision, applied once at flow birth by build_model"
        )
        assert resolved_encoding_placement(model.get_mapper_repr()) is None


class TestBuildModelHonorsThePlacement:
    """``build_model`` is the flow-birth site for a builder that returns a flow."""

    @pytest.mark.parametrize("placement,expected_marks", [("subsume", 1), ("offload", 0)])
    def test_native_flow_marking_follows_the_config(self, placement, expected_marks):
        model = _warm(
            build_model(_builder(), MLP_CONFIG, encoding_placement=placement)
        )
        assert len(_marked(model)) == expected_marks
        assert resolved_encoding_placement(model.get_mapper_repr()) == placement

    def test_subsume_marks_the_input_side_perceptron(self):
        model = _warm(build_model(_builder(), MLP_CONFIG, encoding_placement="subsume"))
        assert _marked(model) == [model.get_perceptrons()[0]]

    def test_a_torch_builder_is_left_for_conversion_to_resolve(self):
        """A torch module has no flow yet; ``build_model`` must not invent one."""
        model = build_model(
            _builder("deep_mlp", (1, 8, 8), 4),
            {"depth": 3, "width": 16},
            encoding_placement="offload",
        )
        assert not hasattr(model, "get_mapper_repr")


class TestTheReportedDefect:
    """The failing run: ``simple_mlp`` + ``offload`` measured the SUBSUMED split."""

    def _fraction(self, placement):
        model = _warm(build_model(_builder(), MLP_CONFIG, encoding_placement=placement))
        return estimate_onchip_fraction(
            model, INPUT_SHAPE, NUM_CLASSES, encoding_placement=placement
        )

    def test_offload_puts_the_encoder_on_chip(self):
        est = self._fraction("offload")
        assert est.host == 0
        assert est.fraction == pytest.approx(1.0)

    def test_subsume_keeps_the_encoder_host_side(self):
        est = self._fraction("subsume")
        assert est.host > 0
        assert est.fraction < 0.5

    def test_the_two_placements_no_longer_measure_the_same(self):
        """The no-op's signature: both placements reported one fraction."""
        assert self._fraction("offload").fraction > self._fraction("subsume").fraction


class TestTheDeployedMappingHonorsThePlacement:
    """The knob must move the encoder in the MAPPED IR, not only in the gate.

    A gate that reads right while the mapping deploys the other placement is the
    same defect wearing a different hat.
    """

    def _ir(self, placement):
        from mimarsinan.mapping.ir_mapping_class import IRMapping
        from mimarsinan.mapping.support.per_source_scales import (
            compute_per_source_scales,
        )

        model = _warm(build_model(_builder(), MLP_CONFIG, encoding_placement=placement))
        repr_ = model.get_mapper_repr()
        repr_.assign_perceptron_indices()
        compute_per_source_scales(repr_)
        return IRMapping(
            q_max=127, firing_mode="Default", max_axons=None, max_neurons=None,
            allow_coalescing=False, hardware_bias=True,
        ).map(repr_)

    def test_subsume_deploys_the_encoder_as_a_host_compute_op(self):
        ir = self._ir("subsume")
        assert [op.op_type for op in ir.get_compute_ops()] == ["Perceptron"]

    def test_offload_deploys_the_encoder_as_a_neural_core(self):
        ir = self._ir("offload")
        assert ir.get_compute_ops() == []
        # the 784-wide input layer is now a core, not a host op
        assert any(core.core_matrix.shape[0] == 784 for core in ir.get_neural_cores())

    def test_offload_maps_one_more_core_than_subsume(self):
        assert len(self._ir("offload").get_neural_cores()) == (
            len(self._ir("subsume").get_neural_cores()) + 1
        )


class TestTheGateMeasuresWhatWillDeploy:
    """A flow whose placement was never resolved must not be silently measured."""

    def test_unresolved_flow_is_refused(self):
        model = _warm(_builder().build(MLP_CONFIG))  # no build_model: unresolved
        with pytest.raises(UnresolvedEncodingPlacementError) as excinfo:
            estimate_onchip_fraction(
                model, INPUT_SHAPE, NUM_CLASSES, encoding_placement="offload"
            )
        assert "encoding_layer_placement" in str(excinfo.value)

    def test_a_flow_resolved_to_a_different_placement_is_refused(self):
        model = _warm(build_model(_builder(), MLP_CONFIG, encoding_placement="subsume"))
        with pytest.raises(UnresolvedEncodingPlacementError) as excinfo:
            estimate_onchip_fraction(
                model, INPUT_SHAPE, NUM_CLASSES, encoding_placement="offload"
            )
        message = str(excinfo.value)
        assert "'subsume'" in message and "'offload'" in message

    def test_the_gate_does_not_rewrite_the_marking_it_measures(self):
        """The late gate runs AFTER the negative-boundary subsume-forward policy;
        re-marking there would erase host placements the deployment depends on."""
        model = _warm(build_model(_builder(), MLP_CONFIG, encoding_placement="offload"))
        extra = model.get_perceptrons()[1]
        extra.is_encoding_layer = True  # stand-in for a negative-boundary subsume

        estimate_onchip_fraction(
            model, INPUT_SHAPE, NUM_CLASSES, encoding_placement="offload"
        )
        assert extra.is_encoding_layer is True
