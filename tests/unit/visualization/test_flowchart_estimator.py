"""Flowchart FC core estimates vs layout softcore collection."""

import torch.nn as nn

from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.visualization.softcore_flowchart_estimate import estimate_fc_cores


class TestFlowchartEstimator:
    def test_single_fc_matches_layout_softcore_count(self):
        inp = InputMapper((16,))
        p = Perceptron(
            32,
            16,
            normalization=nn.Identity(),
            base_activation_name="ReLU",
        )
        repr_ = ModelRepresentation(PerceptronMapper(inp, p))

        max_axons, max_neurons = 256, 256
        layout = LayoutIRMapping(max_axons=max_axons, max_neurons=max_neurons)
        layout_scs = layout.collect_layout_softcores(repr_)

        est = estimate_fc_cores(
            in_features=16,
            out_features=32,
            instances=1,
            has_bias=True,
            max_axons=max_axons,
            max_neurons=max_neurons,
        )
        assert est == len(layout_scs) == 1

    def test_coalescing_mode_core_count(self):
        """A wide fan-in with coalescing fuses N cores — the estimate counts them."""
        est = estimate_fc_cores(
            in_features=512,
            out_features=64,
            instances=1,
            has_bias=True,
            max_axons=64,
            max_neurons=64,
            allow_coalescing=True,
        )
        assert est > 1

    def test_wide_without_coalescing_is_unmappable(self):
        """Without coalescing capability a wide fan-in cannot be mapped (estimate 0)."""
        est = estimate_fc_cores(
            in_features=512,
            out_features=64,
            instances=1,
            has_bias=True,
            max_axons=64,
            max_neurons=64,
            allow_coalescing=False,
        )
        assert est == 0
