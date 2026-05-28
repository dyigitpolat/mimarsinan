"""Boundary-IR exemption layers must be built once per PruningTuner run."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.tuning.tuners.pruning import pruning_tuner_masks


class _StubFlow(nn.Module):
    def __init__(self):
        super().__init__()
        p1 = Perceptron(
            output_channels=8, input_features=8,
            normalization=nn.Identity(), base_activation_name="ReLU",
        )
        p2 = Perceptron(
            output_channels=4, input_features=8,
            normalization=nn.Identity(), base_activation_name="ReLU",
        )
        self._perceptrons = nn.ModuleList([p1, p2])
        inp = InputMapper((8,))
        m1 = PerceptronMapper(inp, p1)
        m2 = PerceptronMapper(m1, p2)
        self._mapper_repr = ModelRepresentation(m2)

    def get_perceptrons(self):
        return list(self._perceptrons)

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


def _make_tuner_stub():
    model = _StubFlow()
    pipeline = SimpleNamespace(config={"weight_bits": 8, "firing_mode": "Default"})
    return SimpleNamespace(model=model, pipeline=pipeline)


class TestBoundaryExemptionCache:
    def test_repeated_calls_build_ir_only_once(self):
        tuner = _make_tuner_stub()
        sentinel = ({0: frozenset()}, {0: frozenset()})

        with patch.object(
            pruning_tuner_masks, "build_boundary_ir_graph",
            return_value=MagicMock(),
        ) as build_mock, patch.object(
            pruning_tuner_masks, "compute_perceptron_io_exemption_indices",
            return_value=sentinel,
        ) as exempt_mock:
            r1 = pruning_tuner_masks._boundary_exemption_layers(tuner)
            r2 = pruning_tuner_masks._boundary_exemption_layers(tuner)
            r3 = pruning_tuner_masks._boundary_exemption_layers(tuner)

        assert r1 is sentinel
        assert r2 is sentinel
        assert r3 is sentinel
        assert build_mock.call_count == 1, (
            f"build_boundary_ir_graph must be called once; got {build_mock.call_count}"
        )
        assert exempt_mock.call_count == 1

    def test_different_tuners_get_separate_caches(self):
        t1 = _make_tuner_stub()
        t2 = _make_tuner_stub()
        with patch.object(
            pruning_tuner_masks, "build_boundary_ir_graph",
            return_value=MagicMock(),
        ) as build_mock, patch.object(
            pruning_tuner_masks, "compute_perceptron_io_exemption_indices",
            return_value=({}, {}),
        ):
            pruning_tuner_masks._boundary_exemption_layers(t1)
            pruning_tuner_masks._boundary_exemption_layers(t2)
            pruning_tuner_masks._boundary_exemption_layers(t1)
            pruning_tuner_masks._boundary_exemption_layers(t2)
        # one build per tuner
        assert build_mock.call_count == 2

    def test_invalidate_forces_rebuild(self):
        tuner = _make_tuner_stub()
        with patch.object(
            pruning_tuner_masks, "build_boundary_ir_graph",
            return_value=MagicMock(),
        ) as build_mock, patch.object(
            pruning_tuner_masks, "compute_perceptron_io_exemption_indices",
            return_value=({}, {}),
        ):
            pruning_tuner_masks._boundary_exemption_layers(tuner)
            pruning_tuner_masks._invalidate_boundary_cache(tuner)
            pruning_tuner_masks._boundary_exemption_layers(tuner)
        assert build_mock.call_count == 2
