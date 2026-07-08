"""The structural non-negativity predicate: which nodes absorb a signed range."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.support.value_domain import (
    node_absorbs_negative_values,
    produces_nonnegative_values,
)
from mimarsinan.models.nn.activations import LIFActivation, LeakyGradReLU
from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation
from mimarsinan.models.nn.decorators.clamp_quantize import ClampDecorator
from mimarsinan.models.nn.layers import TransformedActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class TestNonNegativeActivations:
    """A node absorbs a signed input range iff its output is >= 0 for EVERY
    input — a structural guarantee, never a calibration observation."""

    def test_relu_family_is_nonnegative(self):
        assert produces_nonnegative_values(LeakyGradReLU())
        assert produces_nonnegative_values(nn.ReLU())
        assert produces_nonnegative_values(nn.ReLU6())

    def test_spiking_activations_are_nonnegative(self):
        """Spike rates / spike-time-decoded values are non-negative by construction."""
        assert produces_nonnegative_values(
            LIFActivation(T=4, activation_scale=torch.tensor(1.0))
        )
        assert produces_nonnegative_values(
            TTFSCycleActivation(T=4, activation_scale=torch.tensor(1.0))
        )

    def test_signed_activations_are_not_nonnegative(self):
        assert not produces_nonnegative_values(nn.Identity())
        assert not produces_nonnegative_values(nn.GELU())
        assert not produces_nonnegative_values(nn.LeakyReLU())
        assert not produces_nonnegative_values(nn.LayerNorm(4))
        assert not produces_nonnegative_values(None)

    def test_a_nonnegative_clamp_decorator_forces_the_range(self):
        signed = TransformedActivation(nn.Identity(), [])
        assert not produces_nonnegative_values(signed)
        clamped = TransformedActivation(
            nn.Identity(),
            [ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))],
        )
        assert produces_nonnegative_values(clamped)

    def test_a_negative_clamp_floor_does_not_force_the_range(self):
        clamped = TransformedActivation(
            nn.Identity(),
            [ClampDecorator(torch.tensor(-1.0), torch.tensor(1.0))],
        )
        assert not produces_nonnegative_values(clamped)

    def test_a_transformed_relu_stays_nonnegative_through_its_decorators(self):
        act = TransformedActivation(LeakyGradReLU(), [])
        assert produces_nonnegative_values(act)

    def test_a_perceptron_answers_through_its_activation(self):
        relu = Perceptron(4, 4, base_activation_name="ReLU")
        assert produces_nonnegative_values(relu)
        gelu = Perceptron(4, 4, base_activation_name="GELU")
        assert not produces_nonnegative_values(gelu)


class TestNodeAbsorption:
    """The mapper-graph view of the same question: perceptron nodes answer
    through their activation, host ComputeOps through their module, and a
    structural node is sign-transparent (it never absorbs)."""

    def test_perceptron_node_answers_through_its_activation(self):
        class _Node:
            perceptron = Perceptron(4, 4, base_activation_name="ReLU")

        assert node_absorbs_negative_values(_Node())

    def test_compute_op_node_answers_through_its_module(self):
        from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
        from mimarsinan.mapping.mappers.structural import InputMapper

        src = InputMapper((4,))
        assert not node_absorbs_negative_values(
            ComputeOpMapper(src, nn.LayerNorm(4), input_shape=(4,))
        )
        assert node_absorbs_negative_values(
            ComputeOpMapper(src, nn.ReLU(), input_shape=(4,))
        )

    def test_structural_nodes_never_absorb(self):
        from mimarsinan.mapping.mappers.structural import InputMapper

        assert not node_absorbs_negative_values(InputMapper((4,)))


@pytest.mark.parametrize("floor,expected", [(0.0, True), (0.5, True), (-1e-9, False)])
def test_clamp_floor_boundary(floor, expected):
    act = TransformedActivation(
        nn.Identity(), [ClampDecorator(torch.tensor(floor), torch.tensor(9.0))],
    )
    assert produces_nonnegative_values(act) is expected
