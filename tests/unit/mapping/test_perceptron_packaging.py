"""Perceptron packaging: the converter decides at conversion time.

MM+ + BN? + ACT → Perceptron + PerceptronMapper → NeuralCore.
No activation → ModuleComputeMapper → ComputeOp.

The decision is encoded by the mapper TYPE — no runtime activation checks.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.mapping.mappers.structural import (
    InputMapper, EinopsRearrangeMapper,
)
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper, ModuleComputeMapper
from mimarsinan.mapping.mappers.conv import Conv2DPerceptronMapper
from mimarsinan.mapping.mappers.leading_dim import Ensure2DMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


def _map_to_ir(mapper_repr, input_shape):
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(
        q_max=1, firing_mode="TTFS", max_axons=1024, max_neurons=1024,
    )
    return ir_mapping.map(mapper_repr)


class TestModuleComputeMapperCreatesComputeOp:
    """ModuleComputeMapper → ComputeOp (host-side)."""

    def test_linear_creates_compute_op(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        linear = nn.Linear(16, 4)
        mapper = ModuleComputeMapper(Ensure2DMapper(flat), linear, name="test_linear")
        repr_ = ModelRepresentation(mapper)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))
        compute_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp)]
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]

        assert len(compute_ops) >= 1, "ModuleComputeMapper should create ComputeOps"
        assert len(neural_cores) == 0, "ModuleComputeMapper should NOT create NeuralCores"


class TestPerceptronMapperCreatesNeuralCore:
    """PerceptronMapper always creates NeuralCores (any nonlinear activation)."""

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_creates_neural_core(self, act_name):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(),
                       base_activation_name=act_name)
        fc = PerceptronMapper(Ensure2DMapper(flat), p)
        repr_ = ModelRepresentation(fc)

        ir_graph = _map_to_ir(repr_, (1, 4, 4))
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
        linear_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp) and n.op_type == "linear"]

        assert len(neural_cores) >= 1, f"{act_name} FC should create NeuralCores"
        assert len(linear_ops) == 0, f"{act_name} FC should NOT create linear ComputeOps"


# ---------------------------------------------------------------------------
# Mapper eligibility contract tests
# ---------------------------------------------------------------------------

class TestPerceptronMapperEligibilityContract:
    """PerceptronMapper.owned_perceptron_groups() always returns the perceptron."""

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_always_returns_perceptron(self, act_name):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        p = Perceptron(4, 16, normalization=nn.Identity(), base_activation_name=act_name)
        mapper = PerceptronMapper(Ensure2DMapper(flat), p)
        groups = mapper.owned_perceptron_groups()
        assert len(groups) == 1 and p in groups[0]


class TestModuleComputeMapperEligibilityContract:
    """ModuleComputeMapper.owned_perceptron_groups() returns []."""

    def test_returns_empty(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        linear = nn.Linear(16, 4)
        mapper = ModuleComputeMapper(Ensure2DMapper(flat), linear, name="test")
        assert mapper.owned_perceptron_groups() == []


class TestConv2DPerceptronMapperEligibilityContract:
    """Conv2DPerceptronMapper.owned_perceptron_groups() always returns the perceptron."""

    @pytest.mark.parametrize("act_name", ["ReLU", "LeakyReLU", "GELU"])
    def test_always_returns_perceptron(self, act_name):
        inp = InputMapper((1, 4, 4))
        conv = Conv2DPerceptronMapper(
            inp, in_channels=1, out_channels=2,
            kernel_size=2, stride=2, padding=0,
            bias=False, use_batchnorm=False,
            base_activation_name=act_name,
        )
        groups = conv.owned_perceptron_groups()
        assert len(groups) == 1 and conv.perceptron in groups[0]


class TestGetPerceptronsExcludesComputeMappers:
    """ModelRepresentation.get_perceptrons() returns only PerceptronMapper perceptrons.
    ModuleComputeMapper is invisible to the pipeline.
    """

    def test_compute_mapper_not_in_get_perceptrons(self):
        inp = InputMapper((1, 4, 4))
        flat = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        linear = nn.Linear(16, 8)
        m1 = ModuleComputeMapper(Ensure2DMapper(flat), linear, name="compute")
        p_relu = Perceptron(4, 8, normalization=nn.Identity(), base_activation_name="ReLU")
        m2 = PerceptronMapper(m1, p_relu)
        repr_ = ModelRepresentation(m2)

        perceptrons = repr_.get_perceptrons()
        assert p_relu in perceptrons, "ReLU perceptron must be in get_perceptrons()"
        assert len(perceptrons) == 1, f"Only ReLU perceptron expected, got {len(perceptrons)}"

    def test_torch_mapped_mixed_model(self):
        """Torch-mapped model with Conv(no act)+FC(LeakyReLU): only FC in get_perceptrons()."""
        from mimarsinan.torch_mapping.converter import convert_torch_model

        class ConvNoActFCLeakyReLU(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0)
                self.bn = nn.BatchNorm2d(4)
                self.fc = nn.Linear(4 * 4 * 4, 8)
                self.act = nn.LeakyReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = x.flatten(1)
                x = self.act(self.fc(x))
                return x

        torch.manual_seed(0)
        model = ConvNoActFCLeakyReLU()
        model.eval()
        with torch.no_grad():
            model(torch.randn(2, 1, 8, 8))

        supermodel = convert_torch_model(model, input_shape=(1, 8, 8), num_classes=8)
        perceptrons = supermodel.get_perceptrons()

        assert len(perceptrons) >= 1, "At least one perceptron expected"
        # No Identity-activated perceptrons should exist anymore
        act_names = [type(p.base_activation).__name__ for p in perceptrons]
        assert "Identity" not in act_names, (
            f"No Identity-activated perceptrons should exist. Found: {act_names}"
        )
