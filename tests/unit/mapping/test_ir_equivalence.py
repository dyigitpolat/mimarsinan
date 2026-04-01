"""IR equivalence: raw perceptron forward vs SpikingUnifiedCoreFlow (TTFS) forward.

Check that the IR graph matches the model's forward pass when activations are
plain ReLU (no TransformedActivation decorators), thresholds = 1.0, and
parameter_scale = 1.0.  This isolates the IR construction / weight transfer
from the adaptation pipeline.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from conftest import (
    MockPipeline,
    default_config,
    platform_constraints,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    PerceptronMapper,
    EinopsRearrangeMapper,
    ModelRepresentation,
)
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import ir_graph_to_soft_core_mapping
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.layers import TransformedActivation, SavedTensorDecorator


class _RawMLPModel(nn.Module):
    """Minimal 2-perceptron model with plain ReLU activations (no decorators)."""

    def __init__(self, input_shape=(1, 8, 8), num_classes=4):
        super().__init__()
        in_features = 1
        for d in input_shape:
            in_features *= d

        self.p1 = Perceptron(16, in_features, normalization=nn.Identity())
        self.p2 = Perceptron(num_classes, 16, normalization=nn.Identity())

        inp = InputMapper(input_shape)
        rearrange = EinopsRearrangeMapper(inp, "... c h w -> ... (c h w)")
        m1 = PerceptronMapper(rearrange, self.p1)
        m2 = PerceptronMapper(m1, self.p2)
        self._mapper_repr = ModelRepresentation(m2)

    def get_perceptrons(self):
        return [self.p1, self.p2]

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


def test_ir_forward_matches_raw_model():
    """SpikingUnifiedCoreFlow logits must match the raw model (no decorators)."""
    torch.manual_seed(42)
    input_shape = (1, 8, 8)
    num_classes = 4

    model = _RawMLPModel(input_shape, num_classes)
    model.eval()

    # Capture per-input scales (needed for IR mapping)
    for p in model.get_perceptrons():
        p.input_activation = TransformedActivation(p.input_activation, [])
        p.input_activation.decorate(SavedTensorDecorator())
    model(torch.randn(10, *input_shape))
    for p in model.get_perceptrons():
        dec = p.input_activation.pop_decorator()
        p.set_input_activation_scale(dec.latest_input.max().item())

    # Set activation_scale for inter-layer thresholds
    model.p1.set_activation_scale(model.p2.input_activation_scale.item())
    model.p2.set_activation_scale(1.0)

    # Set parameter_scale = 1.0 (no quantization)
    for p in model.get_perceptrons():
        p.set_parameter_scale(1.0)

    repr = model.get_mapper_repr()
    compute_per_source_scales(repr)

    ir_mapping = IRMapping(
        q_max=1,
        firing_mode="TTFS",
        max_axons=1024,
        max_neurons=1024,
        allow_coalescing=False,
    )
    ir_graph = ir_mapping.map(repr)

    flow = SpikingUnifiedCoreFlow(
        input_shape,
        ir_graph,
        32,
        nn.Identity(),
        "TTFS",
        "TTFS",
        "<=",
        spiking_mode="ttfs",
    )
    flow.eval()

    x = torch.rand(4, *input_shape)
    with torch.no_grad():
        model_out = model(x)
        flow_out = flow(x)

    assert model_out.shape == flow_out.shape
    diff = (model_out - flow_out).abs().max().item()
    assert diff < 1e-4, (
        f"Raw model vs IR (TTFS) logits max diff {diff:.2e} >= 1e-4. "
        "Check IR weight/bias transfer, activation_type, threshold handling."
    )
    model_pred = model_out.argmax(dim=1)
    flow_pred = flow_out.argmax(dim=1)
    agreement = (model_pred == flow_pred).float().mean().item()
    assert agreement == 1.0, (
        f"Argmax agreement {agreement:.0%} < 100% on batch of 4."
    )
