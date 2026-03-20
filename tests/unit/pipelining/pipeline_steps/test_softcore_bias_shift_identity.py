"""Test: SoftCoreMappingStep bias shift correctness.

Tests:
  1. Verify shift magnitude formula.
  2. Single ReLU perceptron: training (staircase) must equal ttfs_quantized IR.

Identity-specific tests removed: Identity layers are no longer perceptrons —
they go through ModuleComputeMapper and never enter the adaptation/shift pipeline.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_relu_perceptron(in_feat, out_feat, seed=0):
    torch.manual_seed(seed)
    p = Perceptron(out_feat, in_feat, normalization=nn.Identity(), base_activation_name="ReLU")
    nn.init.normal_(p.layer.weight, 0, 0.5)
    nn.init.normal_(p.layer.bias, 0, 0.1)
    return p


def _apply_adaptation(perceptron, spiking_mode="ttfs_quantized", tq=8, act_scale=1.0):
    """Apply AdaptationManager at rate=1.0 (fully quantized) and set activation_scale."""
    perceptron.set_activation_scale(act_scale)
    am = AdaptationManager()
    am.clamp_rate = 1.0
    am.quantization_rate = 1.0
    config = {"spiking_mode": spiking_mode, "target_tq": tq}
    am.update_activation(config, perceptron)


def _simulate_softcore_bias_shift(model, tq):
    """Reproduce the SoftCoreMappingStep bias shift loop."""
    pt = PerceptronTransformer()
    for perceptron in model.get_perceptrons():
        shift = calculate_activation_shift(tq, perceptron.activation_scale)
        bias_shift = shift / perceptron.activation_scale
        pt.apply_effective_bias_transform(perceptron, lambda b, s=bias_shift: b + s)


def _build_flow(mapper_repr, input_shape, tq=8, spiking_mode="ttfs_quantized"):
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(q_max=127, firing_mode="TTFS", max_axons=2048, max_neurons=2048)
    ir_graph = ir_mapping.map(mapper_repr)
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            node.threshold = 1.0
            node.parameter_scale = torch.tensor(1.0)
    flow = SpikingUnifiedCoreFlow(
        input_shape, ir_graph, tq, nn.Identity(),
        "TTFS", "TTFS", "<=", spiking_mode=spiking_mode,
    )
    flow.eval()
    return flow, ir_graph


# ---------------------------------------------------------------------------
# Test 1: shift magnitude sanity check
# ---------------------------------------------------------------------------

class TestShiftMagnitude:
    """The shift formula: shift = (act_scale * 0.5) / tq."""

    @pytest.mark.parametrize("act_scale,tq", [
        (1.0, 8), (2.0, 8), (1.0, 16), (3.5, 32),
    ])
    def test_shift_formula(self, act_scale, tq):
        expected = act_scale * 0.5 / tq
        got = float(calculate_activation_shift(tq, act_scale))
        assert abs(got - expected) < 1e-9, (
            f"shift({act_scale=}, {tq=}) = {got}, expected {expected}"
        )

    def test_shift_is_half_quantization_step(self):
        """shift should equal half of one quantization step (act_scale / tq)."""
        act_scale, tq = 2.0, 8
        step = act_scale / tq
        shift = float(calculate_activation_shift(tq, act_scale))
        assert abs(shift - step / 2) < 1e-9, (
            f"shift={shift} is not half step={step/2}"
        )


# ---------------------------------------------------------------------------
# Test 2: ReLU NeuralCore equivalence (should PASS with correct shift)
# ---------------------------------------------------------------------------

class TestReLUNeuralCoreTTFSQuantized:
    """Single ReLU perceptron: training (staircase) must equal ttfs_quantized IR."""

    def test_single_relu_layer_matches_ir_ttfs_quantized(self):
        tq = 8
        act_scale = 1.0
        in_feat, out_feat = 8, 6
        input_shape = (in_feat,)

        p = _make_relu_perceptron(in_feat, out_feat, seed=3)
        p.set_activation_scale(act_scale)
        _apply_adaptation(p, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        inp_m = InputMapper(input_shape)
        fc = PerceptronMapper(inp_m, p)
        repr_ = ModelRepresentation(fc)

        class FakeModel:
            def get_perceptrons(self):
                return [p]

        _simulate_softcore_bias_shift(FakeModel(), tq)

        flow, _ = _build_flow(repr_, input_shape, tq=tq, spiking_mode="ttfs_quantized")

        torch.manual_seed(99)
        x_raw = torch.rand(64, in_feat) * 0.95
        x = torch.floor(x_raw * tq) / tq

        p.eval()
        with torch.no_grad():
            train_out = repr_(x)
            flow_out = flow(x)

        max_diff = (train_out / act_scale - flow_out).abs().max().item()
        assert max_diff < 1.5 / tq, (
            f"ReLU+ttfs_quantized: training vs IR max diff {max_diff:.6f} "
            f"(tolerance = 1.5/tq = {1.5/tq:.4f}). "
            "Staircase and TTFS quantized formula should be equivalent."
        )
