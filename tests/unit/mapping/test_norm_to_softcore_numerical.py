"""Integration test: numerical equivalence from NormalizationFusion through SoftCoreMapping.

This test replicates the exact sequence of transformations that the pipeline applies,
comparing the model accuracy (or logit agreement) between:

  Stage 1: NormalizationFusionStep output  (float model with staircase decorators)
  Stage 2: SoftCoreMappingStep output       (SpikingUnifiedCoreFlow on IRGraph)

Tests are organized around:
  - spiking_mode: "ttfs" and "ttfs_quantized"
  - activation_quantization: True / False
  - model topology: linear-only vs mixed (ReLU + Identity layers)

Per-layer diagnostic tests isolate WHERE the divergence first appears in the graph.

Known issues documented here:
  Bug 1: Identity perceptron gets TTFS bias shift → ComputeOp bias is wrong.
  Bug 2: For ttfs continuous, shift is in training but not baked into IR biases.
  Bug 3: Staircase granularity in training ≠ continuous relu in ttfs deployment.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.tuning.adaptation_manager import AdaptationManager
from mimarsinan.tuning.shift_calculation import calculate_activation_shift
from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.mapping.mappers.structural import InputMapper, EinopsRearrangeMapper
from mimarsinan.mapping.mappers.perceptron import PerceptronMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import NeuralCore, ComputeOp
from mimarsinan.mapping.per_source_scales import compute_per_source_scales
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow


# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

def _make_perceptron(in_feat, out_feat, activation, seed):
    torch.manual_seed(seed)
    p = Perceptron(out_feat, in_feat, normalization=nn.Identity(),
                   base_activation_name=activation)
    nn.init.normal_(p.layer.weight, 0, 0.4)
    nn.init.normal_(p.layer.bias, 0, 0.05)
    return p


def _apply_adaptation_all(perceptrons, spiking_mode, tq, act_scale):
    am = AdaptationManager()
    am.clamp_rate = 1.0
    am.quantization_rate = 1.0
    config = {"spiking_mode": spiking_mode, "target_tq": tq}
    for p in perceptrons:
        p.set_activation_scale(act_scale)
        am.update_activation(config, p)


def _simulate_normfusion(perceptrons):
    """Fuse BN into weights (mirrors NormalizationFusionStep; all are Identity norm here)."""
    # All test perceptrons use nn.Identity() normalization — nothing to fuse.
    pass


def _simulate_softcore_bias_shift_BUGGY(perceptrons, tq):
    """Reproduce current SoftCoreMappingStep: shift applied to ALL perceptrons."""
    pt = PerceptronTransformer()
    for p in perceptrons:
        shift = float(calculate_activation_shift(tq, float(p.activation_scale)))
        bias_shift = shift / float(p.activation_scale)
        pt.apply_effective_bias_transform(p, lambda b, s=bias_shift: b + s)


def _simulate_softcore_bias_shift_FIXED(perceptrons, tq):
    """Apply bias shift to all perceptrons (Identity layers are no longer perceptrons)."""
    pt = PerceptronTransformer()
    for p in perceptrons:
        shift = float(calculate_activation_shift(tq, float(p.activation_scale)))
        bias_shift = shift / float(p.activation_scale)
        pt.apply_effective_bias_transform(p, lambda b, s=bias_shift: b + s)


def _build_flow_from_repr(mapper_repr, input_shape, tq, spiking_mode):
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(q_max=127, firing_mode="TTFS", max_axons=4096, max_neurons=4096)
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


def _logit_agreement(out1, out2):
    pred1 = out1.argmax(1)
    pred2 = out2.argmax(1)
    return (pred1 == pred2).float().mean().item()


def _per_layer_diagnostics(ir_graph, flow, x_flat):
    """
    Trace through each IR node and return per-node (input, output) ranges.
    Useful for finding the first divergence point.
    """
    batch_size = x_flat.shape[0]
    device = x_flat.device
    diagnostics = []
    activation_cache = {}

    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            weight = flow._get_weight(node)
            t_idx = flow._get_threshold_idx(node)
            threshold = flow.thresholds[t_idx]
            spans = flow._input_spans[int(node.id)]
            in_dim = int(len(node.input_sources.flatten()))
            inp = torch.zeros(batch_size, in_dim, device=device)
            flow._fill_activation_from_ir_spans(
                inp, x=x_flat, activation_cache=activation_cache, spans=spans
            )
            V = torch.matmul(weight, inp.T).T
            if node.id in flow._hw_bias_params:
                V = V + flow._hw_bias_params[node.id]
            # ttfs_quantized formula
            S = float(flow.simulation_length)
            k_raw = torch.ceil(S * (1.0 - V / threshold.clamp(min=1e-12)))
            fires = k_raw < S
            k = k_raw.clamp(0, S - 1)
            out = torch.where(fires, (S - k) / S, torch.zeros_like(k))
            activation_cache[node.id] = out
            diagnostics.append({
                "id": node.id, "name": node.name, "type": "NeuralCore",
                "act_type": node.activation_type,
                "in_range": (float(inp.min()), float(inp.max())),
                "V_range": (float(V.min()), float(V.max())),
                "out_range": (float(out.min()), float(out.max())),
                "threshold": float(threshold),
            })
        elif isinstance(node, ComputeOp):
            spans = flow._input_spans[int(node.id)]
            in_dim = int(len(node.input_sources.flatten()))
            inp = torch.zeros(batch_size, in_dim, device=device)
            flow._fill_activation_from_ir_spans(
                inp, x=x_flat, activation_cache=activation_cache, spans=spans
            )
            out = node.execute_on_gathered(inp)
            activation_cache[node.id] = out
            diagnostics.append({
                "id": node.id, "name": node.name, "type": "ComputeOp",
                "op_type": node.op_type,
                "in_range": (float(inp.min()), float(inp.max())),
                "out_range": (float(out.min()), float(out.max())),
            })

    return diagnostics, activation_cache


def _print_diagnostics(diagnostics):
    print("\n=== Per-layer IR diagnostics ===")
    for d in diagnostics:
        if d["type"] == "NeuralCore":
            print(f"  [{d['id']:3d}] NeuralCore  {d['name']:30s}  "
                  f"act={d['act_type'] or 'None':25s}  "
                  f"V=[{d['V_range'][0]:+.3f},{d['V_range'][1]:+.3f}]  "
                  f"out=[{d['out_range'][0]:+.3f},{d['out_range'][1]:+.3f}]  "
                  f"thresh={d['threshold']:.4f}")
        else:
            print(f"  [{d['id']:3d}] ComputeOp   {d['name']:30s}  "
                  f"op={d['op_type']:15s}  "
                  f"in=[{d['in_range'][0]:+.3f},{d['in_range'][1]:+.3f}]  "
                  f"out=[{d['out_range'][0]:+.3f},{d['out_range'][1]:+.3f}]")


# ---------------------------------------------------------------------------
# Test 1: Pure ReLU model — ttfs_quantized — should be near-exact
# ---------------------------------------------------------------------------

class TestPureReLUModel:
    """All-ReLU model: NormFusion→SoftCore must be near-exact for ttfs_quantized."""

    def _build_relu_mlp(self, sizes, seed=42):
        """Build linear chain of ReLU perceptrons."""
        perceptrons = []
        mappers = [InputMapper((sizes[0],))]
        for i in range(1, len(sizes)):
            p = _make_perceptron(sizes[i - 1], sizes[i], "ReLU", seed + i)
            perceptrons.append(p)
            mappers.append(PerceptronMapper(mappers[-1], p))
        repr_ = ModelRepresentation(mappers[-1])
        return repr_, perceptrons, (sizes[0],)

    @pytest.mark.parametrize("tq", [8, 16])
    def test_relu_mlp_ttfs_quantized_near_exact(self, tq):
        """
        All-ReLU 3-layer MLP: after NormFusion + correct bias shift,
        SoftCoreMapping IR output must match NormFusion training output.

        Protocol: capture training output BEFORE the bias shift is applied.
        SoftCoreMappingStep modifies layer.bias in-place; running the training
        model AFTER the shift would double-count the shift (decorator + bias).
        The correct comparison is: pre-shift training ≡ post-shift IR.
        """
        act_scale = 1.0
        repr_, perceptrons, input_shape = self._build_relu_mlp([16, 12, 8, 4], seed=42)

        # Simulate NormFusion (no-op for Identity norm) + adaptation
        _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

        # Capture NormFusion accuracy BEFORE bias shift (as pipeline does)
        torch.manual_seed(77)
        x = torch.floor(torch.rand(128, *input_shape) * tq) / tq
        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            train_out_pre = repr_(x).clone()

        # Apply SoftCoreMapping bias shift (all ReLU — BUGGY == FIXED for this topology)
        _simulate_softcore_bias_shift_BUGGY(perceptrons, tq)

        # Build IR from modified model
        flow, ir_graph = _build_flow_from_repr(repr_, input_shape, tq, "ttfs_quantized")

        with torch.no_grad():
            flow_out = flow(x)

        # Pre-shift training output == post-shift IR output (up to act_scale factor)
        max_diff = (train_out_pre / act_scale - flow_out).abs().max().item()
        agreement = _logit_agreement(train_out_pre, flow_out)

        print(f"\nReLU MLP ttfs_quantized tq={tq}: max_diff={max_diff:.4f}, "
              f"agreement={agreement:.1%}")

        assert max_diff < 2.0 / tq, (
            f"Pure ReLU ttfs_quantized: max_diff={max_diff:.4f} > 2/tq={2/tq:.4f}. "
            "ReLU layers with ttfs_quantized should have near-exact equivalence."
        )

    def test_relu_mlp_ttfs_continuous_has_shift_error(self):
        """
        All-ReLU model with ttfs continuous: there IS a shift mismatch (Bug 2).
        Training adds shift, deployment does not.
        """
        tq = 8
        act_scale = 1.0
        repr_, perceptrons, input_shape = self._build_relu_mlp([16, 8, 4], seed=10)
        _apply_adaptation_all(perceptrons, "ttfs", tq, act_scale)
        # No bias shift applied (as SoftCoreMappingStep does for "ttfs")

        flow, _ = _build_flow_from_repr(repr_, input_shape, tq, "ttfs")

        torch.manual_seed(55)
        x = torch.rand(128, *input_shape) * 0.9

        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            train_out = repr_(x)
            flow_out = flow(x)

        max_diff = (train_out - flow_out * act_scale).abs().max().item()
        shift = float(calculate_activation_shift(tq, act_scale))

        print(f"\nReLU MLP ttfs continuous: max_diff={max_diff:.4f}, "
              f"expected_shift={shift:.4f}")

        # For ttfs continuous, we expect non-zero error due to the shift mismatch
        # The error should be at most ~shift per step * depth (roughly)
        assert max_diff > 0.0, (
            "Expected nonzero error for ttfs continuous (shift in training, not IR). "
            "If this is zero, the shift may not be active in training."
        )


# TestMixedReLUIdentityModel and TestPerLayerDivergencePinpointing removed:
# Identity layers are no longer perceptrons — they go through ModuleComputeMapper
# and never enter the adaptation/shift pipeline. The Identity-shift bug they
# tested is architecturally impossible in the new design.


# ---------------------------------------------------------------------------
# Test 4: act_q=False path — no shift anywhere, both ttfs and ttfs_quantized
# ---------------------------------------------------------------------------

class TestActivationQuantizationFalsePath:
    """
    When activation_quantization=False:
      - For rate mode: ActivationAdaptationStep runs (no clamp, no staircase, no shift).
      - For ttfs/ttfs_quantized: ClampAdaptationStep runs (clamp_rate→1.0 for all-ReLU).
    NormFusion → SoftCoreMapping should be near-exact after adaptation.
    """

    def _apply_no_actq_adaptation(self, perceptrons, act_scale=1.0):
        """Simulate act_q=False (rate mode): set activation_scale, keep base ReLU (no decorators)."""
        for p in perceptrons:
            p.set_activation_scale(act_scale)
            # No decorators — activation stays as base LeakyGradReLU/Identity

    def _apply_clamp_adaptation(self, perceptrons, act_scale=1.0, tq=8, spiking_mode="ttfs"):
        """Simulate ClampAdaptationStep for TTFS (all-ReLU path, clamp_rate=1.0)."""
        config = {"target_tq": tq, "spiking_mode": spiking_mode}
        am = AdaptationManager()
        am.clamp_rate = 1.0
        for p in perceptrons:
            p.set_activation_scale(act_scale)
            am.update_activation(config, p)

    def test_no_actq_ttfs_continuous_near_exact(self):
        """
        act_q=False + ttfs (continuous): ClampAdaptationStep runs (not ActivationAdaptationStep).
        ClampAdaptationStep sets clamp_rate=1.0 for all-ReLU models, applying ClampDecorator.
        Training forward: clamp(relu(V), 0, act_scale).
        Deploy: clamp(relu(V)/threshold, 0, 1) = clamp(relu(V), 0, 1) when threshold=1.0.
        With act_scale=threshold=1.0 these are numerically identical.
        """
        tq = 8
        act_scale = 1.0
        torch.manual_seed(42)
        in_feat, h, out_feat = 16, 12, 4
        p1 = _make_perceptron(in_feat, h, "ReLU", seed=0)
        p2 = _make_perceptron(h, out_feat, "ReLU", seed=1)

        inp = InputMapper((in_feat,))
        m1 = PerceptronMapper(inp, p1)
        m2 = PerceptronMapper(m1, p2)
        repr_ = ModelRepresentation(m2)

        # Simulate ClampAdaptationStep (clamp_rate=1.0): training forward clamps at act_scale
        self._apply_clamp_adaptation([p1, p2], act_scale, tq, spiking_mode="ttfs")

        # No bias shift (act_q=False)
        flow, _ = _build_flow_from_repr(repr_, (in_feat,), tq, "ttfs")

        torch.manual_seed(77)
        x = torch.rand(64, in_feat) * 0.8

        for p in [p1, p2]:
            p.eval()
        with torch.no_grad():
            train_out = repr_(x)
            flow_out = flow(x)

        # Training: clamp(relu(V), 0, 1).  Deploy: clamp(relu(V)/1.0, 0, 1).
        # With act_scale=threshold=1.0 these match exactly (float precision only).
        max_diff = (train_out - flow_out * act_scale).abs().max().item()
        agreement = _logit_agreement(train_out, flow_out)

        print(f"\nNo act_q, ttfs: max_diff={max_diff:.6f}, agreement={agreement:.1%}")

        assert max_diff < 1e-3, (
            f"No act_q, ttfs continuous: max_diff={max_diff:.6f} "
            f"should be near zero (no quantization or shift, clamp adaptation applied)."
        )

    def test_no_actq_ttfs_quantized_matches_within_staircase_granularity(self):
        """
        act_q=False + ttfs_quantized: training = relu(V), deploy = floor(tq*relu(V))/tq.
        These differ by quantization granularity: at most 1/tq per output.
        Outputs must be within [0, act_scale] for this bound to hold.
        """
        tq = 16  # higher tq → smaller staircase step
        act_scale = 1.0
        torch.manual_seed(42)
        in_feat, out_feat = 8, 4

        # Use small weights so outputs are bounded within [0, act_scale]
        p1 = Perceptron(out_feat, in_feat, normalization=nn.Identity(), base_activation_name="ReLU")
        torch.manual_seed(7)
        nn.init.uniform_(p1.layer.weight, 0, 0.05)  # small weights → outputs stay in [0,1]
        nn.init.constant_(p1.layer.bias, 0.01)
        p1.set_activation_scale(act_scale)

        inp = InputMapper((in_feat,))
        fc = PerceptronMapper(inp, p1)
        repr_ = ModelRepresentation(fc)

        # No bias shift
        flow, _ = _build_flow_from_repr(repr_, (in_feat,), tq, "ttfs_quantized")

        torch.manual_seed(77)
        x = torch.rand(128, in_feat) * 0.9

        p1.eval()
        with torch.no_grad():
            train_out = repr_(x)   # relu(V) in [0, ~0.4]
            flow_out = flow(x)     # floor(tq * relu(V)) / tq

        # Verify outputs are in [0, act_scale] (otherwise TTFS clamp changes semantics)
        train_max = train_out.max().item()
        assert train_max <= act_scale + 1e-4, (
            f"Training outputs {train_max:.4f} exceed act_scale={act_scale}. "
            "Use smaller weights for this test."
        )

        max_diff = (train_out - flow_out * act_scale).abs().max().item()
        print(f"\nNo act_q, ttfs_quantized (small weights): max_diff={max_diff:.6f}")

        assert max_diff <= 1.0 / tq + 1e-5, (
            f"No act_q, ttfs_quantized: max_diff={max_diff:.4f} > 1/tq={1/tq:.4f}. "
            "TTFS quantized introduces at most 1/tq staircase granularity."
        )
