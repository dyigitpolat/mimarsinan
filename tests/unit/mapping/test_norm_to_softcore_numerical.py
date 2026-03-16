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
    """Fixed version: skip Identity perceptrons."""
    pt = PerceptronTransformer()
    for p in perceptrons:
        if isinstance(p.base_activation, nn.Identity):
            continue
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


# ---------------------------------------------------------------------------
# Test 2: Mixed model (ReLU + Identity) — ttfs_quantized — Bug 1 pinpointing
# ---------------------------------------------------------------------------

class TestMixedReLUIdentityModel:
    """ReLU + Identity layers: ttfs_quantized. Bug 1 (Identity shift) should appear here."""

    def _build_mixed_mlp(self, sizes, activations, seed=42):
        """
        Build linear chain with mixed activations.
        activations: list of activation names per layer (len = len(sizes) - 1)
        """
        perceptrons = []
        mappers = [InputMapper((sizes[0],))]
        for i, act in enumerate(activations):
            p = _make_perceptron(sizes[i], sizes[i + 1], act, seed + i)
            perceptrons.append(p)
            mappers.append(PerceptronMapper(mappers[-1], p))
        repr_ = ModelRepresentation(mappers[-1])
        return repr_, perceptrons, (sizes[0],)

    def test_buggy_shift_causes_mismatch_on_identity_layer(self):
        """
        With BUGGY bias shift (applied to all perceptrons including Identity):
        training and IR diverge, specifically at Identity (ComputeOp) layers.
        """
        tq = 8
        act_scale = 1.0
        # ReLU → Identity → ReLU
        repr_, perceptrons, input_shape = self._build_mixed_mlp(
            [16, 12, 8, 4], ["ReLU", "Identity", "ReLU"], seed=0
        )
        _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

        # Capture training output BEFORE any shift modifications
        torch.manual_seed(99)
        x = torch.floor(torch.rand(128, *input_shape) * tq) / tq
        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            train_out_clean = repr_(x).clone()

        # Apply BUGGY shift (all perceptrons)
        _simulate_softcore_bias_shift_BUGGY(perceptrons, tq)

        flow_buggy, ir_graph = _build_flow_from_repr(repr_, input_shape, tq, "ttfs_quantized")

        with torch.no_grad():
            flow_out_buggy = flow_buggy(x)

        # There should be at least 1 ComputeOp in the graph
        n_compute_ops = sum(1 for n in ir_graph.nodes if isinstance(n, ComputeOp))
        assert n_compute_ops >= 1, "Expected ComputeOp for Identity perceptron"

        agreement_buggy = _logit_agreement(train_out_clean, flow_out_buggy)
        max_diff_buggy = (train_out_clean - flow_out_buggy * act_scale).abs().max().item()

        print(f"\nMixed (ReLU→Id→ReLU) BUGGY shift: "
              f"agreement={agreement_buggy:.1%}, max_diff={max_diff_buggy:.4f}")

        # We expect the bug to cause some divergence
        expected_shift = float(calculate_activation_shift(tq, act_scale))
        assert max_diff_buggy > expected_shift * 0.3, (
            f"Expected mismatch from Identity bias shift bug. "
            f"max_diff={max_diff_buggy:.4f}, shift={expected_shift:.4f}. "
            "Bug 1 may not be present or its effect is too small to measure."
        )

    def test_fixed_shift_restores_match_on_identity_layer(self):
        """
        With FIXED bias shift (skip Identity perceptrons):
        pre-shift training output and post-shift IR output should match closely.

        Protocol: capture training output BEFORE bias modification.
        """
        tq = 8
        act_scale = 1.0
        # ReLU → Identity → ReLU
        repr_, perceptrons, input_shape = self._build_mixed_mlp(
            [16, 12, 8, 4], ["ReLU", "Identity", "ReLU"], seed=0
        )
        _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

        # Capture training output BEFORE any bias modification
        torch.manual_seed(99)
        x = torch.floor(torch.rand(128, *input_shape) * tq) / tq
        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            train_out_pre = repr_(x).clone()

        # Apply FIXED shift (skip Identity)
        _simulate_softcore_bias_shift_FIXED(perceptrons, tq)

        flow_fixed, ir_graph = _build_flow_from_repr(repr_, input_shape, tq, "ttfs_quantized")

        with torch.no_grad():
            flow_out = flow_fixed(x)

        agreement_fixed = _logit_agreement(train_out_pre, flow_out)
        max_diff_fixed = (train_out_pre / act_scale - flow_out).abs().max().item()

        print(f"\nMixed (ReLU→Id→ReLU) FIXED shift: "
              f"agreement={agreement_fixed:.1%}, max_diff={max_diff_fixed:.4f}")

        assert max_diff_fixed < 2.0 / tq, (
            f"Fixed shift: max_diff={max_diff_fixed:.4f} > 2/tq={2/tq:.4f}. "
            "The fix (skip Identity perceptrons in shift loop) should produce near-exact match."
        )

    def test_identity_only_model_no_shift_exact_match(self):
        """
        A model with ONLY Identity layers (all ComputeOps):
        no shift should be applied; IR output must exactly match training output.
        """
        tq = 8
        act_scale = 1.0
        repr_, perceptrons, input_shape = self._build_mixed_mlp(
            [16, 12, 4], ["Identity", "Identity"], seed=5
        )
        _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

        # No shift at all (FIXED behavior)
        flow, ir_graph = _build_flow_from_repr(repr_, input_shape, tq, "ttfs_quantized")

        n_cores = sum(1 for n in ir_graph.nodes if isinstance(n, NeuralCore))
        n_ops = sum(1 for n in ir_graph.nodes if isinstance(n, ComputeOp))
        assert n_cores == 0, f"Expected no NeuralCores for all-Identity model, got {n_cores}"
        assert n_ops == 2, f"Expected 2 ComputeOps, got {n_ops}"

        torch.manual_seed(77)
        x = torch.rand(64, *input_shape) * 0.8
        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            train_out = repr_(x)
            flow_out = flow(x)

        # Identity with act_scale=1.0: training out ≈ IR out (in [0,1] equivalent space)
        max_diff = (train_out - flow_out * act_scale).abs().max().item()
        assert max_diff < 1e-4, (
            f"All-Identity model: max_diff={max_diff:.6f}. "
            "ComputeOp chain should replicate training exactly."
        )

    def test_buggy_vs_fixed_comparison_report(self):
        """
        Side-by-side comparison of BUGGY vs FIXED shift on a mixed model.
        Reports agreement for both to quantify the improvement from the fix.

        Protocol: for each variant, capture training output BEFORE shift, then build IR.
        """
        tq = 8
        act_scale = 1.0
        # Topology mimicking MLP-Mixer: alternating Identity and ReLU
        sizes = [32, 24, 20, 16, 12, 8]
        activations = ["Identity", "ReLU", "Identity", "ReLU", "Identity"]
        torch.manual_seed(42)
        x = torch.floor(torch.rand(128, sizes[0]) * tq) / tq

        results = {}
        for variant, shift_fn in [("buggy", _simulate_softcore_bias_shift_BUGGY),
                                    ("fixed", _simulate_softcore_bias_shift_FIXED)]:
            repr_, perceptrons, input_shape = self._build_mixed_mlp(sizes, activations, seed=99)
            _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

            # Capture training output BEFORE bias modification
            for p in perceptrons:
                p.eval()
            with torch.no_grad():
                train_out_pre = repr_(x).clone()

            shift_fn(perceptrons, tq)
            flow, _ = _build_flow_from_repr(repr_, input_shape, tq, "ttfs_quantized")
            with torch.no_grad():
                flow_out = flow(x)

            agreement = _logit_agreement(train_out_pre, flow_out)
            max_diff = (train_out_pre / act_scale - flow_out).abs().max().item()
            results[variant] = {"agreement": agreement, "max_diff": max_diff}

        print(f"\nMixed model (Id/ReLU alternating):")
        for v, r in results.items():
            print(f"  {v:6s}: agreement={r['agreement']:.1%}, max_diff={r['max_diff']:.4f}")

        # Fixed should always have better agreement than buggy
        assert results["fixed"]["max_diff"] <= results["buggy"]["max_diff"] + 1e-6, (
            f"Fixed shift ({results['fixed']['max_diff']:.4f}) should have smaller or equal "
            f"max_diff than buggy ({results['buggy']['max_diff']:.4f})."
        )


# ---------------------------------------------------------------------------
# Test 3: Per-layer divergence pinpointing
# ---------------------------------------------------------------------------

class TestPerLayerDivergencePinpointing:
    """
    Trace through each IR node and find the first layer where training ≠ IR.
    This confirms that Bug 1 manifests at the ComputeOp (Identity) layer.
    """

    def test_first_divergence_at_identity_computeop(self):
        """
        For a ReLU → Identity model with the BUGGY shift applied:
        - ReLU NeuralCore output should match training closely
        - Identity ComputeOp output should diverge (Bug 1)
        """
        tq = 8
        act_scale = 1.0
        torch.manual_seed(42)
        in_feat, hidden, out_feat = 8, 6, 4

        p1 = _make_perceptron(in_feat, hidden, "ReLU", seed=0)
        p2 = _make_perceptron(hidden, out_feat, "Identity", seed=1)

        inp = InputMapper((in_feat,))
        m1 = PerceptronMapper(inp, p1)
        m2 = PerceptronMapper(m1, p2)
        repr_ = ModelRepresentation(m2)

        perceptrons = [p1, p2]
        _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

        # Capture training intermediate outputs
        torch.manual_seed(33)
        x = torch.floor(torch.rand(32, in_feat) * tq) / tq

        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            # Layer 1 output (ReLU, with staircase → in [0, act_scale])
            p1_out_train = p1(x).clone()
            # Layer 2 output (Identity, no shift)
            full_out_train = repr_(x).clone()

        # Apply BUGGY shift
        _simulate_softcore_bias_shift_BUGGY(perceptrons, tq)

        flow, ir_graph = _build_flow_from_repr(repr_, (in_feat,), tq, "ttfs_quantized")

        # Get per-layer IR diagnostics
        with torch.no_grad():
            flow_out = flow(x)

        diags, cache = _per_layer_diagnostics(ir_graph, flow, x)
        _print_diagnostics(diags)

        # Find the NeuralCore and ComputeOp nodes
        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
        compute_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp)]
        assert len(neural_cores) == 1, f"Expected 1 NeuralCore, got {len(neural_cores)}"
        assert len(compute_ops) == 1, f"Expected 1 ComputeOp, got {len(compute_ops)}"

        nc = neural_cores[0]
        co = compute_ops[0]

        # Check NeuralCore output vs p1 training output
        # NeuralCore output is in [0, 1]; training output is in [0, act_scale=1]
        nc_out_ir = cache[nc.id]  # (B, hidden)
        relu_diff = (p1_out_train / act_scale - nc_out_ir).abs().max().item()

        # Check ComputeOp output vs p2 training output
        co_out_ir = cache[co.id]  # (B, out_feat)
        identity_diff = (full_out_train / act_scale - co_out_ir).abs().max().item()

        print(f"\nReLU NeuralCore diff vs training: {relu_diff:.6f}")
        print(f"Identity ComputeOp diff vs training: {identity_diff:.6f}")
        print(f"Expected shift error: {calculate_activation_shift(tq, act_scale):.6f}")

        # ReLU NeuralCore: after shift is correctly baked in, should match
        assert relu_diff < 2.0 / tq, (
            f"ReLU NeuralCore diff {relu_diff:.4f} too large (>2/tq={2/tq:.4f}). "
            "The ReLU path should be correct after bias shift."
        )

        # Identity ComputeOp: BUGGY shift causes divergence — confirm it
        shift_error = float(calculate_activation_shift(tq, act_scale))
        assert identity_diff > shift_error * 0.3, (
            f"Identity ComputeOp diff {identity_diff:.4f} too small. "
            f"Expected divergence ~{shift_error:.4f} from Bug 1. "
            "If diff is near zero, the bug may be fixed already."
        )

    def test_first_divergence_at_identity_computeop_fixed(self):
        """
        With FIXED shift (skip Identity), the ComputeOp output matches training.
        """
        tq = 8
        act_scale = 1.0
        torch.manual_seed(42)
        in_feat, hidden, out_feat = 8, 6, 4

        p1 = _make_perceptron(in_feat, hidden, "ReLU", seed=0)
        p2 = _make_perceptron(hidden, out_feat, "Identity", seed=1)

        inp = InputMapper((in_feat,))
        m1 = PerceptronMapper(inp, p1)
        m2 = PerceptronMapper(m1, p2)
        repr_ = ModelRepresentation(m2)

        perceptrons = [p1, p2]
        _apply_adaptation_all(perceptrons, "ttfs_quantized", tq, act_scale)

        torch.manual_seed(33)
        x = torch.floor(torch.rand(32, in_feat) * tq) / tq

        for p in perceptrons:
            p.eval()
        with torch.no_grad():
            full_out_train = repr_(x).clone()

        # Apply FIXED shift (skip Identity p2)
        _simulate_softcore_bias_shift_FIXED(perceptrons, tq)

        flow, ir_graph = _build_flow_from_repr(repr_, (in_feat,), tq, "ttfs_quantized")

        diags, cache = _per_layer_diagnostics(ir_graph, flow, x)

        compute_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp)]
        assert len(compute_ops) == 1

        co = compute_ops[0]
        co_out_ir = cache[co.id]
        identity_diff = (full_out_train / act_scale - co_out_ir).abs().max().item()

        print(f"\nIdentity ComputeOp diff (FIXED): {identity_diff:.6f}")

        assert identity_diff < 2.0 / tq, (
            f"With FIXED shift, Identity ComputeOp diff {identity_diff:.4f} "
            f"should be < 2/tq={2/tq:.4f}."
        )


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
