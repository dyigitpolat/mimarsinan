"""Test: SoftCoreMappingStep bias shift must NOT be applied to Identity perceptrons.

Fix (applied):
  PerceptronMapper.owned_perceptron_groups() now returns [] for non-chip-supported
  activations (Identity, GELU, etc.), so ModelRepresentation.get_perceptrons() correctly
  excludes them.  The SoftCoreMappingStep bias shift loop is fine as-is — it already
  iterates model.get_perceptrons(), which no longer includes Identity perceptrons.

Tests in this file:
  1. Verify shift magnitude is correct for ReLU perceptron.
  2. Verify Identity perceptron bias is UNCHANGED by the shift loop (fix verification).
  3. Single-layer numerical equivalence: Identity ComputeOp training == IR.
  4. Single-layer numerical equivalence: ReLU NeuralCore training == IR ttfs_quantized.
  5. Two-layer (ReLU -> Identity): end-to-end argmax equivalence.
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
# Helpers
# ---------------------------------------------------------------------------

def _make_relu_perceptron(in_feat, out_feat, seed=0):
    torch.manual_seed(seed)
    p = Perceptron(out_feat, in_feat, normalization=nn.Identity(), base_activation_name="ReLU")
    nn.init.normal_(p.layer.weight, 0, 0.5)
    nn.init.normal_(p.layer.bias, 0, 0.1)
    return p


def _make_identity_perceptron(in_feat, out_feat, seed=1):
    torch.manual_seed(seed)
    p = Perceptron(out_feat, in_feat, normalization=nn.Identity(), base_activation_name="Identity")
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
    """Reproduce the SoftCoreMappingStep bias shift loop (current buggy behaviour)."""
    pt = PerceptronTransformer()
    for perceptron in model.get_perceptrons():
        shift = calculate_activation_shift(tq, perceptron.activation_scale)
        bias_shift = shift / perceptron.activation_scale
        pt.apply_effective_bias_transform(perceptron, lambda b, s=bias_shift: b + s)


def _build_flow(mapper_repr, input_shape, tq=8, spiking_mode="ttfs_quantized"):
    compute_per_source_scales(mapper_repr)
    ir_mapping = IRMapping(q_max=127, firing_mode="TTFS", max_axons=2048, max_neurons=2048)
    ir_graph = ir_mapping.map(mapper_repr)
    # No weight quantization: keep floats, threshold=1.0
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
# Test 2: Identity perceptron bias must be UNCHANGED by the shift loop
# ---------------------------------------------------------------------------

class TestIdentityPerceptronBiasShift:
    """Identity perceptrons must NOT receive the TTFS bias shift."""

    def test_identity_bias_unchanged_after_shift_loop(self):
        """
        After _simulate_softcore_bias_shift using a real ModelRepresentation,
        an Identity perceptron's effective bias must be IDENTICAL to its pre-shift value.

        Fix: PerceptronMapper.owned_perceptron_groups() returns [] for Identity activations,
        so ModelRepresentation.get_perceptrons() excludes them from the shift loop.
        """
        tq = 8
        act_scale = 1.0

        p_id = _make_identity_perceptron(4, 4)
        p_id.set_activation_scale(act_scale)
        _apply_adaptation(p_id, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        # Capture effective bias before
        pt = PerceptronTransformer()
        b_before = pt.get_effective_bias(p_id).detach().clone()

        # Use real ModelRepresentation — get_perceptrons() must exclude Identity
        inp = InputMapper((4,))
        fc = PerceptronMapper(inp, p_id)
        repr_ = ModelRepresentation(fc)

        _simulate_softcore_bias_shift(repr_, tq)
        b_after = pt.get_effective_bias(p_id).detach().clone()

        max_diff = (b_after - b_before).abs().max().item()
        assert max_diff == 0.0, (
            f"Identity perceptron effective bias changed by {max_diff:.6f} after "
            f"the shift loop. PerceptronMapper.owned_perceptron_groups() must return [] "
            f"for non-chip-supported activations."
        )

    def test_relu_bias_changed_by_correct_amount(self):
        """ReLU perceptron SHOULD have its bias shifted by shift/act_scale."""
        tq = 8
        act_scale = 1.5

        p_relu = _make_relu_perceptron(4, 4)
        p_relu.set_activation_scale(act_scale)
        _apply_adaptation(p_relu, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        pt = PerceptronTransformer()
        b_before = pt.get_effective_bias(p_relu).detach().clone()

        class FakeModel:
            def get_perceptrons(self):
                return [p_relu]

        _simulate_softcore_bias_shift(FakeModel(), tq)
        b_after = pt.get_effective_bias(p_relu).detach().clone()

        expected_shift = float(calculate_activation_shift(tq, act_scale)) / act_scale
        actual_shift = (b_after - b_before).detach()

        max_err = (actual_shift - expected_shift).abs().max().item()
        assert max_err < 1e-6, (
            f"ReLU perceptron bias shifted by wrong amount. "
            f"Expected {expected_shift:.6f}, max error {max_err:.2e}"
        )

    @pytest.mark.parametrize("act_scale", [1.0, 2.0, 0.5])
    def test_identity_bias_unchanged_for_various_scales(self, act_scale):
        """Identity bias must be stable across different activation scales (via real ModelRepresentation)."""
        tq = 16
        p_id = _make_identity_perceptron(8, 8)
        p_id.set_activation_scale(act_scale)
        _apply_adaptation(p_id, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        pt = PerceptronTransformer()
        b_before = pt.get_effective_bias(p_id).detach().clone()

        inp = InputMapper((8,))
        fc = PerceptronMapper(inp, p_id)
        repr_ = ModelRepresentation(fc)

        _simulate_softcore_bias_shift(repr_, tq)
        b_after = pt.get_effective_bias(p_id).detach().clone()

        max_diff = (b_after - b_before).abs().max().item()
        assert max_diff == 0.0, (
            f"Identity bias drifted by {max_diff:.6f} for act_scale={act_scale}"
        )


# ---------------------------------------------------------------------------
# Test 3: Single Identity layer numerical equivalence
# (Training forward == IR ComputeOp forward, NO bias shift on Identity)
# ---------------------------------------------------------------------------

class TestIdentityComputeOpNumerical:
    """Identity layer: training model output must equal IR ComputeOp output (up to act_scale)."""

    def _build_single_identity_repr(self, in_feat, out_feat, seed=42):
        torch.manual_seed(seed)
        input_shape = (in_feat,)
        p = _make_identity_perceptron(in_feat, out_feat, seed=seed)
        inp = InputMapper(input_shape)
        fc = PerceptronMapper(inp, p)
        return ModelRepresentation(fc), p, input_shape

    def test_identity_layer_no_bias_shift_matches_ir(self):
        """
        Without the bias shift, Identity perceptron output in training ==
        ComputeOp output in IR (up to act_scale factor).

        This test applies NO shift to the Identity perceptron (correct behaviour)
        and verifies exact numerical match.
        """
        tq = 8
        act_scale = 1.0
        repr_, p, input_shape = self._build_single_identity_repr(8, 4, seed=7)
        p.set_activation_scale(act_scale)
        _apply_adaptation(p, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        # Build IR WITHOUT applying the buggy shift
        flow, ir_graph = _build_flow(repr_, input_shape, tq=tq)

        x = torch.rand(16, *input_shape) * 0.9
        with torch.no_grad():
            train_out = repr_(x)    # shape (B, out_feat)
            flow_out = flow(x)      # shape (B, out_feat), scale = 1/act_scale

        # flow_out is in [output/act_scale] space;  train_out is in full scale
        # For Identity, act_scale=1.0 so they should match directly
        max_diff = (train_out - flow_out * act_scale).abs().max().item()
        assert max_diff < 1e-4, (
            f"Identity layer: training vs IR max diff {max_diff:.6f} (no shift applied). "
            "This should be near-zero — if it fails, the IR ComputeOp path is broken."
        )

    def test_identity_layer_with_buggy_bias_shift_diverges(self):
        """
        Applying the buggy bias shift to an Identity perceptron causes training != IR.

        This test DOCUMENTS the bug: after the shift is applied to the model params
        (as SoftCoreMappingStep does), the IR gets the wrong bias while the
        training model (which never used the shift) is unchanged.

        The test asserts divergence exists, confirming Bug 1.
        """
        tq = 8
        act_scale = 1.0
        repr_, p, input_shape = self._build_single_identity_repr(8, 4, seed=7)
        p.set_activation_scale(act_scale)
        _apply_adaptation(p, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        # Capture what training model produces BEFORE shift
        x = torch.rand(16, *input_shape) * 0.9
        with torch.no_grad():
            train_out_clean = repr_(x).clone()

        # Apply the BUGGY shift (what SoftCoreMappingStep currently does)
        class FakeModel:
            def get_perceptrons(self):
                return [p]

        _simulate_softcore_bias_shift(FakeModel(), tq)

        # Build IR AFTER the shift has been applied to model params
        flow, ir_graph = _build_flow(repr_, input_shape, tq=tq)

        with torch.no_grad():
            flow_out_shifted = flow(x)

        # The IR output now includes the spurious shift; training output does not
        max_diff_with_bug = (train_out_clean - flow_out_shifted * act_scale).abs().max().item()
        expected_shift_per_output = float(calculate_activation_shift(tq, act_scale))
        assert max_diff_with_bug > expected_shift_per_output * 0.5, (
            f"Expected a bias shift mismatch of ~{expected_shift_per_output:.4f} "
            f"but max_diff={max_diff_with_bug:.6f} is too small. Bug may not be present?"
        )


# ---------------------------------------------------------------------------
# Test 4: ReLU NeuralCore equivalence (should PASS with correct shift)
# ---------------------------------------------------------------------------

class TestReLUNeuralCoreTTFSQuantized:
    """Single ReLU perceptron: training (staircase) must equal ttfs_quantized IR."""

    def test_single_relu_layer_matches_ir_ttfs_quantized(self):
        """
        For a ReLU perceptron with act_q=True (staircase) and ttfs_quantized:
        - Training: staircase(relu(Wx+b+shift)) in [0, act_scale]
        - IR: floor(tq * relu(Wx+b+shift)/act_scale) / tq  (after bias shift baked in)
        These must be numerically identical (up to act_scale factor).
        """
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

        # Apply the CORRECT shift (only for ReLU perceptrons — bias shift is needed)
        class FakeModel:
            def get_perceptrons(self):
                return [p]

        _simulate_softcore_bias_shift(FakeModel(), tq)

        flow, _ = _build_flow(repr_, input_shape, tq=tq, spiking_mode="ttfs_quantized")

        # Use quantized inputs in [0, 1] (mimicking TTFS inputs)
        torch.manual_seed(99)
        x_raw = torch.rand(64, in_feat) * 0.95
        # Quantize inputs to tq levels (as InputCQ would)
        x = torch.floor(x_raw * tq) / tq

        p.eval()
        with torch.no_grad():
            train_out = repr_(x)   # staircase output in [0, act_scale]
            flow_out = flow(x)     # floor(tq*V)/tq in [0, 1]

        max_diff = (train_out / act_scale - flow_out).abs().max().item()
        assert max_diff < 1.5 / tq, (
            f"ReLU+ttfs_quantized: training vs IR max diff {max_diff:.6f} "
            f"(tolerance = 1.5/tq = {1.5/tq:.4f}). "
            "Staircase and TTFS quantized formula should be equivalent."
        )


# ---------------------------------------------------------------------------
# Test 5: Two-layer model (ReLU → Identity) end-to-end equivalence
# ---------------------------------------------------------------------------

class TestTwoLayerReLUIdentityEquivalence:
    """
    ReLU perceptron → Identity perceptron chain.
    The ReLU layer maps to a NeuralCore; the Identity layer maps to a ComputeOp.
    Bug 1 corrupts the ComputeOp output, causing argmax mismatch.
    """

    def _build_two_layer_repr(self, in_feat=8, hidden=6, out_feat=4, seed=42):
        torch.manual_seed(seed)
        input_shape = (in_feat,)
        p1 = _make_relu_perceptron(in_feat, hidden, seed=seed)
        p2 = _make_identity_perceptron(hidden, out_feat, seed=seed + 1)
        inp = InputMapper(input_shape)
        m1 = PerceptronMapper(inp, p1)
        m2 = PerceptronMapper(m1, p2)
        return ModelRepresentation(m2), p1, p2, input_shape

    def test_two_layer_no_shift_on_identity_exact_match(self):
        """
        Without applying shift to p2 (Identity), training and IR must match.
        Shift is only applied to p1 (ReLU), per the correct algorithm.

        Protocol: capture training output BEFORE modifying biases.
        The bias shift changes layer.bias in-place; running training after would
        cause double-shift for p1 (decorator + modified layer bias).
        """
        tq = 8
        act_scale = 1.0
        repr_, p1, p2, input_shape = self._build_two_layer_repr()

        p1.set_activation_scale(act_scale)
        p2.set_activation_scale(act_scale)
        _apply_adaptation(p1, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)
        _apply_adaptation(p2, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        # Capture training output BEFORE bias modification
        torch.manual_seed(77)
        x = torch.floor(torch.rand(64, *input_shape) * tq) / tq
        p1.eval()
        p2.eval()
        with torch.no_grad():
            train_out_pre = repr_(x).clone()

        # Apply shift ONLY to ReLU perceptron (correct behavior) — modifies p1.layer.bias
        pt = PerceptronTransformer()
        shift1 = float(calculate_activation_shift(tq, act_scale))
        bias_shift1 = shift1 / act_scale
        pt.apply_effective_bias_transform(p1, lambda b, s=bias_shift1: b + s)
        # p2 (Identity): NO shift applied

        flow, ir_graph = _build_flow(repr_, input_shape, tq=tq)

        n_compute_ops = sum(1 for n in ir_graph.nodes if isinstance(n, ComputeOp))
        assert n_compute_ops >= 1, "Expected at least one ComputeOp for Identity perceptron"

        with torch.no_grad():
            flow_out = flow(x)

        # IR uses (b_p1 + shift1) in bias, training saw (b_p1 + shift1 from decorator)
        # p2 (Identity): IR has original b_p2, training had original b_p2 → match
        max_diff = (train_out_pre / act_scale - flow_out).abs().max().item()
        assert max_diff < 1.5 / tq, (
            f"Two-layer (ReLU→Identity) max diff {max_diff:.6f} with correct shift. "
            "Should be near-zero when Identity bias is not shifted."
        )

    def test_two_layer_buggy_shift_on_identity_causes_mismatch(self):
        """
        Applying shift to BOTH p1 (ReLU) and p2 (Identity) — as the current code does —
        causes training != IR. This test confirms Bug 1 exists.
        """
        tq = 8
        act_scale = 1.0
        repr_, p1, p2, input_shape = self._build_two_layer_repr()

        p1.set_activation_scale(act_scale)
        p2.set_activation_scale(act_scale)
        _apply_adaptation(p1, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)
        _apply_adaptation(p2, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        # Capture training output BEFORE any shift
        torch.manual_seed(77)
        x = torch.floor(torch.rand(64, *input_shape) * tq) / tq
        p1.eval()
        p2.eval()
        with torch.no_grad():
            train_out_clean = repr_(x).clone()

        # Apply BUGGY shift (to ALL perceptrons, including Identity)
        class FakeModel:
            def get_perceptrons(self):
                return [p1, p2]

        _simulate_softcore_bias_shift(FakeModel(), tq)

        flow, _ = _build_flow(repr_, input_shape, tq=tq)

        with torch.no_grad():
            flow_out_buggy = flow(x)

        max_diff = (train_out_clean - flow_out_buggy * act_scale).abs().max().item()
        expected_min_drift = float(calculate_activation_shift(tq, act_scale)) * 0.5
        assert max_diff > expected_min_drift, (
            f"Expected mismatch > {expected_min_drift:.4f} from Identity shift bug, "
            f"but max_diff={max_diff:.6f}. Bug may not be active?"
        )

    def test_identity_core_appears_as_compute_op(self):
        """Perceptron with Identity activation must map to ComputeOp, not NeuralCore."""
        repr_, p1, p2, input_shape = self._build_two_layer_repr()
        p1.set_activation_scale(1.0)
        p2.set_activation_scale(1.0)

        compute_per_source_scales(repr_)
        ir_mapping = IRMapping(q_max=127, firing_mode="TTFS", max_axons=2048, max_neurons=2048)
        ir_graph = ir_mapping.map(repr_)

        neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
        compute_ops = [n for n in ir_graph.nodes if isinstance(n, ComputeOp)]

        assert len(neural_cores) == 1, (
            f"Expected 1 NeuralCore (ReLU), got {len(neural_cores)}"
        )
        assert len(compute_ops) == 1, (
            f"Expected 1 ComputeOp (Identity), got {len(compute_ops)}"
        )


# ---------------------------------------------------------------------------
# Test 6: Mixed model — verify shift is selectively applied
# ---------------------------------------------------------------------------

class TestSelectiveBiasShift:
    """Verify the fix: shift loop should check isinstance(p.base_activation, nn.Identity)."""

    def test_selective_shift_produces_correct_results(self):
        """
        A CORRECTED shift loop that skips Identity perceptrons should produce
        training == IR for a ReLU → Identity → ReLU three-layer model.

        Protocol: capture training output BEFORE applying the bias shift.
        SoftCoreMappingStep modifies biases in-place; running the training model
        AFTER the shift would produce double-shift (decorator + layer bias).
        """
        tq = 8
        act_scale = 1.0
        in_feat, h1, h2, out_feat = 8, 6, 5, 4
        input_shape = (in_feat,)

        torch.manual_seed(42)
        p1 = _make_relu_perceptron(in_feat, h1, seed=0)
        p2 = _make_identity_perceptron(h1, h2, seed=1)
        p3 = _make_relu_perceptron(h2, out_feat, seed=2)

        for p in [p1, p2, p3]:
            p.set_activation_scale(act_scale)
            _apply_adaptation(p, spiking_mode="ttfs_quantized", tq=tq, act_scale=act_scale)

        inp = InputMapper(input_shape)
        m1 = PerceptronMapper(inp, p1)
        m2 = PerceptronMapper(m1, p2)
        m3 = PerceptronMapper(m2, p3)
        repr_ = ModelRepresentation(m3)

        # Capture training output BEFORE the bias shift (decorator still adds shift)
        torch.manual_seed(55)
        x = torch.floor(torch.rand(64, *input_shape) * tq) / tq

        for p in [p1, p2, p3]:
            p.eval()
        with torch.no_grad():
            train_out_pre_shift = repr_(x).clone()

        # CORRECT shift: skip Identity (p2)
        # This modifies layer biases in-place (as SoftCoreMappingStep does).
        pt = PerceptronTransformer()
        for p in [p1, p2, p3]:
            if isinstance(p.base_activation, nn.Identity):
                continue
            shift = float(calculate_activation_shift(tq, p.activation_scale))
            bias_shift = shift / float(p.activation_scale)
            pt.apply_effective_bias_transform(p, lambda b, s=bias_shift: b + s)

        # Build IR from the modified model
        flow, ir_graph = _build_flow(repr_, input_shape, tq=tq)

        with torch.no_grad():
            flow_out = flow(x)

        # Compare IR (post-shift) to training (pre-shift): both see Wx+b+shift under the ReLU
        max_diff = (train_out_pre_shift / act_scale - flow_out).abs().max().item()
        assert max_diff < 2.0 / tq, (
            f"Selective shift (skip Identity): max diff {max_diff:.6f} "
            f"(tolerance {2.0/tq:.4f}). With the fix applied this should pass."
        )
