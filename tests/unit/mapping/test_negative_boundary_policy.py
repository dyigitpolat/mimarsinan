"""The negative value-boundary policy: BOTH positions are value-safe.

``negative_value_shift=on``  — a calibrated positive shift moves the boundary
into the encodable domain and the consuming perceptron's bias is pre-corrected.
``negative_value_shift=off`` — the mapper SUBSUMES FORWARD, moving consuming
perceptrons onto the host until a non-negative-value-generating node (ReLU,
LIF, ...) absorbs the signed range.

The silent [0,1] spike-encode clamp of the old OFF position must remain
unreachable: ``apply_negative_boundary_policy`` re-checks the invariant after
applying either mechanism and fails loud if any negative boundary is still
consumed on-chip.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.bias_compensation import calibration_forward_for_mode
from mimarsinan.mapping.support.negative_boundary import (
    apply_negative_boundary_policy,
    calibrated_compute_op_minima,
    lossy_negative_boundaries,
    subsume_forward_negative_boundaries,
)
from mimarsinan.torch_mapping.converter import convert_torch_model

T = 8


# ── Representative graphs ──────────────────────────────────────────────────
# Every graph's first perceptron is the host encoder (convert_torch_model marks
# segment starts); LayerNorm is the ComputeOp whose output goes negative.


class _ComputeOpToPerceptron(nn.Module):
    """… p2 (on-chip) → LayerNorm (host, signed) → p3 (ReLU) → p4 (on-chip)."""

    def __init__(self):
        super().__init__()
        self.fc1, self.a1 = nn.Linear(8, 6), nn.ReLU()
        self.fc2, self.a2 = nn.Linear(6, 6), nn.ReLU()
        self.ln = nn.LayerNorm(6)
        self.fc3, self.a3 = nn.Linear(6, 6), nn.ReLU()
        self.fc4, self.a4 = nn.Linear(6, 4), nn.ReLU()

    def forward(self, x):
        x = self.a2(self.fc2(self.a1(self.fc1(x))))
        x = self.ln(x)
        return self.a4(self.fc4(self.a3(self.fc3(x))))


class _ComputeOpToComputeOp(nn.Module):
    """… p2 → LayerNorm → LayerNorm → p3 (ReLU) → p4: no perceptron between
    the two host ops, so the SHIFT has no bias to pre-correct."""

    def __init__(self):
        super().__init__()
        self.fc1, self.a1 = nn.Linear(8, 6), nn.ReLU()
        self.fc2, self.a2 = nn.Linear(6, 6), nn.ReLU()
        self.ln1, self.ln2 = nn.LayerNorm(6), nn.LayerNorm(6)
        self.fc3, self.a3 = nn.Linear(6, 6), nn.ReLU()
        self.fc4, self.a4 = nn.Linear(6, 4), nn.ReLU()

    def forward(self, x):
        x = self.a2(self.fc2(self.a1(self.fc1(x))))
        x = self.ln2(self.ln1(x))
        return self.a4(self.fc4(self.a3(self.fc3(x))))


class _NoAbsorbingNode(nn.Module):
    """encoder → LayerNorm → p2 (GELU, signed) → out: nothing absorbs, and
    subsuming p2 would leave no on-chip segment at all."""

    def __init__(self):
        super().__init__()
        self.fc1, self.a1 = nn.Linear(8, 6), nn.ReLU()
        self.ln = nn.LayerNorm(6)
        self.fc2, self.a2 = nn.Linear(6, 4), nn.GELU()

    def forward(self, x):
        return self.a2(self.fc2(self.ln(self.a1(self.fc1(x)))))


class _ComputeOpFree(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.a1 = nn.Linear(8, 6), nn.ReLU()

    def forward(self, x):
        return self.a1(self.fc1(x))


def _flow(module_cls, num_classes=4):
    torch.manual_seed(0)
    return convert_torch_model(
        module_cls().eval(), (8,), num_classes, device="cpu",
    ).double()


def _x():
    torch.manual_seed(1)
    return torch.rand(16, 8, dtype=torch.float64)


def _fwd():
    """The analytical NF: pointwise, activation-agnostic, and it never clamps —
    so it records the TRUE boundary minima whatever the mechanism does next."""
    return calibration_forward_for_mode("ttfs")


def _minima(flow, x):
    return calibrated_compute_op_minima(flow, x, T, forward_fn=_fwd())


def _onchip(flow):
    return [p for p in flow.get_perceptrons() if not p.is_encoding_layer]


def _hosted(flow):
    return [p for p in flow.get_perceptrons() if p.is_encoding_layer]


def _layernorm_ops(flow):
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    return [
        n for n in repr_._exec_order
        if isinstance(n, ComputeOpMapper)
        and "LayerNorm" in type(getattr(n, "module", None)).__name__
    ]


# ── The corruption the mechanisms exist to prevent ─────────────────────────


class TestTheUnprotectedBoundaryIsLossy:
    """Both mechanisms are load-bearing: with NEITHER applied, the graph
    carries a negative boundary that an on-chip segment would spike-encode,
    and the [0,1] clamp would silently drop it."""

    def test_the_layernorm_boundary_goes_negative(self):
        flow = _flow(_ComputeOpToPerceptron)
        minima = _minima(flow, _x())
        (ln,) = _layernorm_ops(flow)
        assert float(minima[ln].min()) < 0.0

    def test_an_unprotected_negative_boundary_is_reported_lossy(self):
        flow = _flow(_ComputeOpToPerceptron)
        minima = _minima(flow, _x())
        assert lossy_negative_boundaries(flow, minima) == _layernorm_ops(flow)

    def test_a_host_only_consumer_chain_is_not_a_boundary(self):
        """LayerNorm→LayerNorm is host→host: no spike encode, nothing to lose.
        Only the SECOND op's boundary is consumed on-chip."""
        flow = _flow(_ComputeOpToComputeOp)
        minima = _minima(flow, _x())
        ln1, ln2 = _layernorm_ops(flow)
        assert float(minima[ln1].min()) < 0.0
        assert lossy_negative_boundaries(flow, minima) == [ln2]


# ── OFF: subsume forward ───────────────────────────────────────────────────


class TestSubsumeForward:
    def test_the_consuming_perceptron_moves_to_the_host_and_absorbs(self):
        flow = _flow(_ComputeOpToPerceptron)
        before = len(_hosted(flow))
        subsumed = subsume_forward_negative_boundaries(flow, _minima(flow, _x()))

        p1, p2, p3, p4 = flow.get_perceptrons()
        assert subsumed == [p3]
        assert len(_hosted(flow)) == before + 1
        # p3's ReLU absorbs the range: p4 stays on chip, the walk stops there.
        assert p3.is_encoding_layer and not p4.is_encoding_layer
        assert not p2.is_encoding_layer

    def test_the_boundary_is_no_longer_lossy(self):
        flow = _flow(_ComputeOpToPerceptron)
        x = _x()
        minima = _minima(flow, x)
        subsume_forward_negative_boundaries(flow, minima)
        assert lossy_negative_boundaries(flow, minima) == []

    def test_the_walk_crosses_a_host_op_that_cannot_absorb(self):
        """LayerNorm→LayerNorm: the walk passes THROUGH the second host op
        (it absorbs nothing) and subsumes the first perceptron that does."""
        flow = _flow(_ComputeOpToComputeOp)
        x = _x()
        minima = _minima(flow, x)
        subsumed = subsume_forward_negative_boundaries(flow, minima)

        p1, p2, p3, p4 = flow.get_perceptrons()
        assert subsumed == [p3]
        assert not p4.is_encoding_layer
        assert lossy_negative_boundaries(flow, minima) == []

    def test_no_absorbing_node_fails_loud(self):
        """The topology neither mechanism can make sound: nothing downstream
        absorbs the range, and subsuming forward consumes the last on-chip
        segment. It must fail LOUD, never silently corrupt."""
        flow = _flow(_NoAbsorbingNode)
        minima = _minima(flow, _x())
        with pytest.raises(NotImplementedError, match="no on-chip segment"):
            subsume_forward_negative_boundaries(flow, minima)

    def test_a_clean_graph_subsumes_nothing(self):
        flow = _flow(_ComputeOpFree, num_classes=6)
        onchip_before = _onchip(flow)
        assert subsume_forward_negative_boundaries(flow, {}) == []
        assert _onchip(flow) == onchip_before

    def test_subsumption_is_idempotent(self):
        flow = _flow(_ComputeOpToPerceptron)
        minima = _minima(flow, _x())
        assert subsume_forward_negative_boundaries(flow, minima)
        assert subsume_forward_negative_boundaries(flow, minima) == []


# ── The policy entry point: both positions, one guarantee ──────────────────


class TestPolicyDispatch:
    def test_shift_on_bakes_the_consumer_bias_and_hosts_nothing(self):
        flow = _flow(_ComputeOpToPerceptron)
        hosted_before = _hosted(flow)
        result = apply_negative_boundary_policy(
            flow, _x(), T, shift_enabled=True, forward_fn=_fwd(),
        )
        (ln,) = _layernorm_ops(flow)
        assert ln in result.shifts
        assert getattr(flow.get_perceptrons()[2], "_neg_shift_baked", False)
        assert result.subsumed == []
        assert _hosted(flow) == hosted_before  # mapping structure unchanged

    def test_shift_off_subsumes_and_bakes_nothing(self):
        flow = _flow(_ComputeOpToPerceptron)
        result = apply_negative_boundary_policy(
            flow, _x(), T, shift_enabled=False, forward_fn=_fwd(),
        )
        assert result.shifts == {}
        assert result.subsumed == [flow.get_perceptrons()[2]]
        assert not getattr(flow.get_perceptrons()[2], "_neg_shift_baked", False)
        (ln,) = _layernorm_ops(flow)
        assert getattr(ln, "_negative_shift", None) is None

    def test_shift_on_fails_loud_where_it_cannot_absorb(self):
        """ON has no bias to pre-correct across a ComputeOp→ComputeOp seam."""
        flow = _flow(_ComputeOpToComputeOp)
        with pytest.raises(NotImplementedError, match="ComputeOp"):
            apply_negative_boundary_policy(
                flow, _x(), T, shift_enabled=True, forward_fn=_fwd(),
            )

    def test_shift_off_handles_what_shift_on_cannot(self):
        flow = _flow(_ComputeOpToComputeOp)
        result = apply_negative_boundary_policy(
            flow, _x(), T, shift_enabled=False, forward_fn=_fwd(),
        )
        assert result.subsumed == [flow.get_perceptrons()[2]]

    def test_both_positions_leave_no_lossy_boundary(self):
        """The guarantee that makes silent corruption unauthorable: whichever
        mechanism ran, the post-condition is re-checked from the SAME
        calibrated minima the corruption test uses."""
        for shift_enabled in (True, False):
            flow = _flow(_ComputeOpToPerceptron)
            result = apply_negative_boundary_policy(
                flow, _x(), T, shift_enabled=shift_enabled, forward_fn=_fwd(),
            )
            assert lossy_negative_boundaries(flow, result.minima) == []

    def test_a_computeop_free_graph_pays_no_calibration_forward(self):
        flow = _flow(_ComputeOpFree, num_classes=6)

        def _must_not_run(model, x, t, compute_min_recorder=None):
            raise AssertionError("calibration forward ran on a ComputeOp-free graph")

        for shift_enabled in (True, False):
            result = apply_negative_boundary_policy(
                flow, _x(), T, shift_enabled=shift_enabled, forward_fn=_must_not_run,
            )
            assert result.shifts == {} and result.subsumed == []

    def test_off_path_fails_loud_on_the_unsound_topology(self):
        flow = _flow(_NoAbsorbingNode)
        with pytest.raises(NotImplementedError, match="no on-chip segment"):
            apply_negative_boundary_policy(
                flow, _x(), T, shift_enabled=False, forward_fn=_fwd(),
            )


# ── Numeric value-safety: each mechanism is exact where it acts ────────────


def _boundary_value(flow, x, op):
    """The float64 value the LayerNorm boundary presents, from the no-clamp
    analytical NF (post-shift when the ON mechanism baked one)."""
    recorder: dict = {}
    node_values: dict = {}
    with torch.no_grad():
        _fwd()(flow, x, T, compute_min_recorder=recorder)
    del node_values
    return recorder[op]


class TestNumericValueSafety:
    """Each mechanism is exact IN THE PLACE IT ACTS, and the unprotected clamp
    is not. (The two positions do not agree numerically end-to-end, and must
    not be asserted to: OFF moves the consumer onto the host in exact float,
    while ON keeps it on-chip where the spiking forward quantizes at O(1/T).
    What they share is the boundary-losslessness invariant.)"""

    def _effective(self, perceptron):
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        t = PerceptronTransformer()
        return t.get_effective_weight(perceptron), t.get_effective_bias(perceptron)

    def test_shift_leaves_the_consumer_pre_activation_bit_identical(self):
        """ON's exactness: the boundary moves by +s and the consumer's bias by
        -W·s, so W·(v+s) + (B - W·s) == W·v + B. The next activation absorbs
        the shift EXACTLY — this is why the shift is value-lossless."""
        import torch.nn.functional as F

        x = _x()
        reference = _flow(_ComputeOpToPerceptron)
        (ln_ref,) = _layernorm_ops(reference)
        v = _boundary_value(reference, x, ln_ref)  # per-channel minima carrier
        # A batch of boundary vectors spanning the calibrated range.
        torch.manual_seed(2)
        v_batch = v.unsqueeze(0) + torch.rand(8, v.numel(), dtype=torch.float64)
        w_ref, b_ref = self._effective(reference.get_perceptrons()[2])
        pre_reference = F.linear(v_batch, w_ref, b_ref)

        shifted = _flow(_ComputeOpToPerceptron)
        result = apply_negative_boundary_policy(
            shifted, x, T, shift_enabled=True, forward_fn=_fwd(),
        )
        (ln,) = _layernorm_ops(shifted)
        s = torch.as_tensor(result.shifts[ln], dtype=torch.float64)
        w_shift, b_shift = self._effective(shifted.get_perceptrons()[2])
        pre_shifted = F.linear(v_batch + s, w_shift, b_shift)

        torch.testing.assert_close(pre_shifted, pre_reference, atol=1e-12, rtol=1e-12)
        assert float((v.unsqueeze(0) + s).min()) >= -1e-12  # boundary encodable

    def test_subsume_touches_no_weights_and_hosts_a_nonnegative_output(self):
        """OFF's exactness: it moves the consumer to the host and changes NO
        number. The host runs the perceptron in float on the raw (unclamped)
        boundary value, and its ReLU output is non-negative — so the encode of
        the NEW boundary is lossless."""
        x = _x()
        reference = _flow(_ComputeOpToPerceptron)
        subsumed = _flow(_ComputeOpToPerceptron)
        result = apply_negative_boundary_policy(
            subsumed, x, T, shift_enabled=False, forward_fn=_fwd(),
        )

        p3_ref, p3 = reference.get_perceptrons()[2], subsumed.get_perceptrons()[2]
        assert result.subsumed == [p3]
        torch.testing.assert_close(
            p3.layer.weight, p3_ref.layer.weight, atol=0.0, rtol=0.0,
        )
        torch.testing.assert_close(
            p3.layer.bias, p3_ref.layer.bias, atol=0.0, rtol=0.0,
        )
        assert not getattr(p3, "_neg_shift_baked", False)

        (ln,) = _layernorm_ops(subsumed)
        assert getattr(ln, "_negative_shift", None) is None

        # The host executes p3 on the raw boundary value; its output is >= 0.
        v = _boundary_value(subsumed, x, ln)
        torch.manual_seed(2)
        v_batch = v.unsqueeze(0) + torch.rand(8, v.numel(), dtype=torch.float64)
        with torch.no_grad():
            host_out = p3(v_batch)
        assert float(host_out.min()) >= 0.0

    def test_the_unprotected_clamp_destroys_the_negative_range(self):
        """Without either mechanism the encoder clamps the boundary at 0: the
        dropped mass is real, and it is exactly what both positions prevent."""
        x = _x()
        flow = _flow(_ComputeOpToPerceptron)
        (ln,) = _layernorm_ops(flow)
        v = _boundary_value(flow, x, ln)
        dropped = torch.clamp(-v, min=0.0)
        assert float(dropped.max()) > 1e-3
        assert lossy_negative_boundaries(flow, {ln: v}) == [ln]

    def test_the_recheck_uses_a_fresh_calibration_after_subsumption(self):
        """The post-condition is re-derived from a FRESH forward on the mutated
        graph, not from the pre-mechanism recording."""
        x = _x()
        flow = _flow(_ComputeOpToPerceptron)
        assert apply_negative_boundary_policy(
            flow, x, T, shift_enabled=False, forward_fn=_fwd(),
        ).subsumed
        recheck = calibrated_compute_op_minima(flow, x, T, forward_fn=_fwd())
        assert lossy_negative_boundaries(flow, recheck) == []
