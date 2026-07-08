"""Per-cascade-depth gain correction for the deployed single-spike TTFS cascade.

The ramp-integrate decode under-weights late (deep-layer) spikes, attenuating deep
layers geometrically (the death cascade). ``apply_cascaded_gain_correction`` inverts
this with a per-depth multiplicative trim of ``activation_scale`` (theta_d *= gamma^d,
gamma = 1 - sqrt(S)/(S+1)) and re-propagates input scales, leaving the depth-0
encoding/entry layer untouched. It is a pure calibration change — the decode
mechanism is unchanged, so it stays deployable / NF<->SCM-consistent.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

from conftest import make_tiny_supermodel
from cascade_fixtures import build_cascade_flow, cascade_forward

from mimarsinan.spiking.gain_correction import (
    apply_cascaded_gain_correction,
    apply_gain_at_rate,
    cascaded_gain_factors,
    g_geometric,
    g_relative,
    gamma_of,
    per_perceptron_cascade_depth,
)


class TestFormulas:
    def test_gamma_self_limits_with_S(self):
        assert gamma_of(8) < gamma_of(32) < 1.0
        # gamma(8) ~= 0.5 (char's hand-set 0.5^d falls out of the derived form)
        assert gamma_of(8) == pytest.approx(1.0 - math.sqrt(8) / 9, abs=1e-9)

    def test_relative_leaves_depth0_unchanged(self):
        assert g_relative(8, 0) == 1.0
        assert g_relative(8, 1) == pytest.approx(gamma_of(8))
        assert g_relative(8, 2) == pytest.approx(gamma_of(8) ** 2)

    def test_relative_shrinks_with_depth(self):
        assert g_relative(8, 3) < g_relative(8, 2) < g_relative(8, 1) < 1.0

    def test_geometric_corrects_depth0_too(self):
        assert g_geometric(8, 0) == pytest.approx(1.0 - 1.9 / 8)
        assert g_geometric(8, 0) < 1.0


class TestDepthMap:
    def test_depth_increases_along_cascade(self):
        torch.manual_seed(0)
        model = make_tiny_supermodel()
        depths = per_perceptron_cascade_depth(model.get_mapper_repr())
        ds = [depths[id(p)] for p in model.get_perceptrons()]
        assert ds == [0, 1]


class TestApplyCorrection:
    def _ttfs_model(self, seed=0, depth=4, S=8):
        flow, x = build_cascade_flow(host_ops=False, depth=depth, S=S, seed=seed)
        return flow, x

    def test_depth0_scale_unchanged_deep_scales_shrink(self):
        flow, _ = self._ttfs_model()
        before = [float(p.activation_scale) for p in flow.get_perceptrons()]
        stats = apply_cascaded_gain_correction(flow, 8, rule="relative", input_data_scale=1.0)
        after = [float(p.activation_scale) for p in flow.get_perceptrons()]
        depths = per_perceptron_cascade_depth(flow.get_mapper_repr())
        for p, b, a in zip(flow.get_perceptrons(), before, after):
            d = depths[id(p)]
            assert a == pytest.approx(b * g_relative(8, d), rel=1e-6)
        assert after[0] == pytest.approx(before[0])     # depth 0 untouched
        assert after[-1] < before[-1]                   # deepest shrunk
        assert stats["n_corrected"] >= 1

    def test_input_scales_repropagated(self):
        flow, _ = self._ttfs_model()
        apply_cascaded_gain_correction(flow, 8, rule="relative", input_data_scale=1.0)
        # After propagation, each non-entry perceptron's input_activation_scale equals
        # the mean activation_scale of its perceptron source(s) — here a chain, so it
        # equals the immediate upstream perceptron's (corrected) activation_scale.
        perceptrons = list(flow.get_perceptrons())
        for up, down in zip(perceptrons[:-1], perceptrons[1:]):
            assert float(down.input_activation_scale) == pytest.approx(
                float(up.activation_scale), abs=1e-6
            )

    def test_correction_changes_genuine_cascade_output(self):
        flow, x = self._ttfs_model()
        ref = copy.deepcopy(flow)
        out_before = cascade_forward(ref, x, 8)
        apply_cascaded_gain_correction(flow, 8, rule="relative", input_data_scale=1.0)
        out_after = cascade_forward(flow, x, 8)
        assert not torch.allclose(out_before, out_after, atol=1e-6), (
            "the gain correction must change the deployed cascade output (it revives "
            "the attenuated deep layers)"
        )

    def test_encoding_layer_is_pinned_both_rules(self):
        # The segment-entry (is_encoding_layer) perceptron's scale is fixed by the
        # input-encoding contract; NEITHER rule may retune it (geometric's rho_0
        # would otherwise crater NF<->SCM parity). A deeper layer still gets corrected.
        from mimarsinan.spiking.gain_correction import cascaded_gain_factors

        flow, _ = self._ttfs_model()
        ps = list(flow.get_perceptrons())
        enc = [p for p in ps if getattr(p, "is_encoding_layer", False)]
        assert enc, "fixture must have an encoding-layer entry"
        for rule in ("relative", "geometric"):
            factors = cascaded_gain_factors(flow, 8, rule=rule)
            for p in enc:
                assert factors[id(p)] == 1.0   # pinned
            assert any(factors[id(p)] < 1.0 for p in ps if p not in enc)  # deeper corrected

    def test_geometric_pins_encoding_unchanged_scale(self):
        flow, _ = self._ttfs_model()
        enc = next(p for p in flow.get_perceptrons() if getattr(p, "is_encoding_layer", False))
        before = float(enc.activation_scale)
        apply_cascaded_gain_correction(flow, 8, rule="geometric", input_data_scale=1.0)
        assert float(enc.activation_scale) == pytest.approx(before)  # pinned


class TestRateGated:
    """The rate-gated correction ramps theta_d = base * g_d**rate (rate 0 -> base,
    rate 1 -> full correction) so it can co-ramp with the KD blend."""

    def _flow(self, S=8):
        flow, _ = build_cascade_flow(host_ops=False, depth=4, S=S, seed=0)
        return flow

    def test_factors_match_static_rule(self):
        flow = self._flow()
        factors = cascaded_gain_factors(flow, 8, rule="relative")
        depths = per_perceptron_cascade_depth(flow.get_mapper_repr())
        for p in flow.get_perceptrons():
            assert factors[id(p)] == pytest.approx(g_relative(8, depths[id(p)]))

    def test_rate_zero_is_base_rate_one_is_full(self):
        flow = self._flow()
        base = [float(p.activation_scale) for p in flow.get_perceptrons()]
        factors = cascaded_gain_factors(flow, 8, rule="relative")
        apply_gain_at_rate(flow, base, factors, 0.0, input_data_scale=1.0)
        for p, b in zip(flow.get_perceptrons(), base):
            assert float(p.activation_scale) == pytest.approx(b)            # rate 0 -> base
        apply_gain_at_rate(flow, base, factors, 1.0, input_data_scale=1.0)
        for p, b in zip(flow.get_perceptrons(), base):
            assert float(p.activation_scale) == pytest.approx(b * factors[id(p)])  # rate 1 -> full

    def test_rate_half_is_geometric_midpoint(self):
        flow = self._flow()
        base = [float(p.activation_scale) for p in flow.get_perceptrons()]
        factors = cascaded_gain_factors(flow, 8, rule="relative")
        apply_gain_at_rate(flow, base, factors, 0.5, input_data_scale=1.0)
        for p, b in zip(flow.get_perceptrons(), base):
            assert float(p.activation_scale) == pytest.approx(b * (factors[id(p)] ** 0.5))

    def test_monotone_shrink_with_rate(self):
        flow = self._flow()
        base = [float(p.activation_scale) for p in flow.get_perceptrons()]
        factors = cascaded_gain_factors(flow, 8, rule="relative")
        deep = flow.get_perceptrons()[-1]
        seen = []
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            apply_gain_at_rate(flow, base, factors, r, input_data_scale=1.0)
            seen.append(float(deep.activation_scale))
        assert all(a >= b for a, b in zip(seen, seen[1:]))  # non-increasing (shrinks)


class TestRevivesDeepLayers:
    """Death-cascade revival is about FIRING: lowering theta_d makes a late/starved
    deep neuron cross threshold and fire, so its firing rate (decoded / theta = count/T)
    rises. (The decoded VALUE may move either way since theta itself shrank — what
    matters for downstream signal is that the layer is alive and firing.)"""

    def _firing_rates(self, flow, x, S):
        from mimarsinan.models.spiking.training.ttfs_segment_forward import (
            TTFSSegmentForward,
        )
        from mimarsinan.spiking.segment_partition import perceptron_of

        rec = {}
        drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
        drv._driver.policy.node_value_recorder = rec
        with torch.no_grad():
            drv(x.double())
        drv._driver.policy.node_value_recorder = None
        by = {id(perceptron_of(n)): v for n, v in rec.items()
              if perceptron_of(n) is not None}
        rates = []
        for p in flow.get_perceptrons():
            decoded = by.get(id(p))
            scale = float(p.activation_scale)
            rates.append(float(decoded.abs().mean()) / scale if decoded is not None else 0.0)
        return rates

    def test_deep_layer_firing_rate_rises(self):
        flow, x = build_cascade_flow(host_ops=False, depth=5, S=8, seed=0)
        before = self._firing_rates(flow, x, 8)
        apply_cascaded_gain_correction(flow, 8, rule="relative", input_data_scale=1.0)
        after = self._firing_rates(flow, x, 8)
        assert after[-1] > before[-1], (
            f"deep-layer firing rate must rise after correction (revival): "
            f"{before} -> {after}"
        )


class TestBoundaryDominatedGuard:
    """gamma(S)^d models INTRA-segment ramp attenuation; on a boundary-dominated
    graph (all intra-segment depths <= 1) the per-hop drift is inflation, not
    attenuation, and the trim is mis-signed (T4 SS2b: -0.4/-1.6 pp measured).
    Requesting gain correction there must fail loud, both cold and ramp."""

    def test_host_op_graph_rejects_cold_correction(self):
        flow, _ = build_cascade_flow(host_ops=True, depth=4, S=8, seed=0)
        depths = per_perceptron_cascade_depth(flow.get_mapper_repr())
        assert max(depths.values()) <= 1  # fixture is boundary-dominated
        with pytest.raises(ValueError, match="boundary-dominated"):
            apply_cascaded_gain_correction(flow, 8, rule="relative", input_data_scale=1.0)

    def test_host_op_graph_rejects_ramp_factors(self):
        flow, _ = build_cascade_flow(host_ops=True, depth=4, S=8, seed=0)
        with pytest.raises(ValueError, match="boundary-dominated"):
            cascaded_gain_factors(flow, 8, rule="relative")

    def test_two_layer_chain_rejects(self):
        model = make_tiny_supermodel()  # depths [0, 1]
        with pytest.raises(ValueError, match="boundary-dominated"):
            cascaded_gain_factors(model, 8, rule="relative")

    def test_deep_intra_segment_graph_still_corrects(self):
        flow, _ = build_cascade_flow(host_ops=False, depth=4, S=8, seed=0)
        stats = apply_cascaded_gain_correction(flow, 8, rule="relative", input_data_scale=1.0)
        assert stats["n_corrected"] >= 1

    def test_guard_names_the_depths(self):
        flow, _ = build_cascade_flow(host_ops=True, depth=3, S=8, seed=0)
        with pytest.raises(ValueError, match="max intra-segment depth"):
            cascaded_gain_factors(flow, 8, rule="relative")
