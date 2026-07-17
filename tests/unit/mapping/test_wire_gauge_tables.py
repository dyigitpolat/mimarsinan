"""kappa_buf gauge table + derived boundary divisors (boundary algebra P2).

The buffer gauge kappa_buf makes today's implicit divisor reconstruction an
explicit SSOT: ``divisor = kappa_fold / kappa_buf``. In P2 the gauge
assignment is the LEGACY one (plain host ops pass their sources' gauge
through), so every produced divisor is value-identical to the historical
walk — the wire-currency contract locks (T2/T7) hold unchanged; P3 flips the
plain-op classification together with the NF side.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping_class import IRMapping
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
    build_hybrid_hard_core_mapping,
)
from mimarsinan.mapping.support.activation_scales import (
    compute_node_buffer_scales,
    compute_node_output_scales,
)
from mimarsinan.mapping.support.compute_modules import ScaleNormalizingWrapper
from mimarsinan.mapping.support.value_domain import op_preserves_wire_ratio
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.spiking.segment_boundary import boundary_normalization_scales
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers

T = 8
THETA_1 = 1.7
THETA_2 = 0.9


def _lif_perceptron(out_ch, in_features, theta, *, encoding=False):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.is_encoding_layer = encoding
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    lif.use_cycle_accurate_trains = True
    p.base_activation = lif
    p.activation = lif
    return p


class _HostRelay(nn.Module):
    def forward(self, x):
        return x * 1.0


def _chain_model(*, placement: str, host_module: nn.Module):
    """input(8) -> P1(6, THETA_1) -> host op -> P2(3, THETA_2)."""
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, THETA_1, encoding=(placement == "subsume"))
    m1 = PerceptronMapper(inp, p1)
    host = ComputeOpMapper(m1, host_module, input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, THETA_2)
    p2.per_input_scales = torch.full((6,), float(THETA_1))
    m2 = PerceptronMapper(host, p2)
    repr_ = ModelRepresentation(m2)
    mark_encoding_layers(repr_, placement=placement)
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
    )
    return ir, hybrid


def _ir_ops(ir):
    return [n for n in ir.nodes if isinstance(n, ComputeOp)]


# ---------------------------------------------------------------------------
# The kappa_buf table (legacy gauge assignment).
# ---------------------------------------------------------------------------

class TestBufferScalesLegacyGauge:
    def test_neural_buffers_are_wire_gauge(self):
        ir, _ = _chain_model(placement="offload", host_module=nn.LayerNorm(6))
        fold = compute_node_output_scales(ir)
        buf = compute_node_buffer_scales(ir)
        for node in ir.nodes:
            if isinstance(node, NeuralCore):
                # counts/T buffer: kappa_buf == kappa_fold (divisor 1).
                assert buf[node.id] == pytest.approx(fold[node.id])

    def test_wrapped_compute_op_buffer_is_value_gauge(self):
        ir, _ = _chain_model(placement="subsume", host_module=_HostRelay())
        buf = compute_node_buffer_scales(ir)
        wrapped = [
            n for n in _ir_ops(ir)
            if isinstance((n.params or {}).get("module"), Perceptron)
        ]
        assert wrapped
        for op in wrapped:
            assert buf[op.id] == pytest.approx(1.0)

    def test_plain_op_passes_source_gauge_through(self):
        ir, _ = _chain_model(placement="offload", host_module=nn.LayerNorm(6))
        fold = compute_node_output_scales(ir)
        buf = compute_node_buffer_scales(ir)
        plain = [
            n for n in _ir_ops(ir)
            if isinstance((n.params or {}).get("module"), nn.LayerNorm)
        ]
        assert plain
        for op in plain:
            # LEGACY gauge: the plain op inherits its neural source's wire
            # gauge, so kappa_fold/kappa_buf == 1 (P3 flips this to
            # kappa_buf == 1 for non-homogeneous value ops).
            assert buf[op.id] == pytest.approx(fold[op.id])

    def test_scale_normalizing_wrapper_is_wire_gauge(self):
        producer = NeuralCore(
            id=0, name="n0",
            input_sources=np.array([], dtype=object),
            activation_scale=torch.tensor(float(THETA_1)),
        )
        wrapper = ScaleNormalizingWrapper(
            nn.Identity(),
            input_scales=[torch.tensor(1.0)],
            output_scale=torch.tensor(1.0),
        )
        op = ComputeOp(
            id=1, name="wrapped",
            input_sources=np.array([[IRSource(0, 0)]], dtype=object),
            op_type="module", params={"module": wrapper},
        )
        ir = IRGraph(
            nodes=[producer, op], output_sources=np.array([], dtype=object),
        )
        fold = compute_node_output_scales(ir)
        buf = compute_node_buffer_scales(ir)
        assert buf[1] == pytest.approx(
            float(np.mean(np.asarray(fold[1], dtype=np.float64)))
        )


# ---------------------------------------------------------------------------
# The derived divisor view: value-identical to the legacy walk.
# ---------------------------------------------------------------------------

class TestDerivedDivisorView:
    def test_subsume_chain_divisor_carries_wrapped_theta(self):
        ir, hybrid = _chain_model(placement="subsume", host_module=_HostRelay())
        divisors = boundary_normalization_scales(hybrid)
        wrapped_ids = {
            n.id for n in _ir_ops(ir)
            if isinstance((n.params or {}).get("module"), Perceptron)
        }
        assert wrapped_ids
        for op_id in wrapped_ids:
            assert float(np.mean(np.asarray(divisors[op_id]))) == pytest.approx(
                THETA_1
            )

    def test_plain_neural_fed_op_has_no_divisor_entry(self):
        # The P1 T1 xfail pins the HOLE; P2 must not silently flip it.
        _, hybrid = _chain_model(placement="offload", host_module=nn.LayerNorm(6))
        assert boundary_normalization_scales(hybrid) == {}

    def test_downstream_plain_relay_inherits_wrapped_divisor(self):
        torch.manual_seed(0)
        inp = InputMapper((8,))
        p1 = _lif_perceptron(6, 8, THETA_1, encoding=True)
        m1 = PerceptronMapper(inp, p1)
        relay = ComputeOpMapper(m1, _HostRelay(), input_shape=(6,), output_shape=(6,))
        p2 = _lif_perceptron(3, 6, THETA_2)
        p2.per_input_scales = torch.full((6,), float(THETA_1))
        m2 = PerceptronMapper(relay, p2)
        repr_ = ModelRepresentation(m2)
        mark_encoding_layers(repr_)
        ir = IRMapping(
            q_max=127.0, firing_mode="Default", max_axons=32, max_neurons=32,
        ).map(repr_)
        hybrid = build_hybrid_hard_core_mapping(
            ir_graph=ir,
            cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 4}],
        )
        divisors = boundary_normalization_scales(hybrid)
        relay_ids = [
            n.id for n in _ir_ops(ir)
            if not isinstance((n.params or {}).get("module"), Perceptron)
        ]
        for op_id in relay_ids:
            assert float(np.mean(np.asarray(divisors[op_id]))) == pytest.approx(
                THETA_1
            )

    def test_builders_stamp_buffer_scales(self):
        ir, hybrid = _chain_model(placement="subsume", host_module=_HostRelay())
        assert hybrid.node_buffer_scales
        expected = compute_node_buffer_scales(ir)
        assert set(hybrid.node_buffer_scales) == set(expected)

    def test_unstamped_mapping_falls_back_to_legacy_walk(self):
        # Pre-P2 pickles lack the stamp; the divisor view must still resolve.
        _, hybrid = _chain_model(placement="subsume", host_module=_HostRelay())
        with_stamp = boundary_normalization_scales(hybrid)
        del hybrid.__dict__["node_buffer_scales"]
        without_stamp = boundary_normalization_scales(hybrid)
        assert set(with_stamp) == set(without_stamp)
        for op_id, div in with_stamp.items():
            np.testing.assert_allclose(
                np.asarray(div, dtype=np.float64),
                np.asarray(without_stamp[op_id], dtype=np.float64),
            )


# ---------------------------------------------------------------------------
# The wire_transparent classifier (grounded in homogeneity; consulted in P3).
# ---------------------------------------------------------------------------

class TestOpPreservesWireRatio:
    @pytest.mark.parametrize("op", [
        nn.MaxPool2d(2), nn.AvgPool2d(2), nn.AdaptiveAvgPool2d(1),
        nn.AdaptiveMaxPool2d(1), nn.Identity(), nn.Flatten(),
    ])
    def test_allowlisted_ops_are_homogeneous(self, op):
        assert op_preserves_wire_ratio(op)
        torch.manual_seed(3)
        x = torch.randn(2, 3, 8, 8)
        for alpha in (0.5, 2.5):
            torch.testing.assert_close(
                op(alpha * x), alpha * op(x), atol=1e-6, rtol=1e-6,
            )

    def test_bias_free_linear_is_transparent(self):
        assert op_preserves_wire_ratio(nn.Linear(4, 3, bias=False))

    @pytest.mark.parametrize("op", [
        nn.LayerNorm(6), nn.GELU(), nn.Softmax(dim=-1), nn.Linear(4, 3),
    ])
    def test_value_ops_are_not_transparent(self, op):
        assert not op_preserves_wire_ratio(op)

    def test_unknown_module_answers_false(self):
        class _Mystery(nn.Module):
            def forward(self, x):
                return x + 1.0

        assert not op_preserves_wire_ratio(_Mystery())

    def test_compute_adapter_transparent_only_for_homogeneous_fn(self):
        from mimarsinan.mapping.support.compute_modules import ComputeAdapter

        assert op_preserves_wire_ratio(ComputeAdapter(torch.mean))
        assert not op_preserves_wire_ratio(ComputeAdapter(torch.sigmoid))


class TestValueOpArmingGeometry:
    """Uniform gauges arm with 1-element SCALAR slots: a source-sized vector
    cannot normalize a shape-changing op's output (token-mixing Linear), and
    emission geometry (per-instance split, orientation) follows the wrapped
    payload, not the wrapper."""

    def test_uniform_value_op_arms_scalar_slots(self):
        torch.manual_seed(0)
        inp = InputMapper((8,))
        p1 = _lif_perceptron(6, 8, THETA_1)
        m1 = PerceptronMapper(inp, p1)
        host = ComputeOpMapper(m1, nn.Linear(6, 3), input_shape=(6,), output_shape=(3,))
        p2 = _lif_perceptron(3, 3, THETA_2)
        m2 = PerceptronMapper(host, p2)
        repr_ = ModelRepresentation(m2)
        mark_encoding_layers(repr_, placement="offload")
        from mimarsinan.mapping.support.per_source_scales import (
            compute_per_source_scales,
        )

        compute_per_source_scales(repr_)
        assert host.per_source_scales is not None
        assert tuple(host.output_scale.shape) == (1,)
        assert all(tuple(s.shape) == (1,) for s in host.per_source_scales)
        assert float(host.output_scale) == pytest.approx(THETA_1)
        # The shape-changing wrapped op must forward cleanly on values.
        with torch.no_grad():
            out = host.forward_scale_normalized(torch.rand(4, 6))
        assert tuple(out.shape) == (4, 3)

    def test_emission_geometry_unwraps_the_wrapper(self):
        wrapped = ScaleNormalizingWrapper(
            nn.Linear(6, 3),
            input_scales=[torch.tensor([1.7])],
            output_scale=torch.tensor([1.7]),
        )
        assert ComputeOpMapper._is_per_instance_module(wrapped)
        src = np.empty((6, 5), dtype=object)
        oriented = ComputeOpMapper._orient_2d_for_columns(src, wrapped)
        assert oriented.shape == (6, 5)
