"""Currency coherence (spiking_deployment_calculus.md §11.2): one writer, four
equal gauges per re-encoded edge.

Locks: (1) an ARMED op's boundary out-scale IS its buffer gauge
(``output_scale``) in BOTH walks (the pure table delegated to the polymorphic
walk — the §10.4 κ_T/κ_S split closed); (2) the consumer's trained entry
currency equals the chain-final table currency (one-writer convergence);
(3) the install-seam coherence certificate fails loud on a mismatch; (4) the
LIF walk's re-encode divides ABSOLUTE (raw, unarmed-chain) values first —
the SSOT ``normalize_boundary_value`` — instead of clamp-then-scale, while
wire-domain producers keep the clamp path; (5) mixed wire/absolute fan-in at
a plain host op fails loud.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
from mimarsinan.mapping.mappers.structural import InputMapper
from mimarsinan.mapping.model_representation import ModelRepresentation
from mimarsinan.mapping.support.per_source_scales import compute_per_source_scales
from mimarsinan.models.nn.activations import LIFActivation
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.spiking.wire_semantics import lif_count_staircase
from mimarsinan.spiking.scale_aware_boundaries import (
    propagate_boundary_input_scales,
    read_boundary_out_scales,
    verify_boundary_currency_coherence,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.encoding_layers import mark_encoding_layers
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW

T = 32


def _lif_perceptron(out_ch, in_features, theta):
    p = Perceptron(out_ch, in_features, normalization=nn.Identity())
    p.set_activation_scale(theta)
    lif = LIFActivation(T=T, activation_scale=p.activation_scale)
    p.base_activation = lif
    p.activation = lif
    return p


def _mean(v) -> float:
    return float(torch.as_tensor(v).detach().to(torch.float64).mean())


def _armed_chain_model(cover: float = 2.5):
    """input -> P1(1.7) -> armed LN chain (LN1 -> LN2 with a traffic cover on
    LN2) -> P2 entry: the sigma-install shape at unit scale."""
    torch.manual_seed(0)
    inp = InputMapper((8,))
    p1 = _lif_perceptron(6, 8, 1.7)
    m1 = PerceptronMapper(inp, p1)
    ln1 = nn.LayerNorm(6)
    host1 = ComputeOpMapper(m1, ln1, input_shape=(6,), output_shape=(6,))
    ln2 = nn.LayerNorm(6)
    host2 = ComputeOpMapper(host1, ln2, input_shape=(6,), output_shape=(6,))
    p2 = _lif_perceptron(3, 6, 0.9)
    repr_ = ModelRepresentation(PerceptronMapper(host2, p2))
    mark_encoding_layers(repr_, placement="offload")
    host2.boundary_traffic_scale = float(cover)
    compute_per_source_scales(repr_)
    propagate_boundary_input_scales(repr_, input_data_scale=1.0)
    return repr_, p1, p2, host1, host2


class TestOneWriterCoherence:
    def test_armed_node_table_scale_is_its_buffer_gauge(self):
        repr_, _, _, host1, host2 = _armed_chain_model(cover=2.5)
        assert host2.output_scale is not None
        table = read_boundary_out_scales(repr_, input_data_scale=1.0)
        assert _mean(table[host2]) == pytest.approx(_mean(host2.output_scale))
        assert _mean(table[host1]) == pytest.approx(_mean(host1.output_scale))
        # The cover genuinely lifted the gauge above theta pass-through, so
        # this lock is not vacuous.
        assert _mean(host2.output_scale) == pytest.approx(2.5)

    def test_entry_currency_equals_chain_final_table_currency(self):
        repr_, _, p2, _, host2 = _armed_chain_model(cover=2.5)
        table = read_boundary_out_scales(repr_, input_data_scale=1.0)
        assert _mean(p2.input_activation_scale) == pytest.approx(
            _mean(table[host2])
        )

    def test_coherence_certificate_passes_then_fails_loud(self):
        repr_, _, p2, _, _ = _armed_chain_model(cover=2.5)
        verify_boundary_currency_coherence(repr_, input_data_scale=1.0)
        p2.set_input_activation_scale(torch.tensor(7.31))
        with pytest.raises(RuntimeError, match="coherence"):
            verify_boundary_currency_coherence(repr_, input_data_scale=1.0)


class TestRepresentationDispatch:
    def _absolute_chain_model(self):
        """input(kappa=2) -> UNARMED signed host op with mass above 1 -> entry:
        the entry-1 topology (unity source gauges keep the op unarmed)."""
        torch.manual_seed(1)
        inp = InputMapper((8,))
        ln = nn.LayerNorm(8)
        with torch.no_grad():
            ln.weight.fill_(1.0)
            ln.bias.fill_(1.2)  # pushes real mass into (1, kappa]
        host = ComputeOpMapper(inp, ln, input_shape=(8,), output_shape=(8,))
        p = _lif_perceptron(4, 8, 1.1)
        repr_ = ModelRepresentation(PerceptronMapper(host, p))
        mark_encoding_layers(repr_, placement="offload")
        compute_per_source_scales(repr_)
        propagate_boundary_input_scales(repr_, input_data_scale=2.0)
        assert host.output_scale is None, "fixture must stay unarmed"
        return repr_, host, p

    def test_absolute_producer_re_encode_divides_first(self):
        repr_, host, p = self._absolute_chain_model()
        torch.manual_seed(2)
        x = 2.0 * torch.rand(64, 8)
        driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with torch.no_grad():
            nf = driver(x)
            v = host.module(x)
            kappa = float(p.input_activation_scale)
            grid = torch.round((v / kappa).clamp(0.0, 1.0) * T) / T * kappa
            z = nn.functional.linear(grid, p.layer.weight, p.layer.bias)
            ref = lif_count_staircase(
                z, p.activation.activation_scale, T, compare_mode="<=",
            ).clamp(min=0.0)
        assert float((v > 1.0).float().mean()) > 0.05, "fixture must exercise v>1"
        bound = kappa / (2 * T)
        assert float((nf - ref).abs().mean()) <= bound

    def test_unity_gauge_prearm_is_currency_inert(self):
        """[B4 completion, calculus §15.7] the sigma-installer's pre-arm gives
        a marked-but-unarmed unity-gauge op wrap slots at the PASS-THROUGH
        currency: the walk output is IDENTICAL before and after (the armed
        wire path clamp(v/kappa)*kappa equals the absolute divide-first path),
        and sigma can now transport through the slots."""
        from mimarsinan.spiking.scale_aware_boundaries import (
            stamped_input_boundary_scale,
        )
        from mimarsinan.tuning.orchestration.signed_seam_install import (
            _prearm_marked_value_ops,
        )

        repr_, host, _ = self._absolute_chain_model()
        host.is_wire_value_op = True
        torch.manual_seed(4)
        x = 2.0 * torch.rand(32, 8)
        driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with torch.no_grad():
            before = driver(x)

        table = read_boundary_out_scales(
            repr_, input_data_scale=stamped_input_boundary_scale(repr_),
        )
        armed = _prearm_marked_value_ops(repr_, table)
        assert armed == 1
        assert host.per_source_scales is not None
        assert float(torch.as_tensor(host.output_scale).mean()) == pytest.approx(
            float(table[host])
        )
        # The table still reads the same currency (armed-term == pass-through).
        after_table = read_boundary_out_scales(
            repr_, input_data_scale=stamped_input_boundary_scale(repr_),
        )
        assert float(after_table[host]) == pytest.approx(float(table[host]))

        driver2 = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with torch.no_grad():
            after = driver2(x)
        torch.testing.assert_close(after, before)

    def test_sigma_install_skips_bake_infeasible_ops(self):
        """[σ-scope law, calculus §15.7] an armed op whose consumer walk hits a
        non-compensable host (the ViT stem's cat class) is SKIPPED by the
        σ-install — left trained-clamp exactly as today — instead of crashing
        the whole install."""
        from mimarsinan.mapping.support.compute_modules import ComputeAdapter
        from mimarsinan.tuning.orchestration.signed_seam_install import (
            _bake_walk_feasible,
        )

        torch.manual_seed(5)
        inp = InputMapper((8,))
        p1 = _lif_perceptron(6, 8, 1.7)
        m1 = PerceptronMapper(inp, p1)
        ln = nn.LayerNorm(6)
        host = ComputeOpMapper(m1, ln, input_shape=(6,), output_shape=(6,))
        blocker = ComputeOpMapper(
            host, ComputeAdapter(torch.square), input_shape=(6,), output_shape=(6,),
        )
        p2 = _lif_perceptron(3, 6, 0.9)
        repr_ = ModelRepresentation(PerceptronMapper(blocker, p2))
        mark_encoding_layers(repr_, placement="offload")
        compute_per_source_scales(repr_)
        consumers = repr_.consumer_map()
        assert not _bake_walk_feasible(host, consumers)
        assert _bake_walk_feasible(blocker, consumers)  # entry consumer bakes

    def test_mixed_wire_absolute_fan_in_fails_loud(self):
        from mimarsinan.mapping.support.compute_modules import ComputeAdapter

        torch.manual_seed(3)
        inp = InputMapper((6,))
        p1 = _lif_perceptron(6, 6, 1.3)
        m1 = PerceptronMapper(inp, p1)
        adder = ComputeAdapter(torch.add)
        mixed = ComputeOpMapper(
            [m1, inp], adder, input_shapes=[(6,), (6,)], output_shape=(6,),
        )
        p2 = _lif_perceptron(3, 6, 0.9)
        repr_ = ModelRepresentation(PerceptronMapper(mixed, p2))
        mark_encoding_layers(repr_, placement="offload")
        driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with pytest.raises(NotImplementedError, match="wire/absolute"):
            with torch.no_grad():
                driver(torch.rand(4, 6))
