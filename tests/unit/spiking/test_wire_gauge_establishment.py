"""Wire-gauge establishment (calculus §11.2/§16.6): the seam that makes a
heterogeneous fan-in decodable.

A residual that leaves the host in the ABSOLUTE domain and re-joins a branch
that crossed a neural core carries two currencies at one plain host op. The
graph is only executable once every non-homogeneous host op is ARMED — each
source decoded at its OWN producer gauge (kappa_T == kappa_S) through the
``ScaleNormalizingWrapper`` that IR emission installs. ``establish_wire_gauge``
is that one seam: the twin walk and the deployed op then share one definition
by construction.

Locks: (1) an unarmed residual seam still fails LOUD (the guard survives);
(2) establishment arms it at the two producers' true gauges and the LIF twin
runs; (3) the armed seam value recovers the ABSOLUTE sum exactly; (4) the
install-seam coherence certificate passes; (5) the repair is scoped to
unclassifiable graphs, so a classifiable one is byte-identical.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.support.value_domain import heterogeneous_domain_joins
from mimarsinan.spiking.scale_aware_boundaries import (
    establish_gauge_for_mixed_domain_seams,
    establish_wire_gauge,
    read_boundary_out_scales,
    verify_boundary_currency_coherence,
)
from mimarsinan.spiking.segment_forward import LifSegmentPolicy, SegmentForwardDriver
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW

T = 8


class _ResidualStem(nn.Module):
    """The ViT block shape at minimum size: a host stem (LayerNorm) whose value
    skips the neural core (Linear+ReLU) and re-joins after a host Linear."""

    def __init__(self, d: int = 8, n: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d, d)
        self.head = nn.Linear(d, n)

    def forward(self, x):
        x = x.flatten(1)
        branch = self.fc2(self.act(self.fc1(self.norm(x))))
        return self.head(x + branch)


def _residual_flow(seed: int = 0):
    torch.manual_seed(seed)
    model = _ResidualStem()
    return convert_torch_model(
        model, (1, 1, 8), 4, device="cpu", encoding_layer_placement="offload",
    )


def _named(repr_, name: str):
    return next(
        n for n in repr_.execution_order()
        if isinstance(n, ComputeOpMapper) and n.name == name
    )


def _lif_thetas(flow, theta: float = 1.7):
    """Give every perceptron a LIF activation so the twin has a real cascade."""
    from mimarsinan.models.nn.activations import LIFActivation

    for p in flow.get_perceptrons():
        p.set_activation_scale(torch.tensor(theta))
        lif = LIFActivation(T=T, activation_scale=p.activation_scale)
        p.base_activation = lif
        p.activation = lif
    return flow


def _mean(v) -> float:
    return float(torch.as_tensor(v).detach().to(torch.float64).mean())


class TestUnarmedSeamFailsLoud:
    def test_residual_seam_without_establishment_raises(self):
        flow = _lif_thetas(_residual_flow())
        driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with pytest.raises(NotImplementedError, match="wire/absolute"):
            with torch.no_grad():
                driver(torch.rand(4, 1, 1, 8))


class TestEstablishmentArmsTheSeam:
    def test_seam_decodes_each_source_at_its_producer_gauge(self):
        flow = _lif_thetas(_residual_flow())
        repr_ = flow.get_mapper_repr()
        establish_wire_gauge(flow, input_data_scale=1.0)

        seam = _named(repr_, "add")
        assert seam.per_source_scales is not None, "the residual seam must arm"
        table = read_boundary_out_scales(repr_, input_data_scale=1.0)
        deps = repr_._deps[seam]
        for i, dep in enumerate(deps):
            assert _mean(seam.per_source_scales[i]) == pytest.approx(
                float(table[dep])
            ), "kappa_T must equal the producer's kappa_S at a mixed fan-in"
        # Genuinely heterogeneous: the two sources do NOT share a gauge.
        assert _mean(seam.per_source_scales[0]) != pytest.approx(
            _mean(seam.per_source_scales[1])
        )

    def test_lif_twin_runs_on_the_established_graph(self):
        flow = _lif_thetas(_residual_flow())
        establish_wire_gauge(flow, input_data_scale=1.0)
        driver = SegmentForwardDriver(flow.get_mapper_repr(), T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with torch.no_grad():
            out = driver(torch.rand(4, 1, 1, 8))
        assert out.shape == (4, 4)
        assert torch.isfinite(out).all()

    def test_armed_seam_value_recovers_the_absolute_sum(self):
        """The seam's stored wire value, decoded by its own gauge, equals the
        ABSOLUTE residual sum — raw domain-mixing would be off by the branch
        producer's theta."""
        flow = _lif_thetas(_residual_flow())
        repr_ = flow.get_mapper_repr()
        establish_wire_gauge(flow, input_data_scale=1.0)
        seam = _named(repr_, "add")
        fc2 = _named(repr_, "fc2")

        torch.manual_seed(11)
        x = torch.rand(4, 1, 1, 8)
        joins: dict = {}
        decoded: dict = {}
        driver = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))
        with torch.no_grad():
            driver(x, join_value_recorder=joins, node_value_recorder=decoded)

        perceptron = next(iter(flow.get_perceptrons()))
        branch_value = fc2.module(decoded[id(perceptron)])
        expected = x.flatten(1) + branch_value
        recovered = joins[seam] * seam.output_scale
        torch.testing.assert_close(recovered, expected)

    def test_coherence_certificate_passes_after_establishment(self):
        flow = _lif_thetas(_residual_flow())
        establish_wire_gauge(flow, input_data_scale=1.0)
        verify_boundary_currency_coherence(flow, input_data_scale=1.0)


class _FeedForward(nn.Module):
    """The covered-cell shape: no residual, so every fan-in is single-source
    and the domain map classifies without any gauge establishment."""

    def __init__(self, d: int = 8, n: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.act = nn.ReLU()
        self.head = nn.Linear(d, n)

    def forward(self, x):
        return self.head(self.act(self.fc1(x.flatten(1))))


class TestRepairIsScopedToUnclassifiableGraphs:
    def test_repair_reports_every_mixed_seam(self):
        flow = _lif_thetas(_residual_flow())
        assert len(heterogeneous_domain_joins(flow.get_mapper_repr())) == 1
        assert establish_gauge_for_mixed_domain_seams(
            flow, input_data_scale=1.0,
        ) == 1
        assert not heterogeneous_domain_joins(flow.get_mapper_repr()), (
            "establishment must leave the graph classifiable"
        )

    def test_a_non_unit_join_never_reaches_the_unity_fallback(self, monkeypatch):
        """The fallback's unit gauge is forced, not chosen: any producer at a
        non-unit gauge makes the fan-in non-uniform (or trips the value-op
        wrap), so ``compute_per_source_scales`` arms the join FIRST, at the
        producers' true currencies. Refusing the fallback outright is the pin —
        without it the fallback's gauge would be untestable, because it can
        only ever see 1.0."""
        flow = _lif_thetas(_residual_flow())  # theta 1.7: the gauges differ
        import mimarsinan.spiking.scale_aware_boundaries as scale_aware_boundaries

        def _refuse(node):
            raise AssertionError(
                f"the unity fallback must not see {getattr(node, 'name', node)!r}: "
                "a join with non-unit producer gauges belongs to the policy"
            )

        monkeypatch.setattr(scale_aware_boundaries, "_arm_domain_join", _refuse)
        assert establish_gauge_for_mixed_domain_seams(
            flow, input_data_scale=1.0,
        ) == 1

    def test_unity_gauge_seam_still_gets_a_domain(self):
        """The necessity is STRUCTURAL: at unity gauges the arming policy
        declines (a unit wrapper is numerically inert), but the join still has
        no domain — so the repair arms it anyway, at the pass-through
        currencies. This is the ONLY branch the fallback is reachable on (see
        the test above), which is why its gauge is unity by construction."""
        flow = _lif_thetas(_residual_flow(), theta=1.0)
        repr_ = flow.get_mapper_repr()
        establish_wire_gauge(flow, input_data_scale=1.0)
        assert _named(repr_, "add").output_scale is None, (
            "the policy alone must leave a unity-gauge join unarmed"
        )
        assert establish_gauge_for_mixed_domain_seams(
            flow, input_data_scale=1.0,
        ) == 1
        seam = _named(repr_, "add")
        assert seam.per_source_scales is not None
        assert all(_mean(s) == pytest.approx(1.0) for s in seam.per_source_scales)
        # The EMITTED gauge is the one the fallback owns (the policy refreshes
        # per-source on the next sweep, but never the output scale), and it must
        # be the PASS-THROUGH: a repair whose only job is to classify a domain
        # may not rescale the wire it classifies — that would move the seam's
        # [0,1] clamp headroom and its consumers' weight fold with it.
        assert _mean(seam.output_scale) == pytest.approx(1.0)
        with torch.no_grad():
            out = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))(
                torch.rand(4, 1, 1, 8)
            )
        assert torch.isfinite(out).all()

    def test_classifiable_graph_is_left_untouched(self):
        """No mixed seam => the repair never runs the arming walk, so the twin
        output and every gauge slot are bit-for-bit unchanged: the
        covered-topology byte-identity guarantee, by construction."""
        torch.manual_seed(3)
        flow = _lif_thetas(convert_torch_model(
            _FeedForward(), (1, 1, 8), 4, device="cpu",
            encoding_layer_placement="offload",
        ))
        repr_ = flow.get_mapper_repr()
        head = _named(repr_, "head")
        assert head.module.bias is not None, "fixture must carry an armable op"
        torch.manual_seed(5)
        x = torch.rand(6, 1, 1, 8)
        with torch.no_grad():
            before = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))(x)

        assert establish_gauge_for_mixed_domain_seams(
            flow, input_data_scale=1.0,
        ) == 0
        assert head.output_scale is None and head.per_source_scales is None
        with torch.no_grad():
            after = SegmentForwardDriver(repr_, T, LifSegmentPolicy(soma_law=DEFAULT_SOMA_LAW))(x)
        assert torch.equal(after, before)

    def test_full_establishment_would_have_changed_that_graph(self):
        """The scoping is load-bearing, not vacuous: the unconditional walk DOES
        arm the terminal head there (which is why the repair must not run it)."""
        torch.manual_seed(3)
        flow = _lif_thetas(convert_torch_model(
            _FeedForward(), (1, 1, 8), 4, device="cpu",
            encoding_layer_placement="offload",
        ))
        establish_wire_gauge(flow, input_data_scale=1.0)
        assert _named(flow.get_mapper_repr(), "head").output_scale is not None
