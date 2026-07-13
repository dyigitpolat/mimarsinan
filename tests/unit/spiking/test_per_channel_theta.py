"""[R3/S2] Per-channel theta promotion: exact scale-space reallocation on
matching-axis edges.

``Q_S(z/theta)`` depends on (z, theta) only through z/theta, so a per-channel
theta with per-channel decode is an EXACT identity (sync memo §4.2). The
promotion is only sound where the producer's channel axis is consumed as the
consumer's feature axis (the M4 matching-axis condition); weight-shared /
axis-flipped hops must keep the scalar (mixer memo §3.3 / §6 escape).
"""

import pytest
import torch
import torch.nn as nn

from conftest import make_tiny_supermodel

from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    ModelRepresentation,
    PerceptronMapper,
)
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper
from mimarsinan.mapping.mappers.structural import PermuteMapper
from mimarsinan.mapping.support.compute_modules import ComputeAdapter
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.spiking.per_channel_theta import (
    PerChannelThetaReport,
    eligible_per_channel_perceptrons,
    maybe_promote_per_channel_theta,
    per_channel_theta_armed,
    per_channel_theta_vector,
    promote_per_channel_theta,
)


class _FlowShell(PerceptronFlow):
    """Minimal PerceptronFlow around a hand-built mapper graph."""

    def __init__(self, mapper_repr):
        super().__init__("cpu")
        self.input_activation = nn.Identity()
        self._mapper_repr = mapper_repr

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def get_input_activation(self):
        return self.input_activation

    def set_input_activation(self, activation):
        self.input_activation = activation

    def forward(self, x):
        return self._mapper_repr(x)


def _aligned_chain_model():
    """Encoder(16) -> hidden(16->16) -> head(16->4): one matching-axis edge."""
    return make_tiny_supermodel(hidden_layers=2)


def _axis_flip_model(tokens=4, channels=4):
    """Producer whose channel axis is permuted into the consumer's TOKEN axis.

    Dims are chosen EQUAL so a divisibility heuristic would wrongly admit it —
    only the structural axis walk rejects it.
    """
    p_prod = Perceptron(channels, channels, name="prod")
    p_cons = Perceptron(tokens, tokens, name="cons")
    inp = InputMapper((tokens, channels))
    prod = PerceptronMapper(inp, p_prod)
    flipped = PermuteMapper(prod, (0, 2, 1))
    cons = PerceptronMapper(flipped, p_cons)
    head_p = Perceptron(2, tokens, name="head")
    head = PerceptronMapper(cons, head_p)
    model = _FlowShell(ModelRepresentation(head))
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, tokens, channels))
    return model, p_prod, p_cons


def _mean_readout_model(
    tokens=3, channels=6, classes=2, *, host_readout, tail_after_readout=False,
):
    """Producer -> mean(dim=1) -> readout [-> tail perceptron].

    ``host_readout=True`` consumes through a host nn.Linear module (the mixer
    readout shape — eligible); ``False`` consumes through a PerceptronMapper
    (a segment-entry seam past a ComputeOp — must stay scalar).
    ``tail_after_readout`` hangs a perceptron off the host readout, making the
    readout non-terminal (its wire re-enters a scalar-normalized seam).
    """
    p_prod = Perceptron(channels, channels, name="prod")
    inp = InputMapper((tokens, channels))
    prod = PerceptronMapper(inp, p_prod)
    mean = ComputeOpMapper(
        prod,
        ComputeAdapter(torch.mean, kwargs={"dim": 1}),
        input_shapes=[(tokens, channels)],
        name="mean",
    )
    if host_readout:
        readout = ComputeOpMapper(
            mean, nn.Linear(channels, classes), name="readout",
        )
        tail_perceptron = None
    else:
        tail_perceptron = Perceptron(classes, channels, name="readout")
        readout = PerceptronMapper(mean, tail_perceptron)
    head = readout
    if tail_after_readout:
        head = PerceptronMapper(readout, Perceptron(2, classes, name="tail"))
    model = _FlowShell(ModelRepresentation(head))
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, tokens, channels))
    return model, p_prod, tail_perceptron


def _names(perceptrons):
    return {p.name for p in perceptrons}


class TestEligibility:
    def test_matching_axis_hop_is_eligible(self):
        model = _aligned_chain_model()
        perceptrons = list(model.get_perceptrons())
        eligible = eligible_per_channel_perceptrons(model)
        # encoder excluded, head feeds the output unmediated; only the
        # hidden hop's output edge is matching-axis.
        assert _names(eligible.values()) == {perceptrons[1].name}

    def test_encoder_is_never_eligible(self):
        model = _aligned_chain_model()
        encoder = next(
            p for p in model.get_perceptrons() if p.is_encoding_layer
        )
        assert id(encoder) not in eligible_per_channel_perceptrons(model)

    def test_axis_flip_keeps_scalar_even_when_dims_divide(self):
        model, p_prod, p_cons = _axis_flip_model()
        eligible = eligible_per_channel_perceptrons(model)
        assert id(p_prod) not in eligible, (
            "axis-flipped producer admitted: the guard must be structural, "
            "not divisibility-based"
        )
        # The consumer's own output edge IS matching-axis (feeds head directly).
        assert id(p_cons) in eligible

    def test_host_module_readout_through_mean_is_eligible(self):
        model, p_prod, _ = _mean_readout_model(host_readout=True)
        assert id(p_prod) in eligible_per_channel_perceptrons(model)

    def test_perceptron_consumer_past_compute_op_stays_scalar(self):
        # A perceptron reached through a host ComputeOp is a segment-entry
        # boundary; its scalar snap normalizer cannot carry the vector yet.
        model, p_prod, _ = _mean_readout_model(host_readout=False)
        assert id(p_prod) not in eligible_per_channel_perceptrons(model)

    def test_non_terminal_host_module_readout_stays_scalar(self):
        # A host module with downstream consumers re-enters wire seams the NF
        # twin normalizes by scalars; only a terminal readout carries the vector.
        model, p_prod, _ = _mean_readout_model(
            host_readout=True, tail_after_readout=True,
        )
        assert id(p_prod) not in eligible_per_channel_perceptrons(model)

    def test_channels_first_producer_stays_scalar(self):
        model = _aligned_chain_model()
        hidden = list(model.get_perceptrons())[1]
        hidden.output_channel_axis = 1  # conv-style channels-first declaration
        try:
            assert id(hidden) not in eligible_per_channel_perceptrons(model)
        finally:
            hidden.output_channel_axis = -1


class TestVectorConstruction:
    def test_live_channels_take_their_own_quantile(self):
        vec = per_channel_theta_vector(2.0, [0.5, 4.0, 1.0])
        assert torch.allclose(vec, torch.tensor([0.5, 4.0, 1.0]))

    def test_dead_channels_fall_back_to_the_pooled_theta(self):
        vec = per_channel_theta_vector(2.0, [0.0, 3.0])
        assert torch.allclose(vec, torch.tensor([2.0, 3.0]))

    def test_tiny_positive_quantiles_are_floored(self):
        vec = per_channel_theta_vector(2.0, [1e-12, 1.0], min_scale=1e-6)
        assert vec[0].item() == pytest.approx(1e-6)


class TestPromotion:
    def _promote(self, model, spread=True):
        perceptrons = list(model.get_perceptrons())
        channels = []
        pooled = []
        for p in perceptrons:
            n = p.output_channels
            base = torch.linspace(0.5, 2.0, n) if spread else torch.ones(n)
            channels.append([float(v) for v in base])
            pooled.append(1.0)
        return promote_per_channel_theta(model, channels, pooled)

    def test_eligible_get_vectors_others_keep_scalars(self):
        model = _aligned_chain_model()
        perceptrons = list(model.get_perceptrons())
        report = self._promote(model)

        assert isinstance(report, PerChannelThetaReport)
        assert report.promoted == (perceptrons[1].name,)
        assert perceptrons[1].activation_scale.dim() == 1
        assert perceptrons[1].activation_scale.numel() == 16
        assert perceptrons[0].activation_scale.dim() == 0
        assert perceptrons[2].activation_scale.dim() == 0
        assert set(report.skipped) == {perceptrons[0].name, perceptrons[2].name}

    def test_promotion_keeps_parameter_identity(self):
        # Decorators hold the Parameter object; promotion must mutate .data,
        # never rebind (the stale-reference trap).
        model = _aligned_chain_model()
        hidden = list(model.get_perceptrons())[1]
        param_before = hidden.activation_scale
        self._promote(model)
        assert hidden.activation_scale is param_before

    def test_length_mismatch_fails_loud(self):
        model = _aligned_chain_model()
        perceptrons = list(model.get_perceptrons())
        channels = [[1.0] * p.output_channels for p in perceptrons]
        channels[1] = [1.0] * 3  # wrong width on the eligible hop
        with pytest.raises(ValueError, match="channel"):
            promote_per_channel_theta(
                model, channels, [1.0] * len(perceptrons)
            )

    def test_table_count_mismatch_fails_loud(self):
        model = _aligned_chain_model()
        with pytest.raises(ValueError, match="count"):
            promote_per_channel_theta(model, [[1.0]], [1.0])

    def test_promotion_arms_the_downstream_host_wrap_slots(self):
        """The install seam must stamp per_source_scales/output_scale on host
        ComputeOps downstream of a promoted producer, so the chip-aligned NF
        twin runs the SAME ScaleNormalizingWrapper decode the deployed sim
        emits (the t0_01 torch<->deployed parity break)."""
        model, p_prod, _ = _mean_readout_model(host_readout=True)
        perceptrons = list(model.get_perceptrons())
        channels = [
            [1.0 + 0.1 * c for c in range(p.output_channels)]
            for p in perceptrons
        ]
        report = promote_per_channel_theta(
            model, channels, [1.0] * len(perceptrons)
        )
        assert p_prod.name in report.promoted

        ops = [
            n for n in model.get_mapper_repr()._exec_order
            if isinstance(n, ComputeOpMapper)
        ]
        armed = [
            n for n in ops
            if n.per_source_scales is not None and n.output_scale is not None
        ]
        assert armed, "promotion must arm the downstream host wrap slots"
        # The weight-fold currency stays owned by compute_per_source_scales.
        assert all(
            getattr(p, "per_input_scales", None) is None for p in perceptrons
        )


def _armed_cfg(mode, schedule=None, flag=True):
    cfg = {"per_channel_theta": flag, "spiking_mode": mode, "simulation_steps": 4}
    if schedule is not None:
        cfg["ttfs_cycle_schedule"] = schedule
    return cfg


class TestArmedPredicate:
    def test_lif_and_synchronized_arm(self):
        assert per_channel_theta_armed(_armed_cfg("lif"))
        assert per_channel_theta_armed(
            _armed_cfg("ttfs_cycle_based", "synchronized")
        )

    def test_other_modes_and_flag_off_do_not_arm(self):
        assert not per_channel_theta_armed(_armed_cfg("lif", flag=False))
        assert not per_channel_theta_armed(_armed_cfg("ttfs_quantized"))
        assert not per_channel_theta_armed(_armed_cfg("ttfs"))
        assert not per_channel_theta_armed(
            _armed_cfg("ttfs_cycle_based", "cascaded")
        )


class TestMaybePromote:
    def _stats(self, model):
        return {
            "per_channel_quantiles": {
                "quantile": 0.99,
                "channels": [
                    [1.0 + 0.1 * c for c in range(p.output_channels)]
                    for p in model.get_perceptrons()
                ],
            }
        }

    def test_armed_promotes_and_reports(self, capsys):
        model = _aligned_chain_model()
        cfg = _armed_cfg("lif")
        pooled = [1.0] * len(list(model.get_perceptrons()))
        report = maybe_promote_per_channel_theta(
            cfg, model, self._stats(model), pooled
        )
        assert report is not None and len(report.promoted) == 1
        assert "[PerChannelTheta]" in capsys.readouterr().out

    def test_unarmed_is_a_noop(self):
        model = _aligned_chain_model()
        pooled = [1.0] * len(list(model.get_perceptrons()))
        report = maybe_promote_per_channel_theta(
            _armed_cfg("lif", flag=False), model, self._stats(model), pooled
        )
        assert report is None
        assert all(p.activation_scale.dim() == 0 for p in model.get_perceptrons())

    def test_armed_without_captured_stats_fails_loud(self):
        model = _aligned_chain_model()
        cfg = _armed_cfg("lif")
        with pytest.raises(ValueError, match="per-channel"):
            maybe_promote_per_channel_theta(cfg, model, {}, [1.0, 1.0, 1.0])


class TestClampInstallSeam:
    """The LIF/sync theta install SSOT (ClampTuner scale install) promotes
    eligible hops right after the scalar install, inside one seam."""

    def _install(self, model, cfg, stats):
        from types import SimpleNamespace

        from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

        stub = SimpleNamespace(
            pipeline=SimpleNamespace(config=cfg), model=model,
        )
        scales = [1.0] * len(list(model.get_perceptrons()))
        return ClampTuner._calculate_activation_scales(stub, scales, stats), stub

    def _cfg(self, armed=True):
        cfg = {
            "spiking_mode": "lif",
            "simulation_steps": 4,
            "target_tq": 4,
            "input_shape": (1, 8, 8),
            "num_classes": 4,
        }
        if armed:
            cfg["per_channel_theta"] = True
        return cfg

    def _stats(self, model):
        return {
            "per_channel_quantiles": {
                "quantile": 0.99,
                "channels": [
                    [1.0 + 0.1 * c for c in range(p.output_channels)]
                    for p in model.get_perceptrons()
                ],
            }
        }

    def test_armed_install_promotes_the_eligible_hop(self):
        model = _aligned_chain_model()
        _, stub = self._install(model, self._cfg(), self._stats(model))
        hidden = list(model.get_perceptrons())[1]
        assert hidden.activation_scale.dim() == 1
        assert stub.per_channel_theta_report is not None

    def test_unarmed_install_stays_scalar(self):
        model = _aligned_chain_model()
        diagnostics, stub = self._install(
            model, self._cfg(armed=False), self._stats(model)
        )
        assert all(
            p.activation_scale.dim() == 0 for p in model.get_perceptrons()
        )
        assert stub.per_channel_theta_report is None
        assert len(diagnostics) == 3


class TestExactnessContract:
    """The task's contract: all-channels-equal theta == scalar theta bit-identical;
    unequal theta preserves the float function through the NF fold."""

    def _staircase_forward(self, perceptron, x, steps=4):
        from mimarsinan.models.nn.decorators.clamp_quantize import (
            ClampDecorator,
            TTFSCeilStaircaseDecorator,
        )
        from mimarsinan.models.nn.layers import TransformedActivation

        perceptron.set_activation(
            TransformedActivation(
                perceptron.base_activation,
                [
                    ClampDecorator(
                        torch.tensor(0.0), perceptron.activation_scale
                    ),
                    TTFSCeilStaircaseDecorator(
                        steps, perceptron.activation_scale
                    ),
                ],
            )
        )
        with torch.no_grad():
            return perceptron(x)

    def test_uniform_vector_equals_scalar_bit_identical(self):
        torch.manual_seed(0)
        theta = 1.7
        x = torch.randn(8, 16)

        p_scalar = Perceptron(16, 16, name="s")
        p_vector = Perceptron(16, 16, name="v")
        p_vector.load_state_dict(p_scalar.state_dict())

        p_scalar.set_activation_scale(theta)
        p_vector.set_activation_scale(torch.full((16,), theta))

        out_scalar = self._staircase_forward(p_scalar, x)
        out_vector = self._staircase_forward(p_vector, x)
        assert torch.equal(out_scalar, out_vector)

    def test_uniform_vector_equals_scalar_through_lif_activation(self):
        from mimarsinan.models.nn.activations import LIFActivation

        torch.manual_seed(1)
        x = torch.rand(4, 16) * 2.0
        lif_scalar = LIFActivation(T=4, activation_scale=1.3)
        lif_vector = LIFActivation(
            T=4, activation_scale=nn.Parameter(
                torch.full((16,), 1.3), requires_grad=False
            ),
        )
        with torch.no_grad():
            assert torch.equal(lif_scalar(x), lif_vector(x))

    def test_unequal_theta_preserves_float_function_through_the_fold(self):
        """Scale-space identity: producer decode theta_c folded into the
        consumer's per_input_scales reproduces the raw float pre-activation."""
        from mimarsinan.mapping.mappers.scale_propagation import (
            assign_per_input_scales,
        )
        from mimarsinan.transformations.perceptron.perceptron_transformer import (
            PerceptronTransformer,
        )

        torch.manual_seed(2)
        producer = Perceptron(6, 5, name="prod")
        consumer = Perceptron(3, 6, name="cons")
        theta = torch.linspace(0.5, 3.0, 6)
        producer.set_activation_scale(theta)
        assign_per_input_scales(consumer, theta)

        x = torch.randn(7, 5)
        with torch.no_grad():
            value = torch.relu(producer.layer(x))
            raw_preact = consumer.layer(value)

            # Deployed algebra: the wire carries value/theta_c; the consumer's
            # effective weights carry theta_c back (per_input_scales fold).
            wire = value / theta
            transformer = PerceptronTransformer()
            eff_w = transformer.get_effective_weight(consumer)
            eff_b = transformer.get_effective_bias(consumer)
            folded_preact = wire @ eff_w.T + eff_b

        assert torch.allclose(folded_preact, raw_preact, atol=1e-5)


class TestBoundaryWalkVectorTolerance:
    """The boundary-scale walk (the documented scalar parity contract) must not
    crash on a promoted model: a vector theta mean-collapses in BOTH walks."""

    def test_propagate_boundary_scales_on_promoted_model(self):
        from mimarsinan.spiking.scale_aware_boundaries import (
            propagate_boundary_input_scales,
            read_boundary_out_scales,
        )

        model = _aligned_chain_model()
        perceptrons = list(model.get_perceptrons())
        hidden = perceptrons[1]
        hidden.set_activation_scale(torch.linspace(0.5, 2.0, 16))

        propagate_boundary_input_scales(model, input_data_scale=1.0)
        head = perceptrons[2]
        expected = float(
            hidden.activation_scale.detach().to(torch.float64).mean()
        )
        assert float(head.input_activation_scale) == pytest.approx(expected)

        pure = read_boundary_out_scales(model, input_data_scale=1.0)
        mapper = next(
            n for n in model.get_mapper_repr()._exec_order
            if getattr(n, "perceptron", None) is hidden
        )
        assert pure[mapper] == pytest.approx(expected)


class TestGaugeVectorTolerance:
    """Warn-only gauges read a scalar theta from a promoted model (mean-collapse),
    never crash."""

    def test_lif_temporal_gauge_on_promoted_model(self):
        from mimarsinan.spiking.gain_correction import (
            per_perceptron_cascade_depth,
        )
        from mimarsinan.tuning.orchestration.install_resolution import (
            ChannelStatsAccumulator,
        )
        from mimarsinan.tuning.orchestration.install_resolution.gauges import (
            lif_temporal_gauge,
        )

        model = _aligned_chain_model()
        perceptrons = list(model.get_perceptrons())
        perceptrons[1].set_activation_scale(torch.linspace(0.5, 2.0, 16))

        accs = []
        for _ in perceptrons:
            acc = ChannelStatsAccumulator()
            acc.output_transform(torch.rand(4, 16))
            accs.append(acc)
        gauge = lif_temporal_gauge(
            list(zip(perceptrons, accs)), per_perceptron_cascade_depth(
                model.get_mapper_repr()
            ), window=4,
        )
        assert gauge.total_delay > 0.0
