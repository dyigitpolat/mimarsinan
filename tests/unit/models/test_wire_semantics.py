"""Wire-op kernel pairs: torch and numpy twins must agree bit-for-bit."""

import numpy as np
import pytest
import torch

from mimarsinan.models.spiking.wire_semantics import (
    WireSemantics,
    floor_staircase,
    floor_staircase_np,
    ttfs_grid_quantize,
    ttfs_grid_quantize_np,
    ttfs_quantized_staircase,
    ttfs_quantized_staircase_np,
    ttfs_spike_time,
    ttfs_spike_time_np,
)

STEPS = (1, 2, 3, 4, 8, 16)
THRESHOLDS = (1.0, 2.5, 8.65625)


def _staircase_sweep(S: int, theta: float) -> np.ndarray:
    """Grid points k·θ/S, ±1 ULP neighbours, and a dense out-of-range sweep."""
    grid = np.arange(S + 1, dtype=np.float64) / S * theta
    above = np.nextafter(grid, np.inf)
    below = np.nextafter(grid, -np.inf)
    dense = np.linspace(-0.5, 1.5, 257) * theta
    return np.concatenate([grid, above, below, dense])


class TestQuantizedStaircaseTwins:
    @pytest.mark.parametrize("S", STEPS)
    @pytest.mark.parametrize("theta", THRESHOLDS)
    def test_float64_bitwise_equal(self, S, theta):
        v = _staircase_sweep(S, theta)
        got = ttfs_quantized_staircase(
            torch.from_numpy(v), torch.tensor(theta, dtype=torch.float64), S
        ).numpy()
        expected = ttfs_quantized_staircase_np(v, np.float64(theta), S)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("S", (4, 16))
    def test_float32_grid_points_exact(self, S):
        # Dyadic grid values are exact in float32; the twins must agree there.
        # (Contract reference runs float64; float32 NF may tie-flip within one
        # ULP of the grid — the documented compute-dtype caveat.)
        v32 = (torch.arange(S + 1, dtype=torch.float32) / S)
        got = ttfs_quantized_staircase(v32, torch.tensor(1.0), S).double().numpy()
        expected = ttfs_quantized_staircase_np(v32.numpy().astype(np.float64), 1.0, S)
        np.testing.assert_array_equal(got, expected)

    def test_matches_legacy_ttfs_kernels_import_path(self):
        from mimarsinan.models.spiking.ttfs_kernels import (
            ttfs_quantized_activation,
            ttfs_quantized_activation_np,
        )

        v = _staircase_sweep(4, 1.0)
        np.testing.assert_array_equal(
            ttfs_quantized_activation_np(v, 1.0, 4),
            ttfs_quantized_staircase_np(v, 1.0, 4),
        )
        np.testing.assert_array_equal(
            ttfs_quantized_activation(
                torch.from_numpy(v), torch.tensor(1.0, dtype=torch.float64), 4
            ).numpy(),
            ttfs_quantized_staircase(
                torch.from_numpy(v), torch.tensor(1.0, dtype=torch.float64), 4
            ).numpy(),
        )


class TestFloorCeilEquivalence:
    """Trained-model no-regression lock: on the clamped unit domain with
    integer S, the legacy floor staircase equals the deployment ceil kernel."""

    @pytest.mark.parametrize("S", STEPS)
    def test_floor_equals_ceil_kernel_on_unit_domain(self, S):
        grid = np.arange(S + 1, dtype=np.float64) / S
        dense = np.linspace(0.0, 1.0, 1025)
        r = np.concatenate([grid, dense])
        floor_form = floor_staircase_np(r, S)
        ceil_form = ttfs_quantized_staircase_np(r, 1.0, S)
        np.testing.assert_array_equal(floor_form, ceil_form)

    def test_floor_staircase_twins_equal(self):
        x = np.linspace(-1.0, 2.0, 513)
        for levels in (4.0, 16.0, 5.5):  # QuantizeDecorator uses non-integer levels
            np.testing.assert_array_equal(
                floor_staircase(torch.from_numpy(x), levels).numpy(),
                floor_staircase_np(x, levels),
            )


class TestSpikeTimeAndGridQuantizeTwins:
    @pytest.mark.parametrize("S", STEPS)
    def test_spike_time_twins(self, S):
        ties = (np.arange(S, dtype=np.float64) + 0.5) / S
        x = np.concatenate([
            ties, np.linspace(-0.2, 1.2, 257),
            np.arange(S + 1, dtype=np.float64) / S,
        ])
        got = ttfs_spike_time(torch.from_numpy(x), S).numpy()
        expected = ttfs_spike_time_np(x, S)
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("S", STEPS)
    def test_grid_quantize_twins(self, S):
        ties = (np.arange(S, dtype=np.float64) + 0.5) / S
        x = np.concatenate([ties, np.linspace(0.0, 1.0, 257)])
        got = ttfs_grid_quantize(torch.from_numpy(x), S).numpy()
        expected = ttfs_grid_quantize_np(x, S)
        np.testing.assert_array_equal(got, expected)

    def test_numpy_twins_are_the_encoding_ssot(self):
        from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
            ttfs_input_grid_quantize,
            ttfs_spike_time as encoding_spike_time,
        )

        x = np.linspace(0.0, 1.0, 257)
        np.testing.assert_array_equal(
            encoding_spike_time(x, 4), ttfs_spike_time_np(x, 4)
        )
        np.testing.assert_array_equal(
            ttfs_input_grid_quantize(x, 4), ttfs_grid_quantize_np(x, 4)
        )


class TestWireSemantics:
    def test_bundles_ops_for_inclusive_compare(self):
        wire = WireSemantics(simulation_steps=4)
        v = np.linspace(0.0, 1.0, 33)
        np.testing.assert_array_equal(
            wire.quantized_staircase_np(v, 1.0),
            ttfs_quantized_staircase_np(v, 1.0, 4),
        )
        np.testing.assert_array_equal(
            wire.grid_quantize_np(v), ttfs_grid_quantize_np(v, 4)
        )
        np.testing.assert_array_equal(
            wire.spike_time_np(v), ttfs_spike_time_np(v, 4)
        )

    def test_strict_compare_shifts_exact_grid_ties_only(self):
        """``<`` fires one cycle later when V sits exactly on a grid boundary
        (nevresim ``StrictCompare``); off-grid values are unaffected."""
        S = 4
        inclusive = WireSemantics(simulation_steps=S, compare_mode="<=")
        strict = WireSemantics(simulation_steps=S, compare_mode="<")
        grid = np.arange(1, S + 1, dtype=np.float64) / S  # j/S for j=1..S
        np.testing.assert_array_equal(
            inclusive.quantized_staircase_np(grid, 1.0), grid
        )
        np.testing.assert_array_equal(
            strict.quantized_staircase_np(grid, 1.0), grid - 1.0 / S
        )
        off_grid = np.linspace(0.0, 1.0, 257)[1::2]  # never on the S=4 grid
        np.testing.assert_array_equal(
            strict.quantized_staircase_np(off_grid, 1.0),
            inclusive.quantized_staircase_np(off_grid, 1.0),
        )

    def test_strict_compare_torch_twin_matches(self):
        S = 4
        strict = WireSemantics(simulation_steps=S, compare_mode="<")
        v = _staircase_sweep(S, 1.0)
        np.testing.assert_array_equal(
            strict.quantized_staircase(
                torch.from_numpy(v), torch.tensor(1.0, dtype=torch.float64)
            ).numpy(),
            strict.quantized_staircase_np(v, 1.0),
        )

    def test_contract_wire_factory(self):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        contract = SpikingDeploymentContract.from_pipeline_config({
            "spiking_mode": "ttfs_cycle_based",
            "thresholding_mode": "<=",
            "simulation_steps": 8,
            "ttfs_cycle_schedule": "synchronized",
        })
        wire = contract.wire()
        assert isinstance(wire, WireSemantics)
        assert wire.simulation_steps == 8
        assert wire.compare_mode == "<="


class TestSTEWrappers:
    def test_staircase_function_keeps_floor_form_and_ste_grad(self):
        from mimarsinan.models.nn.activations.autograd import StaircaseFunction

        x = torch.linspace(0.0, 1.0, 33, dtype=torch.float64, requires_grad=True)
        out = StaircaseFunction.apply(x, torch.tensor(5.5, dtype=torch.float64))
        np.testing.assert_array_equal(
            out.detach().numpy(), floor_staircase_np(x.detach().numpy(), 5.5)
        )
        out.sum().backward()
        np.testing.assert_array_equal(x.grad.numpy(), np.ones(33))

    def test_ttfs_staircase_function_is_deployment_kernel_with_ste_grad(self):
        from mimarsinan.models.nn.activations.autograd import TTFSStaircaseFunction

        r = torch.linspace(0.0, 1.0, 33, dtype=torch.float64, requires_grad=True)
        out = TTFSStaircaseFunction.apply(r, 4)
        np.testing.assert_array_equal(
            out.detach().numpy(),
            ttfs_quantized_staircase_np(r.detach().numpy(), 1.0, 4),
        )
        out.sum().backward()
        np.testing.assert_array_equal(r.grad.numpy(), np.ones(33))


class TestNFActivationsUseDeploymentKernel:
    """The torch NF activations must equal the float64 deployment kernel at
    every value, including ±1 ULP around the S-grid (the 12%-tie zone)."""

    def test_ttfs_activation_analytical_forward(self):
        from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

        S, theta = 4, 2.5
        activation = TTFSActivation(T=S, activation_scale=theta)
        v = torch.from_numpy(_staircase_sweep(S, theta))
        with torch.no_grad():
            got = activation(v).numpy()
        r = np.clip(np.maximum(v.numpy(), 0.0) / theta, 0.0, 1.0)
        expected = ttfs_quantized_staircase_np(r, 1.0, S) * theta
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def test_ttfs_cycle_activation_forward(self):
        from mimarsinan.models.nn.activations.ttfs_cycle import TTFSCycleActivation

        S, theta = 4, 2.5
        activation = TTFSCycleActivation(T=S, activation_scale=theta)
        v = torch.from_numpy(_staircase_sweep(S, theta))
        with torch.no_grad():
            got = activation(v).numpy()
        r = np.clip(np.maximum(v.numpy(), 0.0) / theta, 0.0, 1.0)
        expected = ttfs_quantized_staircase_np(r, 1.0, S) * theta
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)


class TestLatchedEncodeUsesSpikeTimeTwin:
    @pytest.mark.parametrize("S", (4, 8))
    def test_latched_spikes_match_numpy_train(self, S):
        from mimarsinan.chip_simulation.recording.spike_modes import (
            to_ttfs_latched_spikes,
        )
        from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
            ttfs_latched_spike_train,
        )

        rates = np.concatenate([
            (np.arange(S, dtype=np.float64) + 0.5) / S,
            np.linspace(0.0, 1.0, 65),
        ]).reshape(1, -1)
        train = ttfs_latched_spike_train(rates, S)  # (1, D, S)
        for cycle in range(S):
            got = to_ttfs_latched_spikes(
                torch.from_numpy(rates), cycle, S
            ).numpy()
            np.testing.assert_array_equal(got, train[:, :, cycle])


class TestComparatorHalfStepKernel:
    """[E3/G6] comparator-side half-step: the SHIFTED COMPARATOR LADDER
    ``ceil(S*(1 - v/theta) - 0.5)`` — every per-cycle compare level
    ``theta*(S-k)/S`` drops by ``theta/(2S)`` — the exact zero-bit-cost twin of
    the ``+theta/(2S)`` bias fold, carried in the threshold so the two-scale WQ
    bias lattice can never erode it."""

    @pytest.mark.parametrize("S", STEPS)
    @pytest.mark.parametrize("theta", THRESHOLDS)
    def test_twins_bitwise_equal(self, S, theta):
        v = _staircase_sweep(S, theta)
        got = ttfs_quantized_staircase(
            torch.from_numpy(v), torch.tensor(theta, dtype=torch.float64), S,
            comparator_half_step=True,
        ).numpy()
        expected = ttfs_quantized_staircase_np(
            v, np.float64(theta), S, comparator_half_step=True,
        )
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize("S", (2, 4, 8, 16))
    def test_equals_bias_fold_on_dyadic_lattice(self, S):
        # v and v + 1/(2S) are both exact dyadic float64 values, so the
        # identity Q_S(v + theta/(2S)) == comparator-shifted Q_S(v) is exact
        # INCLUDING at grid ties.
        v = np.arange(-2 * S, 6 * S + 1, dtype=np.float64) / (4 * S)
        shifted = ttfs_quantized_staircase_np(
            v, 1.0, S, comparator_half_step=True,
        )
        bias_fold = ttfs_quantized_staircase_np(v + 1.0 / (2 * S), 1.0, S)
        np.testing.assert_array_equal(shifted, bias_fold)

    @pytest.mark.parametrize("theta", THRESHOLDS)
    def test_equals_bias_fold_off_ties(self, theta):
        S = 8
        rng = np.random.default_rng(0)
        v = rng.uniform(-0.5, 1.5, size=4096) * theta
        ladder = S * (1.0 - v / theta) - 0.5
        off_tie = np.abs(ladder - np.round(ladder)) > 1e-9
        v = v[off_tie]
        shifted = ttfs_quantized_staircase_np(
            v, theta, S, comparator_half_step=True,
        )
        bias_fold = ttfs_quantized_staircase_np(v + theta / (2 * S), theta, S)
        np.testing.assert_array_equal(shifted, bias_fold)

    @pytest.mark.parametrize("S", STEPS)
    @pytest.mark.parametrize("theta", THRESHOLDS)
    def test_flag_off_is_bitwise_identical(self, S, theta):
        v = _staircase_sweep(S, theta)
        np.testing.assert_array_equal(
            ttfs_quantized_staircase_np(v, theta, S, comparator_half_step=False),
            ttfs_quantized_staircase_np(v, theta, S),
        )
        np.testing.assert_array_equal(
            ttfs_quantized_staircase(
                torch.from_numpy(v), torch.tensor(theta, dtype=torch.float64), S,
                comparator_half_step=False,
            ).numpy(),
            ttfs_quantized_staircase(
                torch.from_numpy(v), torch.tensor(theta, dtype=torch.float64), S,
            ).numpy(),
        )

    def test_dead_zone_halved_at_the_boundary(self):
        # Identity 2: the half-step shrinks the dead zone to [0, theta/(2S));
        # v == theta/(2S) fires the first level exactly. (One-ULP probes are
        # excluded: ``1 - v`` legitimately rounds them onto the boundary.)
        S, theta = 4, 1.0
        half = theta / (2 * S)
        v = np.array([0.0, half / 2, half, 1.5 * half])
        out = ttfs_quantized_staircase_np(v, theta, S, comparator_half_step=True)
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0 / S, 1.0 / S])

    def test_is_a_ladder_shift_not_a_theta_rescale(self):
        # Counterexample pinning the mechanism: at S=4, theta=1, v=0.15 the
        # bias fold fires level 1 (0.25); a multiplicative rescale
        # theta' = theta*(1 - 1/(2S)) leaves the neuron silent (0.0). The
        # comparator form must match the bias fold, never the rescale.
        S, theta, v = 4, 1.0, np.array([0.15])
        shifted = ttfs_quantized_staircase_np(v, theta, S, comparator_half_step=True)
        rescaled = ttfs_quantized_staircase_np(v, theta * (1.0 - 1.0 / (2 * S)), S)
        np.testing.assert_array_equal(shifted, [0.25])
        np.testing.assert_array_equal(rescaled, [0.0])

    def test_strict_compare_twins_and_bias_fold_identity(self):
        S = 4
        strict = WireSemantics(
            simulation_steps=S, compare_mode="<", comparator_half_step=True,
        )
        v = _staircase_sweep(S, 1.0)
        np.testing.assert_array_equal(
            strict.quantized_staircase(
                torch.from_numpy(v), torch.tensor(1.0, dtype=torch.float64)
            ).numpy(),
            strict.quantized_staircase_np(v, 1.0),
        )
        # Dyadic identity holds for the strict kernel too.
        strict_plain = WireSemantics(simulation_steps=S, compare_mode="<")
        dyadic = np.arange(-2 * S, 6 * S + 1, dtype=np.float64) / (4 * S)
        np.testing.assert_array_equal(
            strict.quantized_staircase_np(dyadic, 1.0),
            strict_plain.quantized_staircase_np(dyadic + 1.0 / (2 * S), 1.0),
        )

    def test_wire_semantics_plumb_and_default_off(self):
        S = 8
        v = _staircase_sweep(S, 2.5)
        wire_on = WireSemantics(simulation_steps=S, comparator_half_step=True)
        np.testing.assert_array_equal(
            wire_on.quantized_staircase_np(v, 2.5),
            ttfs_quantized_staircase_np(v, 2.5, S, comparator_half_step=True),
        )
        wire_off = WireSemantics(simulation_steps=S)
        assert wire_off.comparator_half_step is False
        np.testing.assert_array_equal(
            wire_off.quantized_staircase_np(v, 2.5),
            ttfs_quantized_staircase_np(v, 2.5, S),
        )

    def test_ttfs_kernels_alias_passes_the_flag(self):
        from mimarsinan.models.spiking.ttfs_kernels import (
            ttfs_quantized_activation,
            ttfs_quantized_activation_np,
        )

        S = 4
        v = _staircase_sweep(S, 1.0)
        np.testing.assert_array_equal(
            ttfs_quantized_activation_np(v, 1.0, S, comparator_half_step=True),
            ttfs_quantized_staircase_np(v, 1.0, S, comparator_half_step=True),
        )
        np.testing.assert_array_equal(
            ttfs_quantized_activation(
                torch.from_numpy(v), torch.tensor(1.0, dtype=torch.float64), S,
                comparator_half_step=True,
            ).numpy(),
            ttfs_quantized_staircase_np(v, 1.0, S, comparator_half_step=True),
        )

    def test_contract_wire_carries_the_flag(self):
        from mimarsinan.chip_simulation.deployment_contract import (
            SpikingDeploymentContract,
        )

        base = {
            "spiking_mode": "ttfs_cycle_based",
            "thresholding_mode": "<=",
            "simulation_steps": 8,
            "ttfs_cycle_schedule": "synchronized",
        }
        off = SpikingDeploymentContract.from_pipeline_config(base)
        assert off.comparator_half_step is False
        assert off.wire().comparator_half_step is False
        on = SpikingDeploymentContract.from_pipeline_config(
            {**base, "comparator_half_step": True}
        )
        assert on.comparator_half_step is True
        assert on.wire().comparator_half_step is True


class TestComparatorHalfStepNFSide:
    """The NF torch twins (autograd fn + ceil decorator) run the same shifted
    ladder, so both parity sides move together from the one contract flag."""

    def test_comparator_staircase_function_matches_kernel_with_ste_grad(self):
        from mimarsinan.models.nn.activations.autograd import (
            TTFSComparatorHalfStepStaircaseFunction,
        )

        r = torch.linspace(0.0, 1.0, 33, dtype=torch.float64, requires_grad=True)
        out = TTFSComparatorHalfStepStaircaseFunction.apply(r, 4)
        np.testing.assert_array_equal(
            out.detach().numpy(),
            ttfs_quantized_staircase_np(
                r.detach().numpy(), 1.0, 4, comparator_half_step=True,
            ),
        )
        out.sum().backward()
        np.testing.assert_array_equal(r.grad.numpy(), np.ones(33))

    def test_ceil_decorator_flag_on_runs_the_shifted_ladder(self):
        from mimarsinan.models.nn.decorators.clamp_quantize import (
            TTFSCeilStaircaseDecorator,
        )

        S, theta = 4, 2.5
        dec = TTFSCeilStaircaseDecorator(S, theta, comparator_half_step=True)
        v = torch.from_numpy(_staircase_sweep(S, theta))
        with torch.no_grad():
            got = dec.output_transform(v).numpy()
        expected = ttfs_quantized_staircase_np(
            v.numpy() / theta, 1.0, S, comparator_half_step=True,
        ) * theta
        np.testing.assert_allclose(got, expected, rtol=0, atol=0)

    def test_ceil_decorator_default_off_is_bitwise_identical(self):
        from mimarsinan.models.nn.decorators.clamp_quantize import (
            TTFSCeilStaircaseDecorator,
        )

        S, theta = 4, 2.5
        v = torch.from_numpy(_staircase_sweep(S, theta))
        with torch.no_grad():
            got = TTFSCeilStaircaseDecorator(S, theta).output_transform(v)
            explicit_off = TTFSCeilStaircaseDecorator(
                S, theta, comparator_half_step=False,
            ).output_transform(v)
        np.testing.assert_array_equal(got.numpy(), explicit_off.numpy())


class TestComparatorHalfStepSegmentThreading:
    """SCM consumption: run_ttfs_segment threads the flag into the per-core
    staircase, so the numpy deployment side shifts with the NF twin."""

    @staticmethod
    def _one_core_segment(n_in=3, n_out=2):
        from mimarsinan.chip_simulation.ttfs.segment_arrays import SegmentTtfsArrays
        from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan

        w = np.array([[0.5, 0.25, -0.125], [1.0, -0.5, 0.75]], dtype=np.float64)
        return SegmentTtfsArrays(
            core_params=[w],
            thresholds=[np.array(1.0)],
            hw_biases=[np.array([0.05, -0.1])],
            latencies=[0],
            axon_spans=[[SpikeSourceSpan(
                kind="input", src_core=-2, src_start=0, length=n_in, dst_start=0,
            )]],
            output_spans=[SpikeSourceSpan(
                kind="core", src_core=0, src_start=0, length=n_out, dst_start=0,
            )],
            n_output=n_out,
            n_axons_per_core=[n_in],
            n_neurons_per_core=[n_out],
        )

    def test_flag_threads_into_the_core_staircase(self):
        from mimarsinan.chip_simulation.ttfs.ttfs_segment import run_ttfs_segment

        seg = self._one_core_segment()
        x = np.array([[0.25, 0.5, 1.0], [1.0, 0.0, 0.5]], dtype=np.float64)
        v = x @ seg.core_params[0].T + seg.hw_biases[0]
        out_on, _, _ = run_ttfs_segment(
            seg, x, simulation_length=4, spiking_mode="ttfs_cycle_based",
            comparator_half_step=True,
        )
        np.testing.assert_array_equal(
            out_on,
            ttfs_quantized_staircase_np(v, 1.0, 4, comparator_half_step=True),
        )
        out_off, _, _ = run_ttfs_segment(
            seg, x, simulation_length=4, spiking_mode="ttfs_cycle_based",
        )
        np.testing.assert_array_equal(
            out_off, ttfs_quantized_staircase_np(v, 1.0, 4),
        )
