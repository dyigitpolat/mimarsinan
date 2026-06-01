"""Synchronized genuine single-spike TTFS reference == analytical kernel composition.

Validates the spec the nevresim/SANA-FE genuine somas must realize: latency groups
run sequentially (sim_time = S × groups), each neuron fires one spike at
k_fire = ⌈S(1−V/θ)⌉ within its window, inter-group signals are spike timings, and the
decoded result equals ttfs_quantized_activation composed in topological order.
"""

import numpy as np
import torch

from mimarsinan.chip_simulation.ttfs.ttfs_cycle_genuine import (
    NO_SPIKE,
    decode_spike_time_to_activation,
    encode_activation_to_spike_time,
    genuine_total_cycles,
    latency_groups,
    run_ttfs_cycle_genuine_layers,
    synchronized_window,
    ttfs_cycle_fire_step,
)
from mimarsinan.models.spiking.ttfs_kernels import ttfs_quantized_activation


def _analytical(V, theta, S):
    return ttfs_quantized_activation(
        torch.tensor(V, dtype=torch.float64), torch.tensor(theta, dtype=torch.float64), S
    ).numpy()


class TestSpikeTimeCodec:
    def test_encode_decode_roundtrip_on_grid(self):
        S = 16
        a = np.array([0.0, 1/16, 0.5, 15/16, 1.0])
        k = encode_activation_to_spike_time(a, S)
        np.testing.assert_allclose(decode_spike_time_to_activation(k, S), a)

    def test_zero_is_no_spike(self):
        assert encode_activation_to_spike_time(np.array([0.0]), 8)[0] == NO_SPIKE


class TestFireStepMatchesAnalytical:
    def test_single_layer_fire_step_equals_kernel(self):
        S = 16
        theta = 1.3
        V = np.linspace(-0.5, 1.7, 101)
        k = ttfs_cycle_fire_step(V, theta, S)
        out = decode_spike_time_to_activation(k, S)
        np.testing.assert_allclose(out, _analytical(V, theta, S), rtol=0, atol=1e-12)


class TestSynchronizedMultiLayerEqualsAnalytical:
    def test_two_layer_genuine_equals_analytical_composition(self):
        S = 16
        rng = np.random.default_rng(0)
        W1 = rng.uniform(-1, 1, size=(3, 4))
        b1 = rng.uniform(-0.2, 0.2, size=3)
        th1 = 1.1
        W2 = rng.uniform(-1, 1, size=(2, 3))
        b2 = rng.uniform(-0.2, 0.2, size=2)
        th2 = 0.9
        x = rng.uniform(0, 1, size=(5, 4))

        out_genuine, spikes = run_ttfs_cycle_genuine_layers(
            [(W1, b1, th1), (W2, b2, th2)], x, S
        )

        # Analytical composition (topological order), single-spike kernel per layer.
        a1 = _analytical(x @ W1.T + b1, th1, S)
        a2 = _analytical(a1 @ W2.T + b2, th2, S)
        np.testing.assert_allclose(out_genuine, a2, rtol=0, atol=1e-12)

        # Genuine wire signals are single spike timings (or NO_SPIKE), not real values.
        for k in spikes:
            assert k.dtype == np.int64
            assert np.all((k == NO_SPIKE) | ((k >= 0) & (k < S)))


class TestTimeline:
    def test_total_cycles_is_S_times_groups(self):
        assert genuine_total_cycles(num_latency_groups=4, simulation_steps=16) == 64
        # Synchronized (×) not pipelined (+).
        assert genuine_total_cycles(4, 16) != 16 + 4


class TestLatencyGrouping:
    def test_groups_by_latency_ascending(self):
        # latencies: two cores at depth 0, one at depth 1, one at depth 2.
        num, per_core = latency_groups([0, 0, 1, 2])
        assert num == 3
        assert per_core == [0, 0, 1, 2]

    def test_non_contiguous_latencies_ranked(self):
        num, per_core = latency_groups([0, 5, 5, 2])
        assert num == 3
        assert per_core == [0, 2, 2, 1]

    def test_none_latency_goes_to_last_group(self):
        num, per_core = latency_groups([0, 1, None])
        assert num == 2
        assert per_core == [0, 1, 1]

    def test_synchronized_windows_non_overlapping(self):
        S = 16
        w0 = synchronized_window(0, S)
        w1 = synchronized_window(1, S)
        assert w0 == (0, 16)
        assert w1 == (16, 16)
        # group g occupies [g*S, (g+1)*S) — no overlap.
        assert w0[0] + w0[1] == w1[0]
