"""Golden-output regression for the lifted ``_uniform_rate_encode`` helper.

The same function is consumed by ``LavaLoihiRunner`` and (next) by the
SANA-FE runner.  Any drift in its output would silently change the
HCM-vs-Lava parity surface — so we pin a handful of small reference
inputs here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.chip_simulation.recording._spike_encoding import (
    deterministic_rate_encode,
    encode_segment_input,
    flatten_spike_train_sample,
    front_loaded_rate_encode,
    spike_train_rate_encode,
    stochastic_rate_encode,
    uniform_rate_encode,
)
from mimarsinan.chip_simulation.recording import spike_modes


def test_uniform_rate_encode_returns_shape_N_D_T():
    out = uniform_rate_encode(np.zeros((3, 5), dtype=np.float32), T=7)
    assert out.shape == (3, 5, 7)
    assert out.dtype == np.float32


def test_uniform_rate_encode_zero_rates_produce_no_spikes():
    out = uniform_rate_encode(np.zeros((2, 4), dtype=np.float32), T=10)
    assert out.sum() == 0.0


def test_uniform_rate_encode_full_rates_produce_full_train():
    """rate == 1.0 ⇒ a spike at every cycle."""
    out = uniform_rate_encode(np.ones((1, 3), dtype=np.float32), T=8)
    assert np.array_equal(out, np.ones((1, 3, 8), dtype=np.float32))


def test_uniform_rate_encode_half_rate_produces_T_over_2_spikes():
    """rate == 0.5, T == 8 ⇒ exactly 4 spikes per (sample, dim)."""
    out = uniform_rate_encode(np.full((1, 3), 0.5, dtype=np.float32), T=8)
    per_dim = out.sum(axis=2)
    np.testing.assert_array_equal(per_dim, np.full((1, 3), 4.0))


def test_uniform_rate_encode_clips_negative_rates_to_zero():
    out = uniform_rate_encode(np.full((1, 2), -0.3, dtype=np.float32), T=5)
    assert out.sum() == 0.0


def test_uniform_rate_encode_clips_super_one_rates_to_one():
    """Anything > 1.0 saturates at "fire every cycle"."""
    out = uniform_rate_encode(np.full((1, 2), 1.7, dtype=np.float32), T=4)
    assert np.array_equal(out, np.ones((1, 2, 4), dtype=np.float32))


def test_uniform_rate_encode_byte_identical_to_lava_runner_inline_copy():
    """The Lava runner imports the same helper — verify it still re-exports.

    This guards the helper-lift: ``LavaLoihiRunner`` must keep calling the
    canonical implementation, not a stale local one.
    """
    from mimarsinan.chip_simulation.lava_loihi import _uniform_rate_encode

    rates = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(2, 6)
    ours = uniform_rate_encode(rates, T=16)
    theirs = _uniform_rate_encode(rates, T=16)
    np.testing.assert_array_equal(ours, theirs)


def _torch_stacked_encode(rates: np.ndarray, T: int, spike_mode: str) -> np.ndarray:
    tensor = torch.tensor(rates, dtype=torch.float32)
    N, D = rates.shape
    out = np.zeros((N, D, T), dtype=np.float32)
    for cycle in range(T):
        spikes = spike_modes.to_spikes(
            tensor,
            cycle,
            simulation_length=T,
            spike_mode=spike_mode,
        )
        out[:, :, cycle] = spikes.numpy()
    return out


@pytest.mark.parametrize("spike_mode", ["Uniform", "FrontLoaded", "Deterministic"])
def test_batch_encode_matches_torch_stacked(spike_mode):
    rates = np.array([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    T = 8
    batch = encode_segment_input(rates, T, spike_mode)
    stacked = _torch_stacked_encode(rates, T, spike_mode)
    np.testing.assert_array_equal(batch, stacked)


def test_stochastic_encode_seeded_reproducible():
    rates = np.full((1, 4), 0.3, dtype=np.float32)
    a = encode_segment_input(rates, T=10, spike_mode="Stochastic", seed=42)
    b = encode_segment_input(rates, T=10, spike_mode="Stochastic", seed=42)
    np.testing.assert_array_equal(a, b)


def test_encode_segment_input_rejects_ttfs():
    with pytest.raises(ValueError, match="TTFS"):
        encode_segment_input(np.zeros((1, 1), dtype=np.float32), 4, "TTFS")


def test_deterministic_fires_when_rate_above_half():
    out = deterministic_rate_encode(np.array([[0.6, 0.4]], dtype=np.float32), T=5)
    assert out[0, 0, :].sum() == 5.0
    assert out[0, 1, :].sum() == 0.0


def test_front_loaded_half_rate():
    out = front_loaded_rate_encode(np.full((1, 1), 0.5, dtype=np.float32), T=8)
    assert out.sum() == 4.0


def test_spike_train_materialization_matches_uniform():
    rates = np.linspace(0.0, 1.0, 6, dtype=np.float32).reshape(1, 6)
    T = 8
    uniform = uniform_rate_encode(rates, T)
    spike_train = spike_train_rate_encode(rates, T)
    np.testing.assert_array_equal(spike_train, uniform)


def test_spike_train_encode_segment_input_dispatcher():
    rates = np.array([[1.0, 0.5]], dtype=np.float32)
    out = encode_segment_input(rates, T=4, spike_mode="SpikeTrain")
    assert out.shape == (1, 2, 4)


def test_flatten_spike_train_sample_cycle_major_order():
    encoded = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)  # D=2, T=2
    flat = flatten_spike_train_sample(encoded[0])
    np.testing.assert_array_equal(flat, np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64))


def test_behavior_config_spike_train_nevresim_flag():
    from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig

    cfg = NeuralBehaviorConfig(
        spiking_mode="lif",
        firing_mode="Default",
        thresholding_mode="<",
        spike_generation_mode="SpikeTrain",
    )
    assert cfg.nevresim_uses_spike_train_input() is True
    out = cfg.encode_segment_input(np.array([[1.0]], dtype=np.float32), 4)
    assert out.shape == (1, 1, 4)



class TestCombFloat64Canonical:
    """[nevresim parity] comb placement matches the chip's double arithmetic
    at exact-division knife-edges (n=12, T=32, spacing 8/3: cycle 8 fires,
    not 9 — the t0_04 ±1-window class)."""

    def test_knife_edge_placement_matches_double(self):
        import torch

        from mimarsinan.chip_simulation.recording import spike_modes

        T = 32
        v = torch.tensor([0.375], dtype=torch.float32)
        fired = [
            c for c in range(T)
            if float(spike_modes.to_uniform_spikes(v, c, T)) == 1.0
        ]
        assert fired == [0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 27, 30]

    def test_dtype_of_result_follows_input(self):
        import torch

        from mimarsinan.chip_simulation.recording import spike_modes

        for dt in (torch.float32, torch.float64):
            out = spike_modes.to_uniform_spikes(
                torch.tensor([0.5], dtype=dt), 0, 8,
            )
            assert out.dtype == dt


class TestTtfsIntegerLatticeSnap:
    """[t0_15 catch] on the integer chip, TTFS membranes live on the 1/S
    grid; float dust must never decide a staircase tie (V=16.99999943 vs
    theta=17 read one ladder level low while the plugin's exact 17 fired)."""

    def _seg(self, weights):
        import numpy as np

        from mimarsinan.chip_simulation.ttfs.segment_arrays import (
            SegmentTtfsArrays,
        )
        from mimarsinan.mapping.support.spike_source_spans import (
            SpikeSourceSpan,
        )

        n_in = weights.shape[1]
        spans = [SpikeSourceSpan(kind="input", src_core=-1, src_start=0,
                                 dst_start=0, length=n_in)]
        out_spans = [SpikeSourceSpan(kind="core", src_core=0, src_start=0,
                                     dst_start=0, length=1)]
        return SegmentTtfsArrays(
            core_params=[weights],
            thresholds=[np.array([17.0])],
            hw_biases=[None],
            latencies=[0],
            axon_spans=[spans],
            output_spans=out_spans,
            n_output=1,
            n_axons_per_core=[n_in],
            n_neurons_per_core=[1],
        )

    def test_integer_chip_tie_fires_full_level(self):
        import numpy as np

        from mimarsinan.chip_simulation.ttfs.ttfs_segment import (
            run_ttfs_segment,
        )

        w = np.full((1, 8), 17.0 / 8.0)
        # Integer-lattice check needs integral weights: 17/8 is fractional,
        # so craft integral weights summing exactly to theta at rate 1/8 * 8.
        w = np.ones((1, 17))
        seg = self._seg(w)
        # noisy inputs: true value 17 * (1/17 each... use rate 1.0 slightly off
        # on-grid inputs (rate 1.0 = 8/8) carrying float dust
        a = np.full((1, 17), 1.0 - 3.5e-8)
        out, _, membrane = run_ttfs_segment(
            seg, a, simulation_length=8, spiking_mode="ttfs_quantized",
        )
        assert float(membrane[0][0, 0]) == 17.0
        assert float(out[0, 0]) == 1.0

    def test_off_grid_inputs_never_snap(self):
        import numpy as np

        from mimarsinan.chip_simulation.ttfs.ttfs_segment import (
            run_ttfs_segment,
        )

        seg = self._seg(np.ones((1, 17)))
        a = np.full((1, 17), 0.437)
        _, _, membrane = run_ttfs_segment(
            seg, a, simulation_length=8, spiking_mode="ttfs_quantized",
        )
        assert abs(float(membrane[0][0, 0]) - 17 * 0.437) < 1e-9

    def test_float_chip_is_untouched(self):
        import numpy as np

        from mimarsinan.chip_simulation.ttfs.ttfs_segment import (
            run_ttfs_segment,
        )

        w = np.full((1, 17), 1.0001)
        seg = self._seg(w)
        assert seg.integer_lattice() is False
        a = np.full((1, 17), 1.0 - 3.5e-8)
        _, _, membrane = run_ttfs_segment(
            seg, a, simulation_length=8, spiking_mode="ttfs_quantized",
        )
        assert float(membrane[0][0, 0]) != 17.0
