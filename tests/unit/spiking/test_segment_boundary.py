"""SSOT segment-boundary contract: the decode side shared by every backend.

The encode paths (``encode_segment_input`` / ``encode_compute_boundary``) are
exercised end-to-end by ``test_segment_boundary_encode.py``; here we pin the
decode contract and ``BoundaryConfig`` defaults.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.spiking import segment_boundary as sb


def test_decode_segment_output_numpy_is_counts_over_T():
    counts = np.array([0, 2, 4, 8], dtype=np.float64)
    out = sb.decode_segment_output(counts, 4)
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out, np.array([[0.0, 0.5, 1.0, 2.0]]))


def test_decode_clamps_T_to_at_least_one():
    counts = np.array([3.0])
    np.testing.assert_allclose(sb.decode_segment_output(counts, 0), np.array([[3.0]]))


def test_decode_segment_output_torch_preserves_batch():
    counts = torch.tensor([[2.0, 4.0], [0.0, 8.0]])  # (B=2, N=2)
    out = sb.decode_segment_output_torch(counts, 4)
    assert out.shape == (2, 2)
    torch.testing.assert_close(out, torch.tensor([[0.5, 1.0], [0.0, 2.0]]))


def test_numpy_and_torch_decode_agree():
    counts = np.array([1, 3, 4, 7], dtype=np.float64)
    n = sb.decode_segment_output(counts, 4).reshape(-1)
    t = sb.decode_segment_output_torch(torch.tensor(counts), 4).reshape(-1).numpy()
    np.testing.assert_allclose(n, t)


def test_boundary_config_defaults():
    cfg = sb.BoundaryConfig(simulation_length=4, spiking_mode="lif", cycle_accurate=True)
    # Inherited behavior toggle.
    assert cfg.use_cycle_accurate_trains is True
    # Declared-but-inert Round-2 toggles.
    assert cfg.negative_shift is False
    assert cfg.spike_generation_mode == "Uniform"


class TestRetimedLevelStagesIgnoreCachedTrains:
    """[nevresim parity, t0_04 s32] a retimed hop's input is the COUNT
    re-encode; a cached producer train with the same count but a different
    rhythm must NOT leak through the boundary."""

    def test_retimed_level_reencodes_instead_of_cached_train(self):
        import torch

        from mimarsinan.spiking.segment_boundary import (
            BoundaryConfig,
            encode_segment_input,
        )
        from mimarsinan.spiking.spike_trains import uniform_spike_train

        class _Slice:
            node_id = 7
            offset = 0
            size = 1

        class _Stage:
            input_map = [_Slice()]
            is_retimed_level = True
            name = "lvl_hop1"

        T = 32
        rate = torch.tensor([[4.0 / T]])
        # A count-4 train with a NON-uniform rhythm (shifted comb).
        shifted = torch.zeros(T, 1, 1)
        for c in (0, 9, 17, 25):
            shifted[c, 0, 0] = 1.0
        config = BoundaryConfig(
            simulation_length=T, spiking_mode="lif", cycle_accurate=True,
            spike_mode="Uniform", thresholding_mode="<=", firing_mode="Default",
            compute_dtype=torch.float64, phase_dither=False,
        )
        out = encode_segment_input(
            _Stage(), rate, {7: shifted},
            config=config, hybrid_mapping=None, T=T, batch_size=1,
            device=torch.device("cpu"),
        )
        expected = uniform_spike_train(rate, T).to(torch.float64)
        assert torch.equal(out, expected), "cached rhythm leaked through"
        assert float(out.sum()) == 4.0

    def test_plain_stage_still_prefers_cached_trains(self):
        import torch

        from mimarsinan.spiking.segment_boundary import (
            BoundaryConfig,
            encode_segment_input,
        )

        class _Slice:
            node_id = 7
            offset = 0
            size = 1

        class _Stage:
            input_map = [_Slice()]
            name = "plain"

        T = 8
        rate = torch.tensor([[2.0 / T]])
        cached = torch.zeros(T, 1, 1)
        cached[3, 0, 0] = 1.0
        cached[5, 0, 0] = 1.0
        config = BoundaryConfig(
            simulation_length=T, spiking_mode="lif", cycle_accurate=True,
            spike_mode="Uniform", thresholding_mode="<=", firing_mode="Default",
            compute_dtype=torch.float64, phase_dither=False,
        )
        out = encode_segment_input(
            _Stage(), rate, {7: cached},
            config=config, hybrid_mapping=None, T=T, batch_size=1,
            device=torch.device("cpu"),
        )
        assert torch.equal(out, cached.to(torch.float64))
