"""Per-channel phase-dithered uniform encode: count-exact, decode-invariant."""

import inspect

import torch

from mimarsinan.chip_simulation.recording.spike_modes import (
    to_uniform_spikes,
    uniform_phase_offsets,
)
from mimarsinan.spiking.boundary_config import BoundaryConfig
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_policies import LifSegmentPolicy
from mimarsinan.spiking.spike_trains import uniform_spike_train


def _legacy_to_uniform_spikes(tensor, cycle, simulation_length):
    T = simulation_length
    n = torch.round(tensor * T).to(torch.long)
    mask = (n != 0) & (n != T) & (cycle < T)
    n_safe = torch.clamp(n, min=1)
    spacing = T / n_safe.float()
    result = mask & (torch.floor(cycle / spacing) < n_safe) & (
        torch.floor(cycle % spacing) == 0
    )
    result = result.float()
    result[n == T] = 1.0
    return result


def test_offsets_deterministic_integer_in_range_and_varied():
    T = 32
    a = uniform_phase_offsets(768, T)
    b = uniform_phase_offsets(768, T)
    assert torch.equal(a, b)
    assert a.shape == (768,)
    assert torch.equal(a, a.floor())
    assert float(a.min()) >= 0.0 and float(a.max()) < T
    assert len(torch.unique(a)) > T // 2


def test_default_path_matches_legacy_rule():
    torch.manual_seed(0)
    r = torch.rand(4, 33)
    T = 16
    for cycle in range(T):
        got = to_uniform_spikes(r, cycle, T)
        assert torch.equal(got, _legacy_to_uniform_spikes(r, cycle, T))


def test_dithered_counts_exact_including_edges():
    torch.manual_seed(1)
    T = 32
    r = torch.cat([
        torch.rand(500),
        torch.tensor([0.0, 1.0, 1.0 / T, (T - 1.0) / T, 0.5]),
    ]).unsqueeze(0).expand(3, -1).contiguous()
    offsets = uniform_phase_offsets(r.shape[-1], T)
    locked = torch.stack([to_uniform_spikes(r, c, T) for c in range(T)])
    dithered = torch.stack(
        [to_uniform_spikes(r, c, T, phase_offsets=offsets) for c in range(T)]
    )
    assert torch.equal(locked.sum(0), dithered.sum(0))


def test_dithered_pattern_actually_shifts():
    T = 32
    r = torch.full((64,), 0.25)
    offsets = uniform_phase_offsets(64, T)
    locked = torch.stack([to_uniform_spikes(r, c, T) for c in range(T)])
    dithered = torch.stack(
        [to_uniform_spikes(r, c, T, phase_offsets=offsets) for c in range(T)]
    )
    assert not torch.equal(locked, dithered)
    assert float(locked[0].sum()) == 64.0
    assert float(dithered[0].sum()) < 64.0


def test_uniform_spike_train_threads_phase_dither():
    torch.manual_seed(2)
    T = 32
    r = torch.rand(2, 5, 40)
    locked = uniform_spike_train(r, T)
    dithered = uniform_spike_train(r, T, phase_dither=True)
    assert torch.equal(locked.sum(0), dithered.sum(0))
    assert not torch.equal(locked, dithered)


def test_policy_walk_and_boundary_config_carry_the_flag():
    assert LifSegmentPolicy(retime=True, phase_dither=True).phase_dither is True
    assert LifSegmentPolicy().phase_dither is False
    assert "phase_dither" in inspect.signature(chip_aligned_segment_forward).parameters
    assert BoundaryConfig(
        simulation_length=8, spiking_mode="lif", cycle_accurate=True,
    ).phase_dither is False
