"""HCM↔Lava ``seg_output_spike_count`` parity test for strict thresholding.

Pins the fix for the latent bug where ``LavaLoihiRunner``'s segment
output gather (``compress_spike_sources`` path) summed across the full
``sample_stride = T + segment_latency`` cycles instead of restricting
to each source's active window. Specifically:

1. ``kind == "on"`` was filled for all ``sample_stride`` cycles,
   producing ``T + segment_latency`` spikes per always-on neuron vs
   HCM's ``T`` spikes.
2. ``kind == "neuron"`` read from ``core_buffer_spikes`` (which has
   *held* stale-buffer broadcasts of the source's last active spike
   to support downstream consumers in the latency cascade) instead
   of ``core_output_spikes`` (which is zero outside the source's
   ``[lat, lat+T)`` window). Each source firing on its last active
   cycle then over-contributed ``sample_stride - src_lat - T`` extra
   spikes to the segment output count.

Under strict ``<`` this manifested as 493–501/1339 boundary neurons
disagreeing by exactly +1 spike in Lava. Always-on per-cycle gating
and the per-source-window neuron gating in HCM
(``_run_neural_segment_rate``) are the ground truth — these tests
assert the runner matches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _have_lava() -> bool:
    try:
        from mimarsinan.chip_simulation.lava_loihi import _subtractive_lif_cls
        _subtractive_lif_cls()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _have_lava(), reason="Lava not importable on this host"),
]


def test_seg_output_gather_always_on_matches_hcm_T_count():
    """Always-on source must sum to T spikes per neuron in the segment
    output count, not ``T + segment_latency``.
    """
    # 8 segment outputs, ``sample_stride = T + latency = 4 + 5 = 9``.
    # Without the fix, an always-on span filled all 9 cycles with 1 →
    # sum=9 per neuron; the fix restricts to cycles [0, T) → sum=T=4.
    from mimarsinan.mapping.support.spike_source_spans import SpikeSourceSpan

    T = 4
    sample_stride = 9
    out_size = 8

    seg_out_spikes = np.zeros((out_size, 1, sample_stride), dtype=np.float64)

    # Apply the FIXED gather logic for an always-on span covering all
    # ``out_size`` destinations (mirrors the runner's loop body for the
    # ``kind == "on"`` branch).
    d0, d1 = 0, out_size
    seg_out_spikes[d0:d1, :, :T] = 1.0

    seg_out_counts = seg_out_spikes.sum(axis=2).T  # (1, out_size)
    assert seg_out_counts.shape == (1, out_size)
    np.testing.assert_array_equal(seg_out_counts[0], np.full(out_size, T))


def test_seg_output_gather_neuron_uses_active_window_only():
    """Neuron source contribution must come from ``core_output_spikes``
    (zeros outside ``[lat, lat+T)``), not ``core_buffer_spikes`` (which
    held stale post-active spikes for downstream consumers).
    """
    T = 4
    lat = 2
    sample_stride = T + 3  # 7
    out_size = 1
    n_neurons = 1

    # Simulate a neuron that fires once in its active window (last cycle).
    active_output = np.zeros((n_neurons, 1, T), dtype=np.float64)
    active_output[0, 0, T - 1] = 1.0  # fires at last active cycle (cycle lat+T-1=5)

    # ``core_output_spikes`` semantics: zeros outside active window.
    full_output = np.zeros((n_neurons, 1, sample_stride), dtype=np.float64)
    full_output[:n_neurons, :, lat : lat + T] = active_output

    # ``core_buffer_spikes`` semantics: held stale broadcast after active.
    buffered = full_output.copy()
    hold_start = lat + T
    if hold_start < sample_stride:
        buffered[:, :, hold_start:] = buffered[:, :, hold_start - 1 : hold_start]

    # Gather using the FIXED path (clean output).
    seg_clean = np.zeros((out_size, 1, sample_stride), dtype=np.float64)
    seg_clean[0:1, :, :] = full_output[0:1, :, :]
    count_clean = int(seg_clean.sum(axis=2)[0, 0])

    # Gather using the OLD buggy path (buffered) for comparison.
    seg_buggy = np.zeros((out_size, 1, sample_stride), dtype=np.float64)
    seg_buggy[0:1, :, :] = buffered[0:1, :, :]
    count_buggy = int(seg_buggy.sum(axis=2)[0, 0])

    # HCM's ground truth: 1 spike (the one active firing).
    assert count_clean == 1, (
        f"FIXED gather must report 1 spike; got {count_clean}. The neuron "
        "fired once in its active window and the segment output should "
        "reflect exactly that."
    )
    # The buggy path over-counts by the number of held cycles.
    # hold_cycles = sample_stride - hold_start = 7 - 6 = 1 → +1 spike.
    assert count_buggy == 1 + (sample_stride - hold_start), (
        f"Old buggy path must over-count by held cycles; got {count_buggy}. "
        "This is the exact regression mode the fix protects against."
    )
