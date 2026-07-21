"""[calculus §16.12] refcount pruning must cover the cycle-train cache too:
state_buffer_spikes entries (T x larger than value buffers) share their node's
lifetime — retaining them was the 80GB parity-read pathology."""

from __future__ import annotations

import torch

from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import decref_consumers


def test_decref_pops_value_and_spike_buffers_together():
    sb = {3: torch.zeros(2, 4), 5: torch.zeros(2, 4)}
    spikes = {3: torch.zeros(8, 2, 4), 5: torch.zeros(8, 2, 4)}
    remaining = {3: 1, 5: 2}
    decref_consumers(sb, remaining, [3, 5], state_buffer_spikes=spikes)
    assert 3 not in sb and 3 not in spikes and 3 not in remaining
    assert 5 in sb and 5 in spikes and remaining[5] == 1
    decref_consumers(sb, remaining, [5], state_buffer_spikes=spikes)
    assert 5 not in sb and 5 not in spikes and not remaining


def test_decref_without_spikes_dict_is_unchanged():
    sb = {7: torch.zeros(1, 2)}
    remaining = {7: 1}
    decref_consumers(sb, remaining, [7])
    assert 7 not in sb and not remaining
