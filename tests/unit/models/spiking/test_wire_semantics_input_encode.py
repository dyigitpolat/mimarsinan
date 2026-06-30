"""The segment-entry input encode is SSOT-driven by ``WireSemantics``.

The synchronized schedule snaps each neural-segment input onto the single-spike
grid before firing. That encode and the firing staircase MUST share one rounding
convention or they disagree at grid ties (the round-encode vs ceil-fire split that
made synchronized diverge from ``ttfs_quantized``). ``WireSemantics`` owns the
convention as a single field so the two cannot drift.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.models.spiking.wire_semantics import WireSemantics

S = 4
# Inputs in the lower half of each grid interval — where round and ceil disagree.
SPLIT = np.array([[0.7, 0.45, 0.95, 0.2]], dtype=np.float64)


class TestSpikeTimeRoundField:
    def test_default_is_round_backward_compatible(self):
        ws = WireSemantics(S)
        assert ws.spike_time_round == "round"

    def test_rejects_unknown_round_mode(self):
        with pytest.raises(ValueError, match="spike_time_round"):
            WireSemantics(S, spike_time_round="floor")

    @pytest.mark.parametrize("mode", ["round", "ceil"])
    def test_valid_modes_accepted(self, mode):
        assert WireSemantics(S, spike_time_round=mode).spike_time_round == mode


class TestEncodeHonorsConvention:
    def test_round_matches_legacy_rint(self):
        ws = WireSemantics(S, spike_time_round="round")
        got = ws.input_grid_quantize_np(SPLIT)
        expect = np.where(
            np.rint(S * (1.0 - SPLIT)) < S, (S - np.rint(S * (1.0 - SPLIT))) / S, 0.0
        )
        np.testing.assert_array_equal(got, expect)

    def test_ceil_uses_ceil_spike_time(self):
        ws = WireSemantics(S, spike_time_round="ceil")
        got = ws.input_grid_quantize_np(SPLIT)
        k = np.ceil(S * (1.0 - SPLIT))
        expect = np.where(k < S, (S - k) / S, 0.0)
        np.testing.assert_array_equal(got, expect)

    def test_round_and_ceil_differ_on_the_split(self):
        r = WireSemantics(S, spike_time_round="round").input_grid_quantize_np(SPLIT)
        c = WireSemantics(S, spike_time_round="ceil").input_grid_quantize_np(SPLIT)
        assert not np.array_equal(r, c)

    def test_torch_and_numpy_twins_agree(self):
        for mode in ("round", "ceil"):
            ws = WireSemantics(S, spike_time_round=mode)
            t = ws.input_grid_quantize(torch.tensor(SPLIT)).numpy()
            n = ws.input_grid_quantize_np(SPLIT)
            np.testing.assert_array_equal(t, n)


class TestEncodeFireConsistency:
    """``ceil`` makes the input-encode idempotent with the ``<=`` firing staircase:
    snapping an input then firing it yields the same grid cell as firing the snapped
    value again — the property that lets synchronized match the analytical kernel."""

    def test_ceil_encode_is_a_fixed_point_of_the_staircase(self):
        ws = WireSemantics(S, compare_mode="<=", spike_time_round="ceil")
        rates = torch.linspace(0.0, 1.0, 41, dtype=torch.float64).unsqueeze(0)
        snapped = ws.input_grid_quantize(rates)
        # Firing the already-snapped value (θ=1) leaves it unchanged.
        refired = ws.quantized_staircase(snapped, torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(refired, snapped, rtol=0, atol=0)
