"""The uniform comb's COUNT tie rule is the chip's (llround), not torch.round.

nevresim's ``UniformSpikeGenerator`` computes ``llround(rate * T)`` in C++
double: an exact half-integer product rounds AWAY from zero. ``torch.round``
and ``np.rint`` are half-to-EVEN, so rates on the odd 1/(2T) lattice — edges
LSQ exact-QAT trains values onto — used to encode one count LESS than the
chip delivered (t0_05 root-cause family, 2026-08-11)."""

import math

import numpy as np
import torch

from mimarsinan.chip_simulation.recording.spike_modes import (
    comb_spike_count,
    comb_spike_count_np,
    to_uniform_spikes,
)


def _llround(x: float) -> int:
    """C++ ``llround`` for the nonnegative comb domain: half away from zero."""
    frac = x - math.floor(x)
    return int(math.floor(x) + (1 if frac >= 0.5 else 0))


def _chip_comb_cycle(rate: float, cycle: int, T: int) -> int:
    """Literal python transcription of nevresim UniformSpikeGenerator."""
    n = _llround(rate * T)
    if n == 0:
        return 0
    if cycle >= T:
        return 0
    if n == T:
        return 1
    spacing = float(T) / n
    return int(
        (math.floor(cycle / spacing) < n)
        and (math.floor(math.fmod(cycle, spacing)) == 0)
    )


class TestCombCountTieRule:
    def test_count_matches_chip_llround_on_half_lattice(self):
        for T in (4, 8, 16):
            rates = torch.tensor(
                [k / (2.0 * T) for k in range(2 * T + 1)], dtype=torch.float64
            )
            got = comb_spike_count(rates, T).tolist()
            want = [_llround(float(r) * T) for r in rates]
            assert got == want, f"T={T}: {got} != llround {want}"

    def test_half_tie_rounds_away_from_zero_not_to_even(self):
        # rate*T == 0.5 and 2.5: torch.round gives 0 and 2 (banker's); the
        # chip delivers 1 and 3. The comb must side with the chip.
        rates = torch.tensor([0.125, 0.625], dtype=torch.float64)
        assert comb_spike_count(rates, 4).tolist() == [1, 3]
        assert torch.round(rates * 4).tolist() == [0.0, 2.0]  # the old, wrong rule

    def test_numpy_mirror_is_identical(self):
        for T in (4, 8):
            rates = np.array([k / (2.0 * T) for k in range(2 * T + 1)])
            got = comb_spike_count_np(rates, T)
            want = comb_spike_count(torch.tensor(rates), T).numpy()
            np.testing.assert_array_equal(got, want)

    def test_train_placement_matches_chip_generator_cycle_by_cycle(self):
        # Not just totals: every cycle of the torch comb equals the C++
        # generator's emission for every rate on the 1/(2T) lattice.
        for T in (4, 8):
            rates = torch.tensor(
                [k / (2.0 * T) for k in range(2 * T + 1)], dtype=torch.float64
            ).unsqueeze(0)
            for cycle in range(T):
                got = to_uniform_spikes(rates, cycle, T)[0].tolist()
                want = [
                    float(_chip_comb_cycle(float(r), cycle, T)) for r in rates[0]
                ]
                assert got == want, f"T={T} cycle={cycle}: {got} != {want}"

    def test_train_total_equals_count(self):
        T = 4
        rates = torch.tensor([[0.0, 0.125, 0.375, 0.625, 0.875, 1.0]], dtype=torch.float64)
        total = sum(to_uniform_spikes(rates, c, T) for c in range(T))
        want = comb_spike_count(rates, T).to(torch.float64)
        assert torch.equal(total, want)
