"""Genuine binary-spike TTFS-cycle simulation kernel (fire-once + latch).

Unlike the analytical ``ttfs_quantized_activation`` (closed-form from a final V),
this kernel is a per-cycle integrate-and-fire that latches its output once it
fires — the building block of the cycle-based simulation forward.
"""

import torch

from mimarsinan.models.nn.ttfs_cycle_kernels import ttfs_cycle_fire_and_latch
from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire


def _step(memb, weight, inp, th, has_fired, **kw):
    return ttfs_cycle_contribute_and_fire(
        memb, weight, inp, th, has_fired,
        hw_bias=None, thresholding_mode="<=", **kw,
    )


class TestFireOnceLatch:
    def test_latches_high_after_first_crossing_and_never_resets(self):
        # One neuron, one axon, weight 0.4, threshold 1.0. A latched input
        # spike (1.0) each cycle ramps membrane: 0.4, 0.8, 1.2 (fires at t=2),
        # then stays latched.
        memb = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        weight = torch.tensor([[0.4]])
        th = torch.tensor([[1.0]])
        inp = torch.tensor([[1.0]])

        outs = [_step(memb, weight, inp, th, has_fired).item() for _ in range(4)]
        assert outs == [0.0, 0.0, 1.0, 1.0]
        # No reset: membrane keeps integrating past threshold.
        assert memb.item() > 1.0

    def test_decode_equals_S_minus_fire_cycle_over_S(self):
        # Latched output summed over S cycles = S - fire_cycle, so /S recovers
        # the time-decoded activation (S - fire_cycle)/S.
        S = 5
        memb = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        weight = torch.tensor([[0.5]])
        th = torch.tensor([[1.0]])
        inp = torch.tensor([[1.0]])  # ramps 0.5, 1.0 -> fires at t=1

        latched = torch.stack([_step(memb, weight, inp, th, has_fired) for _ in range(S)])
        fire_cycle = 1
        assert latched.sum().item() == (S - fire_cycle)
        assert latched.sum().item() / S == (S - fire_cycle) / S

    def test_never_fires_stays_zero(self):
        memb = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        weight = torch.tensor([[0.1]])
        th = torch.tensor([[1.0]])
        inp = torch.tensor([[1.0]])
        outs = [_step(memb, weight, inp, th, has_fired).item() for _ in range(5)]
        assert outs == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestThresholdingMode:
    def test_strict_vs_inclusive_at_exact_threshold(self):
        # membrane reaches exactly 1.0 at t=0.
        for mode, expected_first in (("<=", 1.0), ("<", 0.0)):
            memb = torch.zeros(1, 1)
            has_fired = torch.zeros(1, 1, dtype=torch.bool)
            out = ttfs_cycle_contribute_and_fire(
                memb, torch.tensor([[1.0]]), torch.tensor([[1.0]]),
                torch.tensor([[1.0]]), has_fired,
                hw_bias=None, thresholding_mode=mode,
            )
            assert out.item() == expected_first, mode


class TestBiasPerCycle:
    def test_bias_is_integrated_each_cycle(self):
        # No input, bias 0.5/cycle, threshold 1.0 -> fires at t=1 (0.5, 1.0).
        memb = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        weight = torch.zeros(1, 1)
        th = torch.tensor([[1.0]])
        inp = torch.zeros(1, 1)
        outs = [
            ttfs_cycle_contribute_and_fire(
                memb, weight, inp, th, has_fired,
                hw_bias=torch.tensor([0.5]), thresholding_mode="<=",
            ).item()
            for _ in range(3)
        ]
        assert outs == [0.0, 1.0, 1.0]


class TestFireAndLatchPrimitive:
    def test_latch_state_accumulates(self):
        memb = torch.tensor([[1.5, 0.2]])
        has_fired = torch.zeros(1, 2, dtype=torch.bool)
        out = ttfs_cycle_fire_and_latch(memb, torch.tensor([[1.0]]), has_fired, thresholding_mode="<=")
        assert out.tolist() == [[1.0, 0.0]]
        assert has_fired.tolist() == [[True, False]]
        # Next cycle: second neuron crosses; first stays latched even if it dipped.
        memb2 = torch.tensor([[0.0, 1.0]])
        out2 = ttfs_cycle_fire_and_latch(memb2, torch.tensor([[1.0]]), has_fired, thresholding_mode="<=")
        assert out2.tolist() == [[1.0, 1.0]]
