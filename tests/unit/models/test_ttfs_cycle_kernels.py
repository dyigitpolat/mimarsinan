"""Greedy cascaded TTFS kernel: single-spike fire + consumer-side ramp.

A genuine TTFS neuron fires exactly once — the kernel emits a single spike on the
firing cycle and is silent after. The integration is a **ramp** reconstructed
from single-spike arrivals: once a spike arrives on an axon at ``t_j`` it
contributes its weight every subsequent cycle (``membrane(t)=Σ w_j·(t−t_j)``).
Building block for the HCM cascade executor; mirrored by nevresim's single-spike
``SpikingCompute`` neuron and the SANA-FE single-spike soma + ramping dendrite.
"""

import torch

from mimarsinan.models.nn.ttfs_cycle_kernels import ttfs_cycle_fire_once
from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire


def _step(memb, ramp, weight, inp, th, has_fired, **kw):
    return ttfs_cycle_contribute_and_fire(
        memb, ramp, weight, inp, th, has_fired,
        hw_bias=None, thresholding_mode="<=", **kw,
    )


class TestSingleSpikeRamp:
    def test_single_arrival_ramps_and_fires_exactly_once(self):
        # One input spike at cycle 0, weight 0.4 → ramp adds 0.4/cycle:
        # membrane 0.4, 0.8, 1.2 → fires at t=2, then SILENT (single spike).
        memb = torch.zeros(1, 1)
        ramp = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        w = torch.tensor([[0.4]]); th = torch.tensor(1.0)
        arrival = torch.tensor([[1.0]]); silent = torch.tensor([[0.0]])
        outs = [_step(memb, ramp, w, (arrival if c == 0 else silent), th, has_fired).item()
                for c in range(5)]
        assert outs == [0.0, 0.0, 1.0, 0.0, 0.0]  # exactly one spike
        assert memb.item() > 1.0  # no reset — membrane keeps ramping

    def test_membrane_is_a_ramp_from_arrival_time(self):
        # Spike arrives at cycle 1; membrane must be 0, then ramp w per cycle.
        memb = torch.zeros(1, 1); ramp = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        w = torch.tensor([[0.3]]); th = torch.tensor(100.0)  # never fires
        spikes = [[[0.0]], [[1.0]], [[0.0]], [[0.0]]]
        membs = []
        for s in spikes:
            _step(memb, ramp, w, torch.tensor(s), th, has_fired)
            membs.append(round(memb.item(), 6))
        # t0: no arrival → 0. t1: arrives → +0.3. t2: ramp +0.3 → 0.6. t3: 0.9.
        assert membs == [0.0, 0.3, 0.6, 0.9]

    def test_never_fires_stays_silent(self):
        memb = torch.zeros(1, 1); ramp = torch.zeros(1, 1)
        has_fired = torch.zeros(1, 1, dtype=torch.bool)
        w = torch.tensor([[0.0]]); th = torch.tensor(1.0)
        inp = torch.tensor([[1.0]])
        assert [_step(memb, ramp, w, inp, th, has_fired).item() for _ in range(5)] == [0.0] * 5


class TestThresholdingMode:
    def test_strict_vs_inclusive_at_exact_threshold(self):
        for mode, first in (("<=", 1.0), ("<", 0.0)):
            memb = torch.zeros(1, 1); ramp = torch.zeros(1, 1)
            has_fired = torch.zeros(1, 1, dtype=torch.bool)
            out = ttfs_cycle_contribute_and_fire(
                memb, ramp, torch.tensor([[1.0]]), torch.tensor([[1.0]]),
                torch.tensor(1.0), has_fired, hw_bias=None, thresholding_mode=mode,
            )
            assert out.item() == first, mode


class TestFireOncePrimitive:
    def test_emits_only_on_the_crossing_cycle(self):
        memb = torch.tensor([[1.5, 0.2]]); has_fired = torch.zeros(1, 2, dtype=torch.bool)
        out = ttfs_cycle_fire_once(memb, torch.tensor(1.0), has_fired, thresholding_mode="<=")
        assert out.tolist() == [[1.0, 0.0]] and has_fired.tolist() == [[True, False]]
        # next cycle: neuron 1 crosses → single spike; neuron 0 already fired → silent.
        out2 = ttfs_cycle_fire_once(torch.tensor([[5.0, 1.0]]), torch.tensor(1.0),
                                    has_fired, thresholding_mode="<=")
        assert out2.tolist() == [[0.0, 1.0]]
        assert has_fired.tolist() == [[True, True]]
