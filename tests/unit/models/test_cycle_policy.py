"""Per-cycle neuron policy: LIF (reset, multi-spike) vs cascaded TTFS (latch, once)."""

import torch

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.models.spiking.cycle_policy import (
    LIFCyclePolicy,
    TTFSGreedyCyclePolicy,
    cycle_neuron_policy,
)


def test_factory_selects_by_mode_and_schedule():
    law = DEFAULT_SOMA_LAW
    assert isinstance(
        cycle_neuron_policy("ttfs_cycle_based", "cascaded", "TTFS", soma_law=law),
        TTFSGreedyCyclePolicy)
    assert isinstance(
        cycle_neuron_policy("lif", "cascaded", "Default", soma_law=law),
        LIFCyclePolicy)
    assert isinstance(
        cycle_neuron_policy("rate", "synchronized", "Default", soma_law=law),
        LIFCyclePolicy)


def test_default_point_returns_the_identical_policy_class():
    """Byte-identical default-off: the serial class is NEVER constructed at
    the default point, and the object the executor gets is exactly today's."""
    policy = cycle_neuron_policy("lif", "cascaded", "Default",
                                 soma_law=DEFAULT_SOMA_LAW)
    assert type(policy) is LIFCyclePolicy
    assert policy.serial is False
    assert policy.membrane_bounds is None


def _run_spikes(policy, weight, inputs, th):
    """Feed a per-cycle list of input tensors; return (outputs, state)."""
    state = policy.make_state(1, weight.shape[0], device=weight.device, dtype=torch.float64)
    outs = [
        policy.step(state, weight.to(torch.float64), inp.to(torch.float64),
                    th.to(torch.float64), hw_bias=None, thresholding_mode="<=",
                    output_dtype=torch.float64).clone()
        for inp in inputs
    ]
    return outs, state


def test_ttfs_greedy_policy_single_spike_then_ramp_fires_once():
    # One arrival at cycle 0, weight 0.4 → ramp 0.4/cycle: 0.4,0.8,1.2 → fires at
    # t=2 with a SINGLE spike (silent before and after).
    spike, zero = torch.tensor([[1.0]]), torch.tensor([[0.0]])
    outs, state = _run_spikes(
        TTFSGreedyCyclePolicy(), torch.tensor([[0.4]]),
        [spike, zero, zero, zero], torch.tensor(1.0),
    )
    assert [o.item() for o in outs] == [0.0, 0.0, 1.0, 0.0]
    assert bool(state["has_fired"].item())


def test_lif_policy_resets_and_can_refire():
    inp = torch.tensor([[1.0]])
    outs, _ = _run_spikes(LIFCyclePolicy("Default"), torch.tensor([[1.0]]),
                          [inp, inp, inp], torch.tensor(1.0))
    assert [o.item() for o in outs] == [1.0, 1.0, 1.0]


def test_ttfs_greedy_state_has_membrane_ramp_and_fired():
    state = TTFSGreedyCyclePolicy().make_state(2, 3, device="cpu", dtype=torch.float64)
    assert state["memb"].shape == (2, 3)
    assert state["ramp_current"].shape == (2, 3)
    assert state["has_fired"].shape == (2, 3) and state["has_fired"].dtype == torch.bool


def test_ttfs_greedy_policy_advertises_single_spike_io():
    assert TTFSGreedyCyclePolicy().single_spike_io is True
    assert LIFCyclePolicy("Default").single_spike_io is False
