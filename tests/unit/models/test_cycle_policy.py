"""Per-cycle neuron policy: LIF (reset, multi-spike) vs TTFS-cycle (latch, once)."""

import torch

from mimarsinan.models.spiking.cycle_policy import (
    LIFCyclePolicy,
    TTFSCyclePolicy,
    cycle_neuron_policy,
)


def test_factory_selects_by_mode():
    assert isinstance(cycle_neuron_policy("ttfs_cycle_based", "TTFS"), TTFSCyclePolicy)
    assert isinstance(cycle_neuron_policy("lif", "Default"), LIFCyclePolicy)
    assert isinstance(cycle_neuron_policy("ttfs_quantized", "TTFS"), LIFCyclePolicy)


def _run(policy, weight, inp, th, cycles, **step_kw):
    state = policy.make_state(1, weight.shape[0], device=weight.device, dtype=torch.float64)
    inp = inp.to(torch.float64)
    return [
        policy.step(state, weight.to(torch.float64), inp, th.to(torch.float64),
                    hw_bias=None, thresholding_mode="<=", output_dtype=torch.float64,
                    **step_kw).clone()
        for _ in range(cycles)
    ], state


def test_ttfs_policy_latches_once():
    policy = TTFSCyclePolicy()
    outs, state = _run(policy, torch.tensor([[0.4]]), torch.tensor([[1.0]]), torch.tensor(1.0), 4)
    assert [o.item() for o in outs] == [0.0, 0.0, 1.0, 1.0]
    assert bool(state["has_fired"].item())


def test_lif_policy_resets_and_can_refire():
    policy = LIFCyclePolicy("Default")
    # weight 1.0, latched input 1.0: each cycle adds 1.0 -> fires every cycle, subtractive reset.
    outs, _ = _run(policy, torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor(1.0), 3)
    assert [o.item() for o in outs] == [1.0, 1.0, 1.0]


def test_ttfs_policy_state_has_membrane_and_fired():
    state = TTFSCyclePolicy().make_state(2, 3, device="cpu", dtype=torch.float64)
    assert state["memb"].shape == (2, 3)
    assert state["has_fired"].shape == (2, 3)
    assert state["has_fired"].dtype == torch.bool
