"""LIF membrane guard: V0 restored by every reset, shifts the count staircase."""

import torch
from spikingjelly.activation_based import functional

from mimarsinan.models.nn.activations import LIFActivation


def _make(membrane_init: float = 0.0, T: int = 8) -> LIFActivation:
    return LIFActivation(
        T=T, activation_scale=1.0, thresholding_mode="<",
        firing_mode="Default", bias_mode="on_chip",
        membrane_init=membrane_init,
    )


def _reference_count(c: float, T: int, v0: float) -> int:
    v, count = v0, 0
    for _ in range(T):
        v += c
        if v > 1.0:
            v -= 1.0
            count += 1
    return count


def test_default_zero_and_attribute():
    lif = _make()
    assert lif.membrane_init == 0.0
    functional.reset_net(lif.if_node)
    assert float(torch.as_tensor(lif.if_node.v)) == 0.0


def test_guard_restored_by_every_reset():
    lif = _make(membrane_init=-0.25)
    assert lif.membrane_init == -0.25
    assert float(torch.as_tensor(lif.if_node.v)) == -0.25
    lif.set_cycle_accurate(True)
    assert float(torch.as_tensor(lif.if_node.v)) == -0.25
    for _ in range(3):
        lif(torch.full((4,), 0.7))
    functional.reset_net(lif.if_node)
    assert float(torch.as_tensor(lif.if_node.v).reshape(-1)[0]) == -0.25


def test_guard_shifts_constant_input_count():
    T = 8
    for v0 in (0.0, -0.25):
        lif = _make(membrane_init=v0, T=T)
        out = lif(torch.tensor([0.4, 0.9, 0.05]))
        for c, got in zip((0.4, 0.9, 0.05), out.tolist()):
            assert abs(got - _reference_count(c, T, v0) / T) < 1e-6
    guarded = _make(membrane_init=-0.25, T=T)(torch.tensor([0.4]))
    plain = _make(membrane_init=0.0, T=T)(torch.tensor([0.4]))
    assert float(guarded) < float(plain)
