"""Causality guard for the cascaded spiking cores: no output spike may depend on
a future-cycle input.

Both deployed cascade kernels integrate single/multi-spike arrivals cycle by
cycle. The invariant every neuromorphic backend must honour is temporal
causality: the spike emitted at cycle ``t`` is a function only of inputs at
cycles ``<= t``. These tests prove it by intervention — corrupt every input at
cycles ``> t_cut`` and require the outputs at cycles ``0..t_cut`` to stay
bit-identical.
"""

import pytest
import torch

from mimarsinan.models.spiking.ttfs_cycle_step import ttfs_cycle_contribute_and_fire
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire

_T = 8
_N_OUT = 6
_N_IN = 5


def _single_spike_input(seed: int) -> torch.Tensor:
    """(_T, _N_IN) genuine single-spike TTFS train: each axon fires once."""
    g = torch.Generator().manual_seed(seed)
    fire_cycle = torch.randint(0, _T, (_N_IN,), generator=g)
    inp = torch.zeros(_T, _N_IN)
    for j, c in enumerate(fire_cycle):
        inp[int(c), j] = 1.0
    return inp


def _multi_spike_input(seed: int) -> torch.Tensor:
    """(_T, _N_IN) dense binary train (LIF axons may spike every cycle)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(_T, _N_IN, generator=g) < 0.4).float()


def _weights(seed: int) -> torch.Tensor:
    return torch.randn(_N_OUT, _N_IN, generator=torch.Generator().manual_seed(seed))


def _corrupt_future(inp: torch.Tensor, t_cut: int) -> torch.Tensor:
    """Bit-flip every input entry strictly after ``t_cut``; leave 0..t_cut intact."""
    out = inp.clone()
    out[t_cut + 1:] = (out[t_cut + 1:] + 1.0) % 2.0
    return out


def _run_ttfs(weight, inp, thresholding_mode):
    memb = torch.zeros(1, _N_OUT)
    ramp = torch.zeros(1, _N_OUT)
    fired = torch.zeros(1, _N_OUT, dtype=torch.bool)
    return torch.cat(
        [
            ttfs_cycle_contribute_and_fire(
                memb, ramp, weight, inp[t:t + 1], torch.ones(_N_OUT), fired,
                hw_bias=None, thresholding_mode=thresholding_mode,
            ).clone()
            for t in range(_T)
        ],
        dim=0,
    )


def _run_lif(weight, inp, thresholding_mode, firing_mode):
    memb = torch.zeros(1, _N_OUT)
    return torch.cat(
        [
            lif_core_contribute_and_fire(
                memb, weight, inp[t:t + 1], torch.ones(_N_OUT),
                hw_bias=None, thresholding_mode=thresholding_mode,
                firing_mode=firing_mode,
            ).clone()
            for t in range(_T)
        ],
        dim=0,
    )


@pytest.mark.parametrize("thresholding_mode", ["<", "<="])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_ttfs_cascade_is_causal(thresholding_mode, seed):
    """Corrupting any future-cycle input never changes a past TTFS output."""
    weight = _weights(seed)
    inp = _single_spike_input(seed)
    baseline = _run_ttfs(weight, inp, thresholding_mode)
    for t_cut in range(_T - 1):
        perturbed = _run_ttfs(weight, _corrupt_future(inp, t_cut), thresholding_mode)
        assert torch.equal(baseline[:t_cut + 1], perturbed[:t_cut + 1]), (
            f"TTFS non-causal: future input (>{t_cut}) changed output at cycle <= {t_cut}"
        )


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_ttfs_fire_latch_is_monotone(seed):
    """A TTFS neuron fires at most once and never un-fires retroactively."""
    weight = _weights(seed)
    inp = _single_spike_input(seed)
    out = _run_ttfs(weight, inp, "<=")
    assert out.sum(dim=0).max() <= 1.0, "TTFS neuron fired more than once"


@pytest.mark.parametrize("firing_mode", ["Default", "Novena"])
@pytest.mark.parametrize("thresholding_mode", ["<", "<="])
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_lif_cascade_is_causal(firing_mode, thresholding_mode, seed):
    """Corrupting any future-cycle input never changes a past LIF output —
    for both the subtractive (Default) and zero (Novena) reset."""
    weight = _weights(seed)
    inp = _multi_spike_input(seed)
    baseline = _run_lif(weight, inp, thresholding_mode, firing_mode)
    for t_cut in range(_T - 1):
        perturbed = _run_lif(
            weight, _corrupt_future(inp, t_cut), thresholding_mode, firing_mode
        )
        assert torch.equal(baseline[:t_cut + 1], perturbed[:t_cut + 1]), (
            f"LIF ({firing_mode}) non-causal: future input (>{t_cut}) "
            f"changed output at cycle <= {t_cut}"
        )


def test_intervention_actually_perturbs_the_future():
    """Guard the guard: the future-input corruption must be observable downstream,
    else the causality assertions above would pass vacuously."""
    weight = _weights(3)
    inp = _multi_spike_input(3)
    baseline = _run_lif(weight, inp, "<=", "Default")
    changed_after_cut = False
    for t_cut in range(_T - 1):
        perturbed = _run_lif(weight, _corrupt_future(inp, t_cut), "<=", "Default")
        if not torch.equal(baseline[t_cut + 1:], perturbed[t_cut + 1:]):
            changed_after_cut = True
    assert changed_after_cut, "intervention never altered any future output — test is vacuous"
