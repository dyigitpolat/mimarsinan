"""[U-LIF2 W15] The membrane program check of integrate then emit: a declared register must hold a program's window sum.

Under integrate then emit a neuron holds its whole window's charge before it emits, so its membrane passes through every
partial sum of the window. Each axon delivers at most T spikes a window and the bias lands on T cycles, so every partial
sum of neuron j lies in v0 theta + T (sum_i min(w_ij, 0) + min(b_j, 0)) .. v0 theta + T (sum_i max(w_ij, 0) + max(b_j,
0)). A program whose bound leaves its declared register is refused by name at generation (Hard Core Mapping, before any
simulation); firing while integrating and an unbounded membrane are not judged here. Points are built explicitly."""

from __future__ import annotations

import inspect

import pytest
import torch

from mimarsinan.chip_simulation.spiking_point import stated_point
from mimarsinan.models.nn.lif_kernels import MembraneRailTouchedError

from unit.models import gated_window_programs as G

CHECK = "mimarsinan.certification.program_check.unit_forms.window_sum"
FWI, ITE = "fire_while_integrating", "integrate_then_emit"


def _point(window_firing: str, **register):
    return stated_point({"spiking_family": "lif", "spiking_variant": "synchronized", **register},
                        window_firing=window_firing)


def _fixture():
    """Three levels at T = 4, signed weights in [-2, 3], biases in [-1, 2]: a window sum an 8-bit register holds."""
    instance = G.layered_instance(G.chain(3), timesteps=4, seed=31, batch=12)
    return instance, G.identity_program(instance, retimed=False)


def test_w15_the_bound_is_every_partial_sum_of_the_window():
    bounds = G.new(CHECK, "core_window_sum_bounds")
    weights = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 0.0, 2.0]])
    lowest, highest = bounds(weights, torch.tensor([1.0, -2.0]), 4.0, window=5, membrane_init=-0.5)
    assert lowest.tolist() == [-2.0 + 5 * -2.0, -2.0 + 5 * (-1.0 - 2.0)]
    assert highest.tolist() == [-2.0 + 5 * (1.0 + 3.0 + 1.0), -2.0 + 5 * 2.0]
    no_bias_low, no_bias_high = bounds(weights, None, 4.0, window=5, membrane_init=0.0)
    assert (no_bias_low.tolist(), no_bias_high.tolist()) == ([-10.0, -5.0], [20.0, 10.0])
    # a threshold per neuron: the guard precharges each neuron by its own
    per_low, per_high = bounds(weights, None, torch.tensor([2.0, 8.0]), window=5, membrane_init=-0.5)
    assert (per_low.tolist(), per_high.tolist()) == ([-1.0 - 10.0, -4.0 - 5.0], [-1.0 + 20.0, -4.0 + 10.0])


def test_w15_a_register_that_holds_the_window_sum_passes_and_a_narrow_one_is_refused_by_name():
    instance, mapping = _fixture()
    held = G.new(CHECK, "require_window_sum_held")
    bounds = held(mapping, _point(ITE, membrane_bits=8, membrane_signed=True), window=4, membrane_init=0.0)
    assert bounds and all(-128 < b.lowest and b.highest < 127 for b in bounds)
    assert max(b.highest for b in bounds) > 7 or min(b.lowest for b in bounds) < -8   # the 4-bit register cannot
    with pytest.raises(G.new(CHECK, "WindowSumRefusal"), match=r"integrate_then_emit.*4-bit"):
        held(mapping, _point(ITE, membrane_bits=4, membrane_signed=True), window=4, membrane_init=0.0)


@pytest.mark.parametrize("window_firing,register", [
    (FWI, dict(membrane_bits=4, membrane_signed=True)),
    (ITE, {}),
])
def test_w15_firing_while_integrating_and_an_unbounded_membrane_are_not_judged(window_firing, register):
    _instance, mapping = _fixture()
    held = G.new(CHECK, "require_window_sum_held")
    assert held(mapping, _point(window_firing, **register), window=4, membrane_init=0.0) == ()
    assert held(mapping, None, window=4, membrane_init=0.0) == ()


def test_w15_a_register_that_clamps_must_hold_the_bound_inside_its_rails():
    """An unsigned register clamps where the window's negative charge falls below zero."""
    _instance, mapping = _fixture()
    with pytest.raises(G.new(CHECK, "WindowSumRefusal"), match="unsigned"):
        G.new(CHECK, "require_window_sum_held")(mapping, _point(ITE, membrane_bits=12), window=4, membrane_init=0.0)


def test_w15_the_check_agrees_with_execution_on_its_fixture():
    """The register the check admits never touches a rail when the gated executor runs it; the register it refuses
    touches one (the register's own assertion, the guard at execution)."""
    instance, mapping = _fixture()
    G.run_point(instance, mapping, soma_law=G.soma(membrane_bits=8, membrane_signed=True))
    with pytest.raises(MembraneRailTouchedError):
        G.run_point(instance, mapping, soma_law=G.soma(membrane_bits=4, membrane_signed=True))


def test_w15_hard_core_mapping_runs_the_check_at_generation_before_any_simulation():
    from mimarsinan.pipelining.pipeline_steps.mapping.hard_core_mapping_step import HardCoreMappingStep

    source = inspect.getsource(HardCoreMappingStep.process)
    at = source.index("require_window_sum_held(")
    assert at < source.index("run_spike_count_certificate_gate(") and at < source.index("run_hcm_mapping_metric(")
    assert "spiking_point" in source[at:source.index(")", at) + 120]


@pytest.mark.parametrize("lowest,highest,refused", [(-127.0, 126.0, False), (-127.0, 127.0, True),
                                                    (-128.0, 0.0, True)])
def test_w15_a_register_that_asserts_its_rails_holds_the_bound_strictly_inside_them(monkeypatch, lowest, highest,
                                                                                    refused):
    """A signed register refuses a membrane that reaches a rail (``enforce_membrane_rails``: equal is reached), so the
    check admits a bound strictly inside [-128, 127] only."""
    import importlib

    module = importlib.import_module(CHECK)
    bound = module.WindowSumBound(0, "stage", 0, lowest, highest)
    monkeypatch.setattr(module, "program_window_sum_bounds", lambda mapping, **kwargs: (bound,))
    point = _point(ITE, membrane_bits=8, membrane_signed=True)
    if refused:
        with pytest.raises(module.WindowSumRefusal, match="strictly inside its rails"):
            module.require_window_sum_held(None, point, window=4, membrane_init=0.0)
    else:
        assert module.require_window_sum_held(None, point, window=4, membrane_init=0.0) == (bound,)
