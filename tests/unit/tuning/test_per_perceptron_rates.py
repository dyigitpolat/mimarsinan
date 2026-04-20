"""Phase B1: AdaptationManager rates become per-perceptron.

``AdaptationManager`` historically exposed scalar rates that applied to every
perceptron uniformly (``activation_adaptation_rate``, ``clamp_rate``,
``quantization_rate``, ``shift_rate``, ``pruning_rate``, ``noise_rate``,
``scale_rate``).  Uniform rates make it impossible to advance sensitive
layers more slowly than robust ones, which is the root cause of several
degenerate tuning traces we've observed (e.g. one fragile layer collapses
the whole network's accuracy at moderate rates, forcing the whole schedule
to stall).

Phase B1 adds per-perceptron overrides on top of the scalar defaults:

1. Each rate field has a companion ``{rate}_overrides: dict[str, float]``.
2. ``get_rate(rate_name, perceptron)`` returns the per-perceptron override
   if present, else the scalar default.
3. ``update_activation(cfg, perceptron)`` consults ``get_rate`` for every
   decorator so the forward graph reflects the per-perceptron schedule.
4. A helper ``set_per_perceptron_schedule(rate_name, t, sensitivities)``
   accepts a global scalar ``t`` and a mapping of ``perceptron_name ->
   sensitivity`` and distributes the rate so that less sensitive perceptrons
   advance faster (their override value reaches 1.0 first).
5. The scalar field (e.g. ``clamp_rate``) continues to work unchanged so
   existing tuners that set a single scalar still get per-layer uniform
   behaviour.
"""

import torch
import pytest

from mimarsinan.tuning.adaptation_manager import AdaptationManager


class _FakePerceptron:
    """Lightweight stand-in for Perceptron that exposes just the fields
    AdaptationManager.get_rate() needs to look up an override."""

    def __init__(self, name):
        self.name = name
        self.activation_scale = torch.tensor(1.0)


class TestRateLookup:
    def test_default_is_scalar(self):
        am = AdaptationManager()
        am.clamp_rate = 0.5
        p = _FakePerceptron("layer0")
        assert am.get_rate("clamp_rate", p) == 0.5

    def test_override_wins_over_scalar(self):
        am = AdaptationManager()
        am.clamp_rate = 0.5
        am.set_per_perceptron_rate("clamp_rate", "layer0", 0.9)
        p = _FakePerceptron("layer0")
        assert am.get_rate("clamp_rate", p) == 0.9

    def test_override_is_per_perceptron(self):
        am = AdaptationManager()
        am.clamp_rate = 0.3
        am.set_per_perceptron_rate("clamp_rate", "layer0", 0.1)
        am.set_per_perceptron_rate("clamp_rate", "layer1", 0.9)
        p0 = _FakePerceptron("layer0")
        p1 = _FakePerceptron("layer1")
        p_other = _FakePerceptron("no_override")
        assert am.get_rate("clamp_rate", p0) == 0.1
        assert am.get_rate("clamp_rate", p1) == 0.9
        assert am.get_rate("clamp_rate", p_other) == 0.3

    def test_unknown_rate_name_raises(self):
        am = AdaptationManager()
        with pytest.raises((AttributeError, KeyError, ValueError)):
            am.get_rate("not_a_real_rate", _FakePerceptron("x"))

    def test_unnamed_perceptron_falls_back_to_scalar(self):
        """If a perceptron has no ``name`` attribute, overrides cannot apply.
        The scalar default must always be safe."""
        am = AdaptationManager()
        am.clamp_rate = 0.7
        class _Anon:
            pass
        p = _Anon()
        assert am.get_rate("clamp_rate", p) == 0.7


class TestSensitivityWeightedSchedule:
    def test_low_sensitivity_advances_first(self):
        """At a small global ``t``, the least-sensitive perceptron should be
        the furthest along (closest to 1.0), and the most-sensitive should
        trail."""
        am = AdaptationManager()
        sensitivities = {"a": 0.1, "b": 1.0, "c": 10.0}
        am.set_per_perceptron_schedule("clamp_rate", t=0.3, sensitivities=sensitivities)

        ra = am.get_rate("clamp_rate", _FakePerceptron("a"))
        rb = am.get_rate("clamp_rate", _FakePerceptron("b"))
        rc = am.get_rate("clamp_rate", _FakePerceptron("c"))
        assert 0.0 <= rc <= rb <= ra <= 1.0, (
            f"least-sensitive 'a' should be >= more-sensitive 'c'; got "
            f"a={ra:.3f}, b={rb:.3f}, c={rc:.3f}"
        )

    def test_t_zero_is_all_zero(self):
        am = AdaptationManager()
        sensitivities = {"a": 0.1, "b": 1.0, "c": 10.0}
        am.set_per_perceptron_schedule("clamp_rate", t=0.0, sensitivities=sensitivities)
        for name in sensitivities:
            assert am.get_rate("clamp_rate", _FakePerceptron(name)) == 0.0

    def test_t_one_is_all_one(self):
        am = AdaptationManager()
        sensitivities = {"a": 0.1, "b": 1.0, "c": 10.0}
        am.set_per_perceptron_schedule("clamp_rate", t=1.0, sensitivities=sensitivities)
        for name in sensitivities:
            assert am.get_rate("clamp_rate", _FakePerceptron(name)) == pytest.approx(1.0)

    def test_uniform_sensitivity_gives_uniform_rate(self):
        am = AdaptationManager()
        sensitivities = {"a": 1.0, "b": 1.0, "c": 1.0}
        am.set_per_perceptron_schedule("clamp_rate", t=0.5, sensitivities=sensitivities)
        rates = [
            am.get_rate("clamp_rate", _FakePerceptron(n))
            for n in sensitivities
        ]
        assert all(r == pytest.approx(rates[0]) for r in rates), (
            f"uniform sensitivities should give uniform rates; got {rates}"
        )

    def test_scheduler_monotonic_in_t(self):
        """For a fixed sensitivity vector, increasing t should never decrease
        any individual perceptron's rate."""
        am = AdaptationManager()
        sensitivities = {"a": 0.1, "b": 1.0, "c": 10.0}
        prev_rates = {n: 0.0 for n in sensitivities}
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            am.set_per_perceptron_schedule(
                "clamp_rate", t=t, sensitivities=sensitivities
            )
            for n in sensitivities:
                r = am.get_rate("clamp_rate", _FakePerceptron(n))
                assert r + 1e-9 >= prev_rates[n], (
                    f"rate for {n} decreased from {prev_rates[n]} to {r} "
                    f"when t grew"
                )
                prev_rates[n] = r


class TestResetOverrides:
    def test_clear_per_perceptron_overrides(self):
        am = AdaptationManager()
        am.clamp_rate = 0.4
        am.set_per_perceptron_rate("clamp_rate", "a", 0.9)
        am.clear_per_perceptron_overrides("clamp_rate")
        p = _FakePerceptron("a")
        assert am.get_rate("clamp_rate", p) == 0.4

    def test_clear_all_overrides(self):
        am = AdaptationManager()
        am.clamp_rate = 0.4
        am.quantization_rate = 0.2
        am.set_per_perceptron_rate("clamp_rate", "a", 0.9)
        am.set_per_perceptron_rate("quantization_rate", "a", 0.7)
        am.clear_per_perceptron_overrides()
        p = _FakePerceptron("a")
        assert am.get_rate("clamp_rate", p) == 0.4
        assert am.get_rate("quantization_rate", p) == 0.2
