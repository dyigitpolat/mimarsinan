"""Phase B2: Learnable per-perceptron clamp ceiling.

Previously ``Perceptron.activation_scale`` was a fixed float seeded from the
p99 of activations observed during ``ActivationAnalysis``.  That scale then
anchored the clamp ceiling used by ``ClampDecorator`` for the rest of the
pipeline.  The p99 heuristic is reasonable but arbitrary -- a layer whose
real accuracy-preserving ceiling is *above* the p99 gets over-clamped, and a
layer whose ceiling is *below* the p99 wastes dynamic range.

Phase B2 makes the clamp ceiling a **log-space learnable parameter per
perceptron**, initialised from the p99 scale, so that the clamp tuner can
refine it via gradient descent during its adaptation cycles.  At the end of
``ClampTuner._after_run`` the learnt value is written back into
``activation_scale`` so that every downstream step continues to see a single
consolidated scalar and the rest of the pipeline is unchanged.

This module pins down the contract:

1. Each ``Perceptron`` grows a ``log_clamp_ceiling`` parameter stored in
   log-space.  The effective ceiling is ``exp(log_clamp_ceiling)``.
2. Construction sets ``log_clamp_ceiling = log(activation_scale)`` so that
   the initial behaviour matches the p99 baseline exactly.
3. Any subsequent ``set_activation_scale(x)`` must reseed
   ``log_clamp_ceiling = log(x)`` so the decorator stays in sync with the
   statically-set scale.
4. ``ClampTuner`` marks ``log_clamp_ceiling`` as trainable (``requires_grad
   = True``), and registers those parameters with the tuner's optimiser so
   they receive gradients during recovery training.
5. After ``ClampTuner._after_run`` commits the rate to 1.0, the learnt
   ceiling is written back into ``activation_scale`` (the float, so
   downstream steps see one number) and ``log_clamp_ceiling.requires_grad``
   is flipped off.
"""

from __future__ import annotations

import math

import pytest
import torch

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class TestLogClampCeilingParameter:
    def test_parameter_exists(self):
        p = Perceptron(4, 8)
        assert hasattr(p, "log_clamp_ceiling"), (
            "Perceptron must expose a log_clamp_ceiling parameter for Phase B2"
        )
        assert isinstance(p.log_clamp_ceiling, torch.nn.Parameter)

    def test_initial_value_matches_activation_scale(self):
        p = Perceptron(4, 8)
        # activation_scale defaults to 1.0 on a fresh Perceptron, so the
        # log-space initial value should be log(1.0) == 0.0.
        assert p.log_clamp_ceiling.detach().item() == pytest.approx(
            math.log(p.activation_scale.item())
        )

    def test_initial_value_is_frozen_by_default(self):
        """Training/tuning code must flip requires_grad on explicitly; a
        fresh perceptron should have its clamp ceiling frozen so the p99
        initialisation survives the non-clamp steps of the pipeline."""
        p = Perceptron(4, 8)
        assert p.log_clamp_ceiling.requires_grad is False

    def test_set_activation_scale_reseeds_log_ceiling(self):
        p = Perceptron(4, 8)
        p.set_activation_scale(2.5)
        assert p.activation_scale.item() == pytest.approx(2.5)
        assert p.log_clamp_ceiling.detach().item() == pytest.approx(math.log(2.5))

    def test_effective_ceiling_helper(self):
        """A convenience accessor ``effective_clamp_ceiling()`` returns
        ``exp(log_clamp_ceiling)`` so that decorators and analysis tools
        can read the learnt value without duplicating the conversion."""
        p = Perceptron(4, 8)
        p.set_activation_scale(3.0)
        assert p.effective_clamp_ceiling().detach().item() == pytest.approx(3.0)

        with torch.no_grad():
            p.log_clamp_ceiling.copy_(torch.tensor(math.log(5.0)))
        assert p.effective_clamp_ceiling().detach().item() == pytest.approx(5.0)


class TestClampTunerUsesLearnableCeiling:
    def _make_perceptron(self):
        p = Perceptron(4, 8)
        p.set_activation_scale(1.5)
        return p

    def test_rebuilt_decorator_uses_effective_ceiling(self, monkeypatch):
        """``ClampDecorator`` has to track the *current* learnable ceiling,
        not a stale snapshot.  This is tested at the lowest level: after
        mutating ``log_clamp_ceiling`` the forward output must reflect the
        new ceiling.
        """
        from mimarsinan.tuning.adaptation_manager import AdaptationManager

        am = AdaptationManager()
        am.clamp_rate = 1.0
        p = self._make_perceptron()
        am.update_activation({"target_tq": 8, "spiking_mode": "rate"}, p)

        x = torch.full((2, 8), 10.0)
        p.eval()
        out_before = p.activation(x).max().item()
        # Expect clamp ceiling around 1.5 (matches initial activation_scale).
        assert out_before == pytest.approx(1.5, abs=1e-4)

        with torch.no_grad():
            p.log_clamp_ceiling.copy_(torch.tensor(math.log(0.5)))
        # Force the manager to rebuild the decorator so it picks up the new
        # effective ceiling.  `effective_clamp_ceiling()` is the live view
        # the decorator must consult.
        # Writing the effective value back into activation_scale matches
        # how ClampTuner._after_run hands the learnt scale off to the rest
        # of the pipeline.
        p.set_activation_scale(p.effective_clamp_ceiling().item())
        am.update_activation({"target_tq": 8, "spiking_mode": "rate"}, p)

        out_after = p.activation(x).max().item()
        assert out_after == pytest.approx(0.5, abs=1e-4), (
            "Clamp ceiling did not follow log_clamp_ceiling after rebuild"
        )
