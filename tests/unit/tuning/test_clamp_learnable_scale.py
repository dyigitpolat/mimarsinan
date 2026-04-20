"""Tests for ``ClampDecorator`` with a learnable activation scale.

Design (see ``smooth_adaptation_refactor`` plan):
- ``ClampDecorator`` may be constructed with an optional learnable
  ``scale_param``. When provided, the effective clamp ceiling is the
  learnable scalar; when omitted, behaviour is identical to the legacy
  fixed-ceiling construction.
- A regulariser pulls the learnable scale toward a reference
  (typically ``p99 * (1 + margin)``) so the IR-visible scale remains
  bounded.
- At the end of ``ClampTuner``, the learned scale is frozen back onto
  ``perceptron.set_activation_scale(...)``.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.models.decorators import ClampDecorator


class TestBackwardsCompatibleConstruction:
    def test_construct_without_learnable(self):
        dec = ClampDecorator(torch.tensor(0.0), torch.tensor(1.0))
        x = torch.tensor([2.0, -1.0, 0.5])
        out = dec.output_transform(x)
        assert torch.all(out <= 1.0 + 1e-6)
        assert torch.all(out >= 0.0 - 1e-6)

    def test_construct_with_scale_param_uses_learnable(self):
        scale = nn.Parameter(torch.tensor(1.5))
        dec = ClampDecorator(torch.tensor(0.0), scale_param=scale)
        x = torch.tensor([3.0, -0.5, 1.0])
        out = dec.output_transform(x)
        assert torch.all(out <= 1.5 + 1e-6)


class TestLearnableScaleIsTrainable:
    def test_scale_param_receives_gradient(self):
        scale = nn.Parameter(torch.tensor(2.0))
        dec = ClampDecorator(torch.tensor(0.0), scale_param=scale)
        x = torch.randn(16, requires_grad=False) + 3.0  # forces saturation
        out = dec.output_transform(x)
        loss = out.sum()
        loss.backward()
        assert scale.grad is not None
        # Saturation biases the clamp; we only require the gradient to be finite.
        assert torch.isfinite(scale.grad).item()


class TestRegulariserPenalty:
    def test_regulariser_zero_at_reference(self):
        from mimarsinan.tuning.tuners.clamp_tuner import clamp_scale_regulariser

        scale = nn.Parameter(torch.tensor(1.0))
        reference = 1.0
        penalty = clamp_scale_regulariser(scale, reference)
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_regulariser_penalises_drift(self):
        from mimarsinan.tuning.tuners.clamp_tuner import clamp_scale_regulariser

        scale = nn.Parameter(torch.tensor(2.0))
        reference = 1.0
        penalty = clamp_scale_regulariser(scale, reference)
        assert penalty.item() > 0.0

    def test_regulariser_is_symmetric(self):
        from mimarsinan.tuning.tuners.clamp_tuner import clamp_scale_regulariser

        p1 = clamp_scale_regulariser(nn.Parameter(torch.tensor(1.5)), 1.0).item()
        p2 = clamp_scale_regulariser(nn.Parameter(torch.tensor(0.5)), 1.0).item()
        assert p1 == pytest.approx(p2, rel=1e-3)


class TestFreezeToScalar:
    def test_freeze_detaches_to_float(self):
        from mimarsinan.tuning.tuners.clamp_tuner import freeze_learnable_scale

        scale = nn.Parameter(torch.tensor(1.75))
        frozen = freeze_learnable_scale(scale)
        assert isinstance(frozen, float)
        assert frozen == pytest.approx(1.75)
