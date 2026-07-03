"""Polymorphic ramp strategy for the KD-blend tuners (LIF/TTFS)."""

from __future__ import annotations

from mimarsinan.tuning.axes import BlendAxis


class RampStrategy:
    """How the model is driven 0→1. Each seam takes the tuner; the base returns
    the value-domain proxy ramp behavior (the golden non-destructive ramp)."""

    def is_bare_target(self, tuner) -> bool:
        """Whether ``base_activation`` IS the bare target node (no value blend)."""
        return False

    def make_axis(self, tuner):
        return BlendAxis()

    def make_blend(self, tuner, old, target, rate):
        # Lazy: kd_blend_adaptation_tuner imports this module, so importing it at top
        # would be a circular import.
        from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
            BlendActivation,
        )

        return BlendActivation(
            old, target, rate,
            target_type=tuner._target_activation_type,
            old_type=tuner._old_activation_type,
        )

    def ramp_forward(self, tuner, model):
        """Cross-layer forward installed during the ramp (``None`` = value-domain)."""
        return None

    def make_kd_loss(self, tuner):
        return tuner._kd_classification_loss(tuner._teacher)

    def after_install_blend_pre(self, tuner) -> None:
        """Strategy-specific pre-ramp setup (genuine rebuild / blend distmatch),
        before the ramp forward is installed. Calibration stays in the tuner."""

    def on_remove_forward(self, tuner) -> None:
        """Strategy-specific cleanup when the instance forward is removed."""


class ValueDomainProxyRamp(RampStrategy):
    """The default: ramp the per-perceptron ``BlendActivation`` in the value domain
    (the plain class forward), the golden non-destructive ramp."""
