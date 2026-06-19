"""Polymorphic ramp strategy for the KD-blend tuners (LIF/TTFS).

One cohesive class per ramp mode bundles the seams that vary together — the rate
axis, the per-perceptron blend, the cross-layer ramp forward, the KD loss, and
the strategy-specific pre-ramp setup / forward-removal cleanup — so the tuner
holds ONE ``self._ramp`` and delegates instead of dispatching on a flag thicket.
Mirrors ``spiking/segment_policies.py``. The base must NOT import the concrete
tuner modules (the TTFS strategies live next to their forwards/losses in
``tuners/ttfs_cycle_adaptation_tuner.py``), so it stays a leaf the tuner imports.
"""

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
        from mimarsinan.tuning.orchestration.kd_blend_adaptation_tuner import (
            _KDClassificationLoss,
        )

        return _KDClassificationLoss(tuner._teacher)

    def after_install_blend_pre(self, tuner) -> None:
        """Strategy-specific pre-ramp setup (genuine rebuild / blend distmatch),
        before the ramp forward is installed. Calibration stays in the tuner."""

    def on_remove_forward(self, tuner) -> None:
        """Strategy-specific cleanup when the instance forward is removed."""


class ValueDomainProxyRamp(RampStrategy):
    """The default: ramp the per-perceptron ``BlendActivation`` in the value domain
    (the plain class forward), the golden non-destructive ramp."""
