"""Polymorphic ramp strategy for the KD-blend tuners (LIF/TTFS)."""

from __future__ import annotations

import torch.nn.functional as F

from mimarsinan.models.spiking.training.blended_genuine_forward import (
    BlendedGenuineForward,
)
from mimarsinan.tuning.axes import BlendAxis
from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase
from mimarsinan.tuning.axes.blend_axis import GenuineBlendAxis
from mimarsinan.tuning.forward_install import LazyExecutorForward
from mimarsinan.tuning.orchestration.blend_ramp import (
    BlendActivation,
    KDClassificationLoss,
)


class RampStrategy:
    """How the model is driven 0→1. Each seam takes the tuner; the base returns
    the value-domain proxy ramp behavior (the golden non-destructive ramp)."""

    def is_bare_target(self, tuner) -> bool:
        """Whether ``base_activation`` IS the bare target node (no value blend)."""
        return False

    def make_axis(self, tuner) -> AdaptationAxisBase:
        return BlendAxis()

    def make_blend(self, tuner, old, target, rate):
        return BlendActivation(
            old, target, rate,
            target_type=tuner._target_activation_type,
            old_type=tuner._old_activation_type,
        )

    def ramp_forward(self, tuner, model) -> LazyExecutorForward | None:
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


class GenuineRampBase(RampStrategy):
    """Shared base of the genuine cascade ramps: ``base_activation`` IS the bare
    spiking target node (no value blend) and the deployed on-chip rebuild runs
    before the ramp forward is installed."""

    def is_bare_target(self, tuner) -> bool:
        return True

    def make_blend(self, tuner, old, target, rate):
        target.rate = float(rate)
        return target

    def after_install_blend_pre(self, tuner) -> None:
        tuner._finalize_rebuild()


class _BlendGenuineKDLoss(KDClassificationLoss):
    """KD (vs frozen teacher) on the blend output + a CE on the PURE genuine logits.

    The extra ``genuine_ce_alpha · CE(genuine_logits, y)`` term sharpens the pure cascade
    so intermediate-rate training lifts the rate-1 endpoint; skipped when no blend forward.
    """

    def __init__(self, teacher, *, genuine_ce_alpha: float, blend_forward_provider,
                 temperature: float = 3.0, alpha: float = 0.3):
        super().__init__(teacher, temperature=temperature, alpha=alpha)
        self.genuine_ce_alpha = float(genuine_ce_alpha)
        self._blend_forward = blend_forward_provider

    def __call__(self, model, x, y):
        loss = super().__call__(model, x, y)
        if self.genuine_ce_alpha > 0.0:
            blend = self._blend_forward()
            if blend is not None:
                loss = loss + self.genuine_ce_alpha * F.cross_entropy(
                    blend.genuine_logits(x), y,
                )
        return loss


class GenuineBlendRamp(GenuineRampBase):
    """Ramp a teacher<->genuine OUTPUT blend (``BlendedGenuineForward``, driven by
    ``GenuineBlendAxis``) calibrated to the teacher distribution; finalize deploys
    the pure cascade."""

    def make_axis(self, tuner):
        return GenuineBlendAxis()

    def ramp_forward(self, tuner, model):
        tuner._blend_forward = BlendedGenuineForward(
            model, tuner._teacher, tuner._T, rate=0.0,
            boundary_surrogate_temp=tuner._boundary_surrogate_temp,
        )
        return tuner._blend_forward

    def make_kd_loss(self, tuner):
        return _BlendGenuineKDLoss(
            tuner._teacher,
            genuine_ce_alpha=tuner._genuine_ce_alpha,
            blend_forward_provider=lambda: tuner._blend_forward,
        )

    def after_install_blend_pre(self, tuner) -> None:
        super().after_install_blend_pre(tuner)
        tuner._calibrate_to_teacher_distribution()

    def on_remove_forward(self, tuner) -> None:
        tuner._blend_forward = None
