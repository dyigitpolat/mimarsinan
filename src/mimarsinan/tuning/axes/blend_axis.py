"""Adapters for the KD-blend family (LIF/TTFS) — live ``BlendActivation.rate``.

``set_rate`` delegates to ``perceptron_rate.set_blend_rate`` (mutate every
perceptron's ``BlendActivation.rate`` in place, no decorator rebuild), byte-
identical to ``KDBlendAdaptationTuner._set_rate``. State carriage is the list of
per-perceptron blend rates. ``finalize`` is intentionally NOT owned here — the
parity-critical forward-install stays on the tuner's inherited ``_finalize``
(``test_finalize_contract`` forbids reimplementing it).
"""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase
from mimarsinan.tuning.perceptron_rate import set_blend_rate, set_surrogate_alpha


class BlendAxis(AdaptationAxisBase):
    """Linear ANN→target blend ramp via live per-perceptron ``BlendActivation.rate``."""

    name = "blend"
    interpolation_mode = "functional_blend"
    monotonicity = "expected"

    def set_rate(self, alpha: float) -> None:
        set_blend_rate(self._model, float(alpha))

    def get_extra_state(self):
        return [p.base_activation.rate for p in self._model.get_perceptrons()]

    def set_extra_state(self, extra) -> None:
        for perceptron, rate in zip(self._model.get_perceptrons(), extra):
            perceptron.base_activation.rate = float(rate)

    def descriptor(self) -> str:
        return self.name


class LIFAxis(BlendAxis):
    """ANN→LIFActivation blend ramp."""

    name = "lif"


class TTFSAxis(BlendAxis):
    """ANN→TTFSActivation blend ramp."""

    name = "ttfs"


class TTFSGenuineAxis(BlendAxis):
    """Genuine-cascade ramp: ANN→TTFS blend plus an annealed surrogate sharpness.

    ``set_rate`` walks the same per-perceptron ``BlendActivation.rate`` as
    ``TTFSAxis`` AND anneals the spike surrogate ``alpha`` smooth→sharp on the
    geometric schedule ``alpha_min·(alpha_max/alpha_min)**rate`` (read from
    ``ttfs_ramp_alpha_min`` / ``ttfs_ramp_alpha_max``), so the deployment
    dynamics are exact at rate 1 (``alpha_max``) while intermediate reps stay
    well-conditioned. ``alpha`` is backward-only — the forward stays bit-identical.
    """

    name = "ttfs_genuine"

    def _alpha_for_rate(self, r: float) -> float:
        config = self._config or {}
        alpha_min = float(config.get("ttfs_ramp_alpha_min", 0.5))
        alpha_max = float(config.get("ttfs_ramp_alpha_max", 2.0))
        return alpha_min * (alpha_max / alpha_min) ** float(r)

    def set_rate(self, alpha: float) -> None:
        rate = float(alpha)
        set_blend_rate(self._model, rate)
        set_surrogate_alpha(self._model, self._alpha_for_rate(rate))


class GenuineBlendAxis(AdaptationAxisBase):
    """Teacher<->genuine OUTPUT blend ramp: drives the installed forward's ``rate``.

    The genuine teacher->cascade blend ramp installs a ``BlendedGenuineForward``
    (``out = (1-rate)*teacher + rate*genuine``) as ``model.forward`` for the whole
    ramp. ``set_rate`` mutates that instance's live scalar ``rate`` in place — the
    blend is at the model OUTPUT, not per-perceptron, so there is no decorator
    rebuild and no ``BlendActivation.rate`` carriage. State carriage is the scalar
    forward rate. ``set_rate`` is a no-op when no blend forward is installed (e.g.
    during the finalize swap to the pure genuine cascade)."""

    name = "genuine_blend"
    interpolation_mode = "functional_blend"
    monotonicity = "expected"

    def _installed_forward(self):
        return self._model.__dict__.get("forward")

    def set_rate(self, alpha: float) -> None:
        forward = self._installed_forward()
        if forward is not None and hasattr(forward, "rate"):
            forward.rate = float(alpha)

    def get_extra_state(self):
        forward = self._installed_forward()
        return None if forward is None else float(getattr(forward, "rate", 0.0))

    def set_extra_state(self, extra) -> None:
        if extra is not None:
            self.set_rate(float(extra))

    def descriptor(self) -> str:
        return self.name
