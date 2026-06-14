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
from mimarsinan.tuning.perceptron_rate import set_blend_rate


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
