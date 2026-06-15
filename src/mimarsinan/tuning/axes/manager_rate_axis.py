"""Adapters for the ``AdaptationManager`` rate-field family.

``RateAdjustedDecorator``-backed rates (``quantization_rate`` / ``clamp_rate`` /
``activation_adaptation_rate``) drive a shared in-place ``RateBuffer``: the decorator
stack is built once, then a ramp step is an O(1) buffer write (the report's W9 fix).
This path is output- and RNG-conformant with a full per-step rebuild (see
``test_rate_buffer``). State carriage is the single buffer float.

``NoisyDropout``-backed rates (``noise_rate``) are not decorator-driven, so they keep
the rebuild path: ``set_rate`` delegates to the ``perceptron_rate.apply_manager_rate``
SSOT (set one manager field, rebuild every perceptron's decorator stack).
"""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase
from mimarsinan.tuning.perceptron_rate import apply_manager_rate, rebuild_activations


# Rates whose decorator is a RateAdjustedDecorator (read a live RateBuffer).
_INPLACE_ELIGIBLE_RATES = frozenset(
    {"quantization_rate", "clamp_rate", "activation_adaptation_rate"}
)


class ManagerRateAxis(AdaptationAxisBase):
    """Drive one ``adaptation_manager.<rate_attr>`` across all perceptrons."""

    rate_attr: str = "quantization_rate"
    name: str = "manager_rate"

    def __init__(self, rate_attr: str | None = None, *, name: str | None = None):
        super().__init__()
        if rate_attr is not None:
            self.rate_attr = rate_attr
        if name is not None:
            self.name = name

    def attach(self, model, adaptation_manager, config) -> None:
        super().attach(model, adaptation_manager, config)
        # A fresh attach targets a manager with no buffer bound yet, so the
        # one-time stack install must run again.
        self._inplace_installed = False

    def _inplace_enabled(self) -> bool:
        # RateAdjustedDecorator-backed rates drive a shared in-place RateBuffer
        # (build the decorator stack once, then O(1) writes); NoisyDropout-backed
        # rates (noise_rate) are not decorator-driven, so they keep the rebuild path.
        return self.rate_attr in _INPLACE_ELIGIBLE_RATES

    def set_rate(self, alpha: float) -> None:
        alpha = float(alpha)
        if self._inplace_enabled():
            self._set_rate_inplace(alpha)
            return
        apply_manager_rate(
            self._model, self._manager, self._config, self.rate_attr, alpha
        )

    def _set_rate_inplace(self, alpha: float) -> None:
        """Write the shared ``RateBuffer`` in place; build the stack once.

        The decorators read the buffer live (the O(1) ramp step), so the manager
        field is no longer load-bearing — but it is kept in sync as a write-through
        so state queries (and the pickled manager) never see a stale rate."""
        buffer = self._manager.bind_rate_buffer(self.rate_attr)
        first_install = getattr(self, "_inplace_installed", False) is False
        buffer.set(alpha)
        setattr(self._manager, self.rate_attr, alpha)
        if first_install:
            rebuild_activations(self._model, self._manager, self._config)
            self._inplace_installed = True

    def get_extra_state(self):
        buffer = self._manager._rate_buffer(self.rate_attr) if self._inplace_enabled() else None
        if buffer is not None:
            return float(buffer.alpha)
        return getattr(self._manager, self.rate_attr)

    def set_extra_state(self, extra) -> None:
        self.set_rate(extra)

    def descriptor(self) -> str:
        return f"manager_rate:{self.rate_attr}"


class ClampAxis(ManagerRateAxis):
    """Gradual activation clamping (``MixAdjustmentStrategy`` — functional blend)."""

    rate_attr = "clamp_rate"
    name = "clamp"
    interpolation_mode = "functional_blend"
    monotonicity = "expected"


class ActQuantAxis(ManagerRateAxis):
    """Activation quantization (nested random-mask path — stochastic)."""

    rate_attr = "quantization_rate"
    name = "activation_quantization"
    interpolation_mode = "stochastic_mask"
    is_stochastic = True


class NoiseAxis(ManagerRateAxis):
    """Training-noise injection (``NoisyDropout`` — stochastic regularizer)."""

    rate_attr = "noise_rate"
    name = "noise"
    interpolation_mode = "parameter_path"
    is_stochastic = True


class ActivationAdaptationAxis(ManagerRateAxis):
    """Blend the base activation toward chip ReLU (``activation_adaptation_rate``)."""

    rate_attr = "activation_adaptation_rate"
    name = "activation_adaptation"
    interpolation_mode = "functional_blend"
