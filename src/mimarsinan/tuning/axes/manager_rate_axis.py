"""Adapters for the ``AdaptationManager`` rate-field family (zero behavior change).

These wrap the ``AdaptationRateTuner`` mechanism: ``set_rate`` delegates to the
``perceptron_rate.apply_manager_rate`` SSOT (set one manager field, rebuild every
perceptron's decorator stack), so the axis path is byte-identical to today's
``AdaptationRateTuner._apply_rate``. State carriage is the single manager float.
"""

from __future__ import annotations

from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase
from mimarsinan.tuning.perceptron_rate import apply_manager_rate


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

    def set_rate(self, alpha: float) -> None:
        apply_manager_rate(
            self._model, self._manager, self._config, self.rate_attr, float(alpha)
        )

    def get_extra_state(self):
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
