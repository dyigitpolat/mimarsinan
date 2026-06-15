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

import torch

from mimarsinan.models.nn.decorators.adjustment import RandomMaskAdjustmentStrategy
from mimarsinan.models.nn.decorators.transforms import NoisyDropout
from mimarsinan.tuning.axes.adaptation_axis import AdaptationAxisBase
from mimarsinan.tuning.perceptron_rate import apply_manager_rate, rebuild_activations


# Rates whose decorator is a RateAdjustedDecorator (read a live RateBuffer).
_INPLACE_ELIGIBLE_RATES = frozenset(
    {"quantization_rate", "clamp_rate", "activation_adaptation_rate"}
)

# Containers walked to reach the stochastic decision objects in an activation.
_DECISION_CHILD_ATTRS = ("base_activation", "decorator", "adjustment_strategy", "target_activation")
_DECISION_SEQ_ATTRS = ("decorators", "strategies")


def _iter_decision_objects(node, seen=None):
    """Yield the stochastic-decision objects (``RandomMaskAdjustmentStrategy`` /
    ``NoisyDropout``) reachable from an activation node through its decorator and
    strategy containers."""
    if seen is None:
        seen = set()
    if node is None or id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, (RandomMaskAdjustmentStrategy, NoisyDropout)):
        yield node
    for attr in _DECISION_CHILD_ATTRS:
        yield from _iter_decision_objects(getattr(node, attr, None), seen)
    for attr in _DECISION_SEQ_ATTRS:
        seq = getattr(node, attr, None)
        if isinstance(seq, (list, tuple)):
            for child in seq:
                yield from _iter_decision_objects(child, seen)


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
        self._decision_seed = None
        self._decision_generators = {}

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
        else:
            apply_manager_rate(
                self._model, self._manager, self._config, self.rate_attr, alpha
            )
        # Rebuilds (or the one-time in-place install) create fresh decorator
        # objects; re-wire the seeded generator into them. No-op when unseeded.
        self._wire_decision_generators()

    def set_decision_seed(self, seed: int) -> None:
        """Give this axis's stochastic decorators their own seeded RNG so a
        re-seed reproduces the exact masks regardless of global RNG. A no-op for
        non-stochastic rates (no ``RandomMask``/``NoisyDropout`` decorators)."""
        self._decision_seed = int(seed)
        for gen in self._decision_generators.values():
            gen.manual_seed(self._decision_seed)
        self._wire_decision_generators()

    def _decision_generator_for(self, device):
        key = str(device)
        gen = self._decision_generators.get(key)
        if gen is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(self._decision_seed)
            self._decision_generators[key] = gen
        return gen

    def _wire_decision_generators(self) -> None:
        if self._decision_seed is None or self._model is None:
            return
        for perceptron in self._model.get_perceptrons():
            try:
                device = next(perceptron.parameters()).device
            except StopIteration:
                device = "cpu"
            gen = self._decision_generator_for(device)
            for obj in _iter_decision_objects(getattr(perceptron, "activation", None)):
                obj._generator = gen

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
