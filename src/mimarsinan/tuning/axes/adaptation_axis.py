"""The ``AdaptationAxis`` contract — a rate-driven, orchestration-facing axis."""

from __future__ import annotations

import copy
from typing import Any, Iterable, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class AdaptationAxis(Protocol):
    """Control-facing axis contract (see :class:`AdaptationAxisBase` for defaults)."""

    name: str
    interpolation_mode: str
    monotonicity: str
    is_stochastic: bool
    supports_smooth: bool

    def attach(self, model, adaptation_manager, config) -> None: ...
    def set_rate(self, alpha: float) -> None: ...
    def calibrate(self, model, batches) -> None: ...
    def tunable_parameters(self) -> Iterable[nn.Parameter]: ...
    def recovery_hooks(self, alpha: float) -> list: ...
    def finalize(self, model) -> None: ...
    def get_extra_state(self): ...
    def set_extra_state(self, extra) -> None: ...
    def set_decision_seed(self, seed: int) -> None: ...
    def descriptor(self) -> str: ...


class AdaptationAxisBase:
    """Default ``AdaptationAxis`` implementation; subclasses override ``set_rate``.

    ``attach`` records the (model, manager, config); every other method has a benign
    default so an adapter only implements what is genuinely axis-specific.
    """

    name: str = "axis"
    interpolation_mode: str = "parameter_path"
    monotonicity: str = "expected"
    is_stochastic: bool = False
    supports_smooth: bool = True

    def __init__(self) -> None:
        # Duck-typed attach context: models/managers/configs vary per pipeline.
        self._model: Any = None
        self._manager: Any = None
        self._config: Any = None

    def attach(self, model, adaptation_manager, config) -> None:
        """Record the application context. Idempotent: re-attach overwrites refs."""
        self._model = model
        self._manager = adaptation_manager
        self._config = config

    def probe_replica(self, model, adaptation_manager, config) -> "AdaptationAxisBase":
        """A detached copy of this axis attached to isolated probe targets.

        The replica must share no mutable state with the live axis, so driving it
        can never perturb the live ramp (``_reset_replica_state`` drops owned caches).
        """
        replica = copy.copy(self)
        replica._reset_replica_state()
        replica.attach(model, adaptation_manager, config)
        return replica

    def _reset_replica_state(self) -> None:
        """Drop any mutable caches a shallow copy would share with the live axis."""

    def set_rate(self, alpha: float) -> None:
        raise NotImplementedError

    def calibrate(self, model, batches) -> None:
        return None

    def tunable_parameters(self) -> Iterable[nn.Parameter]:
        return ()

    def recovery_hooks(self, alpha: float) -> list:
        return []

    def finalize(self, model) -> None:
        return None

    def get_extra_state(self) -> Any:
        return None

    def set_extra_state(self, extra) -> None:
        return None

    def set_decision_seed(self, seed: int) -> None:
        return None

    def descriptor(self) -> str:
        return self.name


class ClosureApplyAxisBase(AdaptationAxisBase):
    """Base for axes driving a live-bound apply closure (NAPQ / pruning / shift).

    The live closure targets the owning tuner's model and trainer, so a probe
    replica must NEVER fire it: replicas dispatch to the model-targeted
    ``replica_apply_fn(model, rate)`` against the attach target, and fail loud
    when the tuner provides none.
    """

    def __init__(self, apply_fn, *, replica_apply_fn=None):
        super().__init__()
        self._apply_fn = apply_fn
        self._replica_apply_fn = replica_apply_fn
        self._is_probe_replica = False

    def _reset_replica_state(self) -> None:
        self._is_probe_replica = True

    def set_rate(self, alpha: float) -> None:
        alpha = float(alpha)
        if not self._is_probe_replica:
            self._apply_fn(alpha)
            return
        if self._replica_apply_fn is None:
            raise RuntimeError(
                f"{type(self).__name__} has no replica_apply_fn: its live-bound "
                "apply closure would mutate the LIVE model from a probe replica. "
                "Pass a model-targeted replica_apply_fn(model, rate) at "
                "construction to make this axis probe-safe."
            )
        self._replica_apply_fn(self._model, alpha)
