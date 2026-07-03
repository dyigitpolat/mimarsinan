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
