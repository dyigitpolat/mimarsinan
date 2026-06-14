"""The ``AdaptationAxis`` contract — a rate-driven, orchestration-facing axis.

An ``AdaptationAxis`` is the homotopy α-axis the tuner walks from 0 (original
behavior) to 1 (full transform). It is the spec's "Transformation" contract
narrowed to this codebase, renamed to avoid colliding with the existing
``mimarsinan.transformations`` package (stateless transform *math*). Each axis
*delegates* its math to ``transformations/`` and its rate application to the
``perceptron_rate`` SSOT — the axis owns only the control-facing seam
(attach / set_rate / calibrate / recovery / finalize / state / descriptor).

Axes are transient per-run objects built from config + the adaptation manager;
they are never stored on the model or the pickled ``adaptation_manager`` cache.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

import torch.nn as nn


@runtime_checkable
class AdaptationAxis(Protocol):
    """Control-facing axis contract (see :class:`AdaptationAxisBase` for defaults)."""

    name: str
    interpolation_mode: str  # functional_blend | parameter_path | stochastic_mask
    monotonicity: str        # guaranteed | expected | none
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

    ``attach`` records the (model, manager, config) the rate application needs;
    every other method has a benign default so an adapter only implements the
    behavior that is genuinely axis-specific.
    """

    name: str = "axis"
    interpolation_mode: str = "parameter_path"
    monotonicity: str = "expected"
    is_stochastic: bool = False
    supports_smooth: bool = True

    def __init__(self) -> None:
        self._model = None
        self._manager = None
        self._config = None

    def attach(self, model, adaptation_manager, config) -> None:
        """Record the application context. Idempotent: re-attach overwrites refs."""
        self._model = model
        self._manager = adaptation_manager
        self._config = config

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

    def get_extra_state(self):
        return None

    def set_extra_state(self, extra) -> None:
        return None

    def set_decision_seed(self, seed: int) -> None:
        return None

    def descriptor(self) -> str:
        return self.name
