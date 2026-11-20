"""Simulator-facing neural activation behavior configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from mimarsinan.chip_simulation.firing_strategy import FiringStrategy, FiringStrategyFactory
from mimarsinan.chip_simulation.nevresim_policy_types import (
    nevresim_compare_policy,
    nevresim_lif_fire_policy,
    nevresim_reset_policy,
)
from mimarsinan.chip_simulation.recording._spike_encoding import encode_segment_input
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.chip_simulation.spiking_point import (
    HISTORICAL_WINDOW_FIRING,
    SpikingPoint,
    resolve_spiking_point,
    stated_point,
)
from mimarsinan.chip_simulation.spiking_semantics import is_novena_firing_mode
from mimarsinan.models.spiking.spiking_config import SPIKE_MODES


@dataclass(frozen=True)
class NeuralBehaviorConfig:
    spiking_mode: str
    firing_mode: str
    thresholding_mode: str
    spike_generation_mode: str
    spike_encoding_seed: int | None = None
    #: [U-LIF2 W5] The run's resolved point when built from a config (the runners' checks read it); outside equality.
    point: Optional[SpikingPoint] = field(default=None, compare=False, repr=False)

    @classmethod
    def from_deployment_config(cls, cfg: dict[str, Any]) -> NeuralBehaviorConfig:
        return cls(
            spiking_mode=str(cfg.get("spiking_mode", "lif")),
            # An absent firing_mode is the soma law's question, not a literal:
            # a per_event document resolves to the hard-zero reset it requires.
            firing_mode=str(cfg.get("firing_mode") or SomaLaw.resolve(cfg).firing_mode),
            thresholding_mode=str(cfg.get("thresholding_mode", "<=")),
            spike_generation_mode=str(cfg.get("spike_generation_mode", "Uniform")),
            spike_encoding_seed=cfg.get("spike_encoding_seed"),
            point=resolve_spiking_point(cfg),
        )

    def spiking_point(self, schedule: Any = None) -> SpikingPoint:
        """The resolved point this behavior runs: the config's, else [U-LIF2 r2] the point its own fields state (a
        behavior built directly, or stored before the point existed), with the historical window firing, never the
        declared default."""
        if self.point is not None:
            return self.point
        return stated_point({"spiking_mode": self.spiking_mode, "ttfs_cycle_schedule": schedule,
                             "firing_mode": self.firing_mode, "thresholding_mode": self.thresholding_mode},
                            window_firing=HISTORICAL_WINDOW_FIRING)

    def _mode_policy(self):
        return policy_for_spiking_mode(self.spiking_point())

    @classmethod
    def for_lava(cls, cfg: dict[str, Any]) -> NeuralBehaviorConfig:
        behavior = cls.from_deployment_config(cfg)
        behavior._mode_policy().require_backend_supported(
            backend="lava",
            context="NeuralBehaviorConfig.for_lava",
        )
        return behavior

    def firing_strategy(self) -> FiringStrategy:
        return FiringStrategyFactory.from_config(
            {
                "spiking_mode": self.spiking_mode,
                "firing_mode": self.firing_mode,
                "thresholding_mode": self.thresholding_mode,
            }
        )

    def require_backend(self, backend: str) -> None:
        if backend.lower() in ("lava", "loihi"):
            self._mode_policy().require_backend_supported(
                backend=backend,
                context="NeuralBehaviorConfig",
            )
        self.firing_strategy().require_backend(backend)

    def nevresim_reset_policy(self) -> str:
        return nevresim_reset_policy(self.firing_mode)

    def nevresim_compare_policy(self) -> str:
        return nevresim_compare_policy(self.thresholding_mode)

    def nevresim_lif_fire_policy(self) -> str:
        return nevresim_lif_fire_policy(self.firing_mode, self.thresholding_mode)

    def lava_zero_reset(self) -> bool:
        return is_novena_firing_mode(self.firing_mode)

    def sanafe_reset_mode(self) -> str:
        return self.firing_strategy().sanafe_reset_mode()

    def encode_segment_input(self, rates: np.ndarray, T: int) -> np.ndarray:
        if self.spike_generation_mode == "TTFS":
            raise ValueError(
                "encode_segment_input does not support spike_generation_mode='TTFS'; "
                "use the TTFS encoding path"
            )
        if self.spike_generation_mode not in SPIKE_MODES:
            raise ValueError(f"Invalid spike_generation_mode: {self.spike_generation_mode!r}")
        return encode_segment_input(
            rates,
            T,
            self.spike_generation_mode,
            seed=self.spike_encoding_seed,
        )

    def nevresim_uses_spike_train_input(self) -> bool:
        return self.spike_generation_mode == "SpikeTrain"
