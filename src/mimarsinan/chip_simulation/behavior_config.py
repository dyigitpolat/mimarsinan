"""Simulator-facing neural activation behavior configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from mimarsinan.chip_simulation.firing_strategy import FiringStrategy, FiringStrategyFactory
from mimarsinan.chip_simulation.spiking_semantics import require_spiking_mode_supported
from mimarsinan.models.spiking.spiking_config import SPIKE_MODES


@dataclass(frozen=True)
class NeuralBehaviorConfig:
    spiking_mode: str
    firing_mode: str
    thresholding_mode: str
    spike_generation_mode: str
    spike_encoding_seed: int | None = None

    @classmethod
    def from_deployment_config(cls, cfg: dict[str, Any]) -> NeuralBehaviorConfig:
        return cls(
            spiking_mode=str(cfg.get("spiking_mode", "lif")),
            firing_mode=str(cfg.get("firing_mode", "Default")),
            thresholding_mode=str(cfg.get("thresholding_mode", "<=")),
            spike_generation_mode=str(cfg.get("spike_generation_mode", "Uniform")),
            spike_encoding_seed=cfg.get("spike_encoding_seed"),
        )

    @classmethod
    def for_lava(cls, cfg: dict[str, Any]) -> NeuralBehaviorConfig:
        behavior = cls.from_deployment_config(cfg)
        require_spiking_mode_supported(
            behavior.spiking_mode,
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
            require_spiking_mode_supported(
                self.spiking_mode,
                backend=backend,
                context="NeuralBehaviorConfig",
            )
        self.firing_strategy().require_backend(backend)

    def nevresim_reset_policy(self) -> str:
        return "SubtractiveReset" if self.firing_mode == "Default" else "ZeroReset"

    def nevresim_compare_policy(self) -> str:
        return "StrictCompare" if self.thresholding_mode == "<" else "InclusiveCompare"

    def nevresim_lif_fire_policy(self) -> str:
        reset = self.nevresim_reset_policy()
        compare = self.nevresim_compare_policy()
        return f"LIFirePolicy<{reset}, {compare}>"

    def lava_zero_reset(self) -> bool:
        return self.firing_mode == "Novena"

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
        from mimarsinan.chip_simulation.recording._spike_encoding import encode_segment_input

        return encode_segment_input(
            rates,
            T,
            self.spike_generation_mode,
            seed=self.spike_encoding_seed,
        )

    def nevresim_uses_spike_train_input(self) -> bool:
        return self.spike_generation_mode == "SpikeTrain"
