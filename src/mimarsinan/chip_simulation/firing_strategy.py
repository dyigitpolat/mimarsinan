"""Single source of truth for LIF firing-mode semantics across backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

from mimarsinan.models.nn.lif_kernels import lif_fire_and_reset


class FiringMode(str, Enum):
    DEFAULT = "Default"
    NOVENA = "Novena"
    TTFS = "TTFS"


_LIF_MODES = {FiringMode.DEFAULT, FiringMode.NOVENA}
_TTFS_MODES = {FiringMode.TTFS}


@dataclass(frozen=True)
class BackendFiringCapabilities:
    supports_default: bool
    supports_novena: bool
    supports_ttfs: bool


@dataclass(frozen=True)
class FiringStrategy:
    mode: FiringMode
    thresholding_mode: str

    def validate_for_spiking_mode(self, spiking_mode: str) -> None:
        from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

        spiking = str(spiking_mode or "lif")
        if requires_ttfs_firing(spiking):
            if self.mode != FiringMode.TTFS:
                raise ValueError(
                    f"spiking_mode={spiking!r} requires firing_mode='TTFS', "
                    f"got {self.mode.value!r}"
                )
        elif self.mode not in _LIF_MODES:
            raise ValueError(
                f"spiking_mode={spiking!r} requires firing_mode in "
                f"{{'Default', 'Novena'}}, got {self.mode.value!r}"
            )
        if self.thresholding_mode not in ("<", "<="):
            raise ValueError(f"Invalid thresholding_mode: {self.thresholding_mode!r}")

    def capabilities(self, backend: str) -> BackendFiringCapabilities:
        b = backend.lower()
        if b in ("hcm", "nevresim", "unified", "hybrid"):
            return BackendFiringCapabilities(True, True, self.mode == FiringMode.TTFS)
        if b in ("training", "lif_activation"):
            return BackendFiringCapabilities(True, True, False)
        if b in ("lava", "loihi"):
            return BackendFiringCapabilities(True, True, False)
        if b in ("sanafe",):
            return BackendFiringCapabilities(True, True, True)
        return BackendFiringCapabilities(True, False, False)

    def require_backend(self, backend: str) -> None:
        caps = self.capabilities(backend)
        if self.mode == FiringMode.NOVENA and not caps.supports_novena:
            raise ValueError(
                f"Backend {backend!r} does not support firing_mode='Novena'"
            )
        if self.mode == FiringMode.DEFAULT and not caps.supports_default:
            raise ValueError(
                f"Backend {backend!r} does not support firing_mode='Default'"
            )

    def hcm_reset_step(self) -> Callable:
        mode = self.mode.value

        def _step(memb, threshold, *, thresholding_mode: str, output_dtype=None):
            return lif_fire_and_reset(
                memb,
                threshold,
                thresholding_mode=thresholding_mode,
                firing_mode=mode,
                output_dtype=output_dtype,
            )

        return _step

    def nevresim_policy_suffix(self) -> str:
        if self.mode == FiringMode.TTFS:
            raise ValueError("nevresim_policy_suffix not used for TTFS spiking_mode")
        return self.mode.value

    def training_lif_v_reset(self) -> float | None:
        if self.mode == FiringMode.NOVENA:
            return 0.0
        return None

    def sanafe_reset_mode(self) -> str:
        return "hard" if self.mode == FiringMode.NOVENA else "soft"


class FiringStrategyFactory:
    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> FiringStrategy:
        mode_str = str(cfg.get("firing_mode", "Default"))
        try:
            mode = FiringMode(mode_str)
        except ValueError as exc:
            raise ValueError(f"Invalid firing_mode: {mode_str!r}") from exc
        strategy = FiringStrategy(
            mode=mode,
            thresholding_mode=str(cfg.get("thresholding_mode", "<=")),
        )
        strategy.validate_for_spiking_mode(str(cfg.get("spiking_mode", "lif")))
        return strategy
