"""Single source of truth for LIF firing-mode semantics across backends."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict

from mimarsinan.chip_simulation.spiking_semantics import (
    backend_capabilities,
    is_declared_backend,
)
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
        """The firing-law columns, DERIVED from the one capability table.

        A backend that runs the LIF family runs both of its reset laws; the
        TTFS column is the genuine cycle-based TTFS capability. An UNDECLARED
        backend supports nothing — the old permissive default let an unknown
        name pass the Default-reset gate silently.
        """
        caps = backend_capabilities(backend)
        declared = is_declared_backend(backend)
        return BackendFiringCapabilities(
            supports_default=declared and caps.lif,
            supports_novena=declared and caps.lif,
            supports_ttfs=declared and caps.ttfs_cycle_based,
        )

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

    def require_chip_faithful_lif_forward(
        self, *, cycle_accurate_lif_forward: bool
    ) -> None:
        """Novena's zero-reset needs the cycle-accurate cascade to stay chip-faithful.

        The analytical rate forward diverges from the deployed HCM under Novena's
        arrival-order-sensitive zero-reset (≈12pp on mmixcore); Default reset is unaffected.
        """
        if self.mode == FiringMode.NOVENA and not cycle_accurate_lif_forward:
            raise ValueError(
                "firing_mode='Novena' requires cycle_accurate_lif_forward=True: "
                "the analytical rate forward is not chip-faithful under Novena's "
                "zero-reset (deployed HCM diverges from the trained metric). Set "
                "cycle_accurate_lif_forward=true (its default for LIF) or use "
                "firing_mode='Default'."
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


def require_chip_faithful_lif_forward(config: Dict[str, Any], spiking_mode: str) -> None:
    """LIF-family gate: a Novena deployment must run the chip-faithful cycle-accurate forward (skipped for TTFS)."""
    from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

    if requires_ttfs_firing(spiking_mode):
        return
    strategy = FiringStrategyFactory.from_config({
        "spiking_mode": spiking_mode,
        "firing_mode": config.get("firing_mode", "Default"),
        "thresholding_mode": config.get("thresholding_mode", "<="),
    })
    strategy.require_chip_faithful_lif_forward(
        cycle_accurate_lif_forward=bool(
            config.get("cycle_accurate_lif_forward", True)
        ),
    )
