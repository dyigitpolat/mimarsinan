"""Backend-enable derivation helpers for the platform registry entries."""

from __future__ import annotations

from typing import Optional, Tuple

from mimarsinan.chip_simulation.point_capability import SIM_ENABLE_BACKENDS
from mimarsinan.chip_simulation.spiking_point import POINT_BASIS_KEYS, resolve_spiking_point
from mimarsinan.tuning.orchestration.conversion_policy import DEVICE_OPT_IN_ENABLES, ConversionPolicy

#: [U-LIF2 W4] What every backend enable derives from: the keys of the resolved point its capability judgment reads.
ENABLE_BASIS: Tuple[str, ...] = POINT_BASIS_KEYS


def _enable_refusal(cfg: dict, key: str) -> Tuple[bool, Optional[str]]:
    """[U-LIF2 W5] Whether the recipe of the config's resolved point admits ``key``, and the value of the point its
    backend does not execute when it does not (the recipe's own reason; each entry reads its own key)."""
    recipe = ConversionPolicy.derive(resolve_spiking_point(cfg))
    return bool(recipe.sim_enables.get(key, False)), recipe.sim_off_reasons.get(key)


def _why_backend_enable(key: str, off_reason: str):
    """WHY text for a recipe-defaulted backend enable (user-off and opt-in aware): a mode refusal keeps its
    backend's own reason, every other refusal names the value of the point the backend does not execute."""
    backend = SIM_ENABLE_BACKENDS[key]

    def why(cfg: dict) -> str:
        mode = cfg.get("spiking_mode")
        if cfg.get(key):
            return f"on — ConversionPolicy runs the {backend} gate for {mode!r}"
        supported, refusal = _enable_refusal(cfg, key)
        if supported:
            if key in DEVICE_OPT_IN_ENABLES:
                return "off (opt-in): the capability admits it, and it runs only when the document declares it"
            return f"off — disabled in this config (recipe default for {mode!r}: on)"
        if refusal is None or refusal.startswith("spiking_mode=") or refusal.startswith("spiking_variant="):
            return f"off — {off_reason} (spiking_mode={mode!r})"
        return f"off: the backend does not execute {refusal}"
    return why


def _meta_backend_enable(key: str):
    """Machine-readable support flag so the wizard renders toggle vs muted line."""
    def meta(cfg: dict) -> dict:
        return {"supported": _enable_refusal(cfg, key)[0]}
    return meta
