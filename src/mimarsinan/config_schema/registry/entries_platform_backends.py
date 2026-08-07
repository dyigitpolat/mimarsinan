"""Backend-enable derivation helpers for the platform registry entries."""

from __future__ import annotations

from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy


def _backend_supported(cfg: dict, backend: str) -> bool:
    """Mode capability from the ConversionPolicy SSOT (variant-aware)."""
    recipe = ConversionPolicy.derive(
        str(cfg.get("spiking_mode", "lif")), cfg.get("ttfs_cycle_schedule"),
        spiking_variant=cfg.get("spiking_variant"))
    return bool(recipe.sim_enables.get(f"enable_{backend}_simulation", False))


def _why_backend_enable(backend: str, off_reason: str):
    """WHY text for a recipe-defaulted backend enable (user-off aware)."""
    def why(cfg: dict) -> str:
        key = f"enable_{backend}_simulation"
        mode = cfg.get("spiking_mode")
        if cfg.get(key):
            return f"on — ConversionPolicy runs the {backend} gate for {mode!r}"
        if _backend_supported(cfg, backend):
            return f"off — disabled in this config (recipe default for {mode!r}: on)"
        return f"off — {off_reason} (spiking_mode={mode!r})"
    return why


def _meta_backend_enable(backend: str):
    """Machine-readable support flag so the wizard renders toggle vs muted line."""
    def meta(cfg: dict) -> dict:
        return {"supported": _backend_supported(cfg, backend)}
    return meta
