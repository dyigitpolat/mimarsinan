"""Synthesise a SANA-FE ``Architecture`` from a hybrid mapping."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from mimarsinan.chip_simulation.sanafe.arch_synth.floorplan import (
    _mesh_dims,
    resolve_floorplan,
)
from mimarsinan.chip_simulation.sanafe.presets import (
    CUSTOM_ZERO_PRESET,
    PerEventEnergy,
    PRESETS,
)

CUSTOM_PRESET_NAME = "custom"


def _plugin_path(name: str) -> Optional[str]:
    """Absolute path to ``build/mimarsinan_sanafe_plugins/libmimarsinan_<name>.so``."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))
    candidate = os.path.join(
        project_root, "build", "mimarsinan_sanafe_plugins",
        f"libmimarsinan_{name}.so",
    )
    return candidate if os.path.isfile(candidate) else None


_SANAFE_MODULE: Any = None

# The integration targets these SANA-FE versions; 2.2.x SIGFPEs on arch load.
_SUPPORTED_SANAFE_VERSIONS = ("2.1.1",)


def _check_sanafe_version(version: Optional[str]) -> None:
    """Fail loud on an unsupported SANA-FE rather than let it SIGFPE in C++.

    ``None`` is permissive: only versions known incompatible are blocked.
    """
    if version is not None and version not in _SUPPORTED_SANAFE_VERSIONS:
        supported = _SUPPORTED_SANAFE_VERSIONS[0]
        raise RuntimeError(
            f"SANA-FE {version} is unsupported — the mimarsinan integration "
            f"targets {supported} (2.2.x SIGFPEs on arch load). Pin it: "
            f"`pip install sanafe=={supported}` or re-run "
            f"scripts/bootstrap_sanafe.sh (now pinned)."
        )


def _sanafe() -> Any:
    """Lazy ``import sanafe`` (cached; monkey-patched in tests), version-guarded."""
    global _SANAFE_MODULE
    if _SANAFE_MODULE is None:
        try:
            import sanafe  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "SANA-FE is not installed.  Run scripts/bootstrap_sanafe.sh "
                "to enable the detailed-stats backend."
            ) from e
        import importlib.metadata as _md

        try:
            _version = _md.version("sanafe")
        except _md.PackageNotFoundError:
            _version = getattr(sanafe, "__version__", None)
        _check_sanafe_version(_version)
        _SANAFE_MODULE = sanafe
    return _SANAFE_MODULE


@dataclass(frozen=True)
class ArchSpec:
    """Geometry and preset for one SANA-FE architecture instance."""

    name: str
    n_tiles: int
    n_cores_per_tile: List[int]
    axons_per_core: int
    neurons_per_core: int
    preset: PerEventEnergy = field(repr=False)
    dendrite_plugin_path: str = field(default="")
    soma_plugin_path: str = field(default="")
    ttfs_continuous_plugin_path: str = field(default="")
    ttfs_quantized_plugin_path: str = field(default="")
    ttfs_cycle_plugin_path: str = field(default="")
    ttfs_cascade_plugin_path: str = field(default="")
    mesh_width: int = 1
    mesh_height: int = 1
    cores_per_tile_resolved: int = 1
    # The mapping's actual core count (<= total_cores when idle slots exist).
    packed_cores: int = 0
    # >1 for scheduled mappings: the physical floorplan row-stacked k times so
    # every logical (per-pass) core materializes; passes reuse cores serially
    # on the real chip.
    floorplan_replicas: int = 1

    @property
    def total_cores(self) -> int:
        return sum(self.n_cores_per_tile)


def _resolve_preset(
    preset_name: str, custom_arch_path: Optional[str]
) -> PerEventEnergy:
    """Per-event energy table for ``preset_name``; validates ``custom``."""
    if preset_name == CUSTOM_PRESET_NAME:
        if not custom_arch_path:
            raise ValueError(
                "sanafe_arch_preset='custom' requires sanafe_custom_arch_path "
                "(the user architecture YAML to load in place of synthesis)"
            )
        return CUSTOM_ZERO_PRESET
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown SANA-FE arch preset {preset_name!r}; "
            f"expected one of {sorted(PRESETS.keys()) + [CUSTOM_PRESET_NAME]}"
        )
    return PRESETS[preset_name]


def _resolve_plugins(preset_name: str) -> dict[str, str]:
    """Built mimarsinan plugin paths; ``custom`` YAMLs reference their own."""
    plugin_names = (
        "dendrite",
        "soma",
        "ttfs_continuous_soma",
        "ttfs_quantized_soma",
        "ttfs_cycle_soma",
        "ttfs_cascade_soma",
    )
    if preset_name == CUSTOM_PRESET_NAME:
        return {name: "" for name in plugin_names}
    candidates = {name: _plugin_path(name) for name in plugin_names}
    missing = [name for name, path in candidates.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "mimarsinan SANA-FE plugins are not built (missing: "
            f"{', '.join(missing)}).  Run ``scripts/bootstrap_sanafe.sh`` "
            "to build all libmimarsinan_*.so artifacts."
        )
    return {name: path for name, path in candidates.items() if path is not None}


def derive_arch_spec(
    mapping: Any,
    *,
    preset_name: str,
    cores_per_tile: int = 0,
    tile_grid_rows: int = 0,
    tile_grid_cols: int = 0,
    declared_core_capacity: int = 0,
    custom_arch_path: Optional[str] = None,
) -> ArchSpec:
    """Walk every neural segment of ``mapping`` and produce an ArchSpec.

    With ``declared_core_capacity`` > 0 the floorplan is FIXED by the declared
    platform (``resolve_floorplan``): every tile carries the full complement,
    idle slots are defined (not phantom), and the floorplan is independent of
    the packed core count. With 0 (a runner outside a pipeline, no declared
    platform) the legacy packed-count derivation applies — unless an explicit
    tile grid is declared, which is honored as a full floorplan over the
    packed cores (never silently ignored).
    """
    preset = _resolve_preset(preset_name, custom_arch_path)

    segments = list(mapping.get_neural_segments())
    if not segments:
        raise ValueError(
            "no neural segments in the mapping; SANA-FE has nothing to simulate"
        )

    packed_cores = 0
    max_axons = 0
    max_neurons = 0
    for seg in segments:
        for core in seg.cores:
            packed_cores += 1
            ax = int(core.axons_per_core)
            ne = int(core.neurons_per_core)
            if ax > max_axons:
                max_axons = ax
            if ne > max_neurons:
                max_neurons = ne

    if packed_cores == 0:
        raise ValueError(
            "no neural cores in the mapping's segments; SANA-FE has nothing to simulate"
        )

    plugins = _resolve_plugins(preset_name)

    effective_capacity = int(declared_core_capacity)
    if effective_capacity <= 0 and (tile_grid_rows or tile_grid_cols):
        # An explicit grid is a full-floorplan declaration: honor it over the
        # packed cores rather than silently ignore it.
        effective_capacity = packed_cores
    replicas = 1
    if effective_capacity > 0:
        # The physical invariant is PER-PASS: a multi-pass schedule packs more
        # logical cores than the chip by design (passes reuse cores serially).
        max_pass_cores = _max_cores_per_pass(mapping, packed_cores)
        if max_pass_cores > effective_capacity:
            raise ValueError(
                f"a single schedule pass needs {max_pass_cores} cores but the "
                f"declared platform capacity is {effective_capacity} — "
                "the mapping cannot be placed on the declared floorplan"
            )
        cores_per_tile, rows, cols = resolve_floorplan(
            effective_capacity, preset_name,
            cores_per_tile, tile_grid_rows, tile_grid_cols,
        )
        slots = rows * cols * cores_per_tile
        replicas = max(1, -(-packed_cores // slots))
        rows *= replicas
        n_tiles = rows * cols
        n_cores_per_tile = [cores_per_tile] * n_tiles
        mesh_width, mesh_height = cols, rows
    else:
        if cores_per_tile <= 0:
            cores_per_tile = max(1, math.isqrt(packed_cores))
            if cores_per_tile * cores_per_tile < packed_cores:
                cores_per_tile += 1
        n_tiles = (packed_cores + cores_per_tile - 1) // cores_per_tile
        n_cores_per_tile = [cores_per_tile] * (n_tiles - 1)
        last = packed_cores - cores_per_tile * (n_tiles - 1)
        n_cores_per_tile.append(last)
        mesh_width, mesh_height = _mesh_dims(n_tiles)

    total_cores = sum(n_cores_per_tile)
    name = f"mimarsinan_{preset_name}_{total_cores}core"
    return ArchSpec(
        name=name,
        n_tiles=n_tiles,
        n_cores_per_tile=n_cores_per_tile,
        axons_per_core=max_axons,
        neurons_per_core=max_neurons,
        preset=preset,
        dendrite_plugin_path=plugins["dendrite"],
        soma_plugin_path=plugins["soma"],
        ttfs_continuous_plugin_path=plugins["ttfs_continuous_soma"],
        ttfs_quantized_plugin_path=plugins["ttfs_quantized_soma"],
        ttfs_cycle_plugin_path=plugins["ttfs_cycle_soma"],
        ttfs_cascade_plugin_path=plugins["ttfs_cascade_soma"],
        mesh_width=mesh_width,
        mesh_height=mesh_height,
        cores_per_tile_resolved=cores_per_tile,
        packed_cores=packed_cores,
        floorplan_replicas=replicas,
    )


def _max_cores_per_pass(mapping: Any, packed_cores: int) -> int:
    """Largest simultaneous core need: the widest single stage. Stages execute
    serially (segments AND passes; cores are freed/reprogrammed between them),
    so per-stage residency is the physical constraint — never a cross-stage
    sum. Falls back to the whole packed count without stage structure."""
    stages = getattr(mapping, "stages", None)
    if not stages:
        return packed_cores
    widths = [
        len(stage.hard_core_mapping.cores)
        for stage in stages
        if getattr(stage, "kind", None) == "neural"
        and getattr(stage, "hard_core_mapping", None) is not None
    ]
    return max(widths) if widths else packed_cores


def _thresholding_mode_to_soma_attr(thresholding_mode: str) -> str:
    """Map pipeline ``thresholding_mode`` to soma plugin ``inclusive``/``strict``."""
    if thresholding_mode in ("<=", "inclusive"):
        return "inclusive"
    if thresholding_mode in ("<", "strict"):
        return "strict"
    raise ValueError(
        f"unsupported thresholding_mode {thresholding_mode!r}; expected "
        "one of ('<', '<=', 'strict', 'inclusive')"
    )

