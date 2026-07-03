"""Synthesise a SANA-FE ``Architecture`` from a hybrid mapping."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from mimarsinan.chip_simulation.sanafe.presets import PerEventEnergy, PRESETS


def _mesh_dims(n_tiles: int) -> tuple[int, int]:
    """Most-square exact factorization ``(width>=height, width*height==n_tiles)``.

    Must be exact: a ceil-padded mesh leaves phantom tiles the YAML never defines
    and SANA-FE's C++ NoC then SIGFPEs indexing them.
    """
    n = max(1, int(n_tiles))
    height = 1
    for h in range(int(math.isqrt(n)), 0, -1):
        if n % h == 0:
            height = h
            break
    return n // height, height


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

    @property
    def total_cores(self) -> int:
        return sum(self.n_cores_per_tile)


def derive_arch_spec(
    mapping: Any,
    *,
    preset_name: str,
    cores_per_tile: int = 0,
) -> ArchSpec:
    """Walk every neural segment of ``mapping`` and produce an ArchSpec."""
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown SANA-FE arch preset {preset_name!r}; "
            f"expected one of {sorted(PRESETS.keys())}"
        )
    preset = PRESETS[preset_name]

    segments = list(mapping.get_neural_segments())
    if not segments:
        raise ValueError(
            "no neural segments in the mapping; SANA-FE has nothing to simulate"
        )

    total_cores = 0
    max_axons = 0
    max_neurons = 0
    for seg in segments:
        for core in seg.cores:
            total_cores += 1
            ax = int(core.axons_per_core)
            ne = int(core.neurons_per_core)
            if ax > max_axons:
                max_axons = ax
            if ne > max_neurons:
                max_neurons = ne

    if total_cores == 0:
        raise ValueError(
            "no neural cores in the mapping's segments; SANA-FE has nothing to simulate"
        )

    plugin_names = (
        "dendrite",
        "soma",
        "ttfs_continuous_soma",
        "ttfs_quantized_soma",
        "ttfs_cycle_soma",
        "ttfs_cascade_soma",
    )
    candidates = {name: _plugin_path(name) for name in plugin_names}
    missing = [name for name, path in candidates.items() if path is None]
    if missing:
        raise FileNotFoundError(
            "mimarsinan SANA-FE plugins are not built (missing: "
            f"{', '.join(missing)}).  Run ``scripts/bootstrap_sanafe.sh`` "
            "to build all libmimarsinan_*.so artifacts."
        )
    plugins = {name: path for name, path in candidates.items() if path is not None}

    if cores_per_tile <= 0:
        cores_per_tile = max(1, math.isqrt(total_cores))
        if cores_per_tile * cores_per_tile < total_cores:
            cores_per_tile += 1
    n_tiles = (total_cores + cores_per_tile - 1) // cores_per_tile
    n_cores_per_tile = [cores_per_tile] * (n_tiles - 1)
    last = total_cores - cores_per_tile * (n_tiles - 1)
    n_cores_per_tile.append(last)

    mesh_width, mesh_height = _mesh_dims(n_tiles)

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
    )


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

