"""Synthesise a SANA-FE ``Architecture`` from a hybrid mapping."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Optional

from mimarsinan.chip_simulation.sanafe.presets import (
    AXON_IN_NAME, AXON_OUT_NAME, DENDRITE_NAME,
    PerEventEnergy, PRESETS, SOMA_INPUT_RANGE_NAME, SOMA_LIF_NAME,
    SOMA_TTFS_CONTINUOUS_NAME, SOMA_TTFS_QUANTIZED_NAME, SYNAPSE_NAME,
)


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


def _sanafe() -> Any:
    """Lazy ``import sanafe`` (cached; monkey-patched in tests)."""
    global _SANAFE_MODULE
    if _SANAFE_MODULE is None:
        try:
            import sanafe  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover — exercised by integration tests
            raise ImportError(
                "SANA-FE is not installed.  Run scripts/bootstrap_sanafe.sh "
                "(or `pip install sanafe`) to enable the detailed-stats backend."
            ) from e
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

    dendrite_so = _plugin_path("dendrite")
    soma_so = _plugin_path("soma")
    ttfs_cont_so = _plugin_path("ttfs_continuous_soma")
    ttfs_q_so = _plugin_path("ttfs_quantized_soma")
    missing = [
        name for name, path in (
            ("dendrite", dendrite_so),
            ("soma", soma_so),
            ("ttfs_continuous_soma", ttfs_cont_so),
            ("ttfs_quantized_soma", ttfs_q_so),
        )
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "mimarsinan SANA-FE plugins are not built (missing: "
            f"{', '.join(missing)}).  Run ``scripts/bootstrap_sanafe.sh`` "
            "to build all libmimarsinan_*.so artifacts."
        )

    if cores_per_tile <= 0:
        cores_per_tile = max(1, math.isqrt(total_cores))
        if cores_per_tile * cores_per_tile < total_cores:
            cores_per_tile += 1
    n_tiles = (total_cores + cores_per_tile - 1) // cores_per_tile
    n_cores_per_tile = [cores_per_tile] * (n_tiles - 1)
    last = total_cores - cores_per_tile * (n_tiles - 1)
    n_cores_per_tile.append(last)

    mesh_width = max(1, math.isqrt(n_tiles))
    if mesh_width * mesh_width < n_tiles:
        mesh_width += 1
    mesh_height = (n_tiles + mesh_width - 1) // mesh_width

    name = f"mimarsinan_{preset_name}_{total_cores}core"
    return ArchSpec(
        name=name,
        n_tiles=n_tiles,
        n_cores_per_tile=n_cores_per_tile,
        axons_per_core=max_axons,
        neurons_per_core=max_neurons,
        preset=preset,
        dendrite_plugin_path=dendrite_so,
        soma_plugin_path=soma_so,
        ttfs_continuous_plugin_path=ttfs_cont_so,
        ttfs_quantized_plugin_path=ttfs_q_so,
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

