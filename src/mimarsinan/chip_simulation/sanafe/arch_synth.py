"""Synthesize a SANA-FE ``Architecture`` from a mimarsinan ``HybridHardCoreMapping``.

Two-stage pipeline:

1. :func:`derive_arch_spec` — pure-Python.  Walks all neural segments of
   the hybrid mapping, computes ``axons_per_core``, ``neurons_per_core``,
   total core count, and packs cores into tiles (default: one tile per
   segment-set; configurable via ``cores_per_tile``).  Returns an
   :class:`ArchSpec` carrying the geometry plus the chosen per-event
   energy/latency preset.

2. :func:`build_architecture` — touches SANA-FE.  Either constructs the
   architecture programmatically (one ``Architecture(name)`` + ``create_tile``
   + ``create_core`` per spec entry) or loads a custom YAML via
   ``sanafe.load_arch``.  Validates that the loaded YAML has at least as
   many cores as the spec demands.

The lazy ``_sanafe()`` accessor is the single import point; tests
monkey-patch it to keep the suite runnable without SANA-FE installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .presets import PerEventEnergy, PRESETS


# ---------------------------------------------------------------------------
# Lazy SANA-FE accessor — overridden in tests via monkeypatch
# ---------------------------------------------------------------------------


_SANAFE_MODULE: Any = None


def _sanafe() -> Any:
    """Lazy ``import sanafe``.

    Raises a clear ``ImportError`` with installation instructions if the
    package is missing.  Cached after the first successful import.
    """
    global _SANAFE_MODULE
    if _SANAFE_MODULE is None:
        try:
            import sanafe  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover — exercised by integration tests
            raise ImportError(
                "SANA-FE is not installed.  Run scripts/bootstrap_sanafe.sh "
                "(which calls `pip install -e ./sana_fe`) to enable the "
                "detailed-stats backend."
            ) from e
        _SANAFE_MODULE = sanafe
    return _SANAFE_MODULE


# ---------------------------------------------------------------------------
# ArchSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchSpec:
    """Geometry + preset describing a SANA-FE architecture to instantiate."""

    name: str
    n_tiles: int
    n_cores_per_tile: List[int]
    axons_per_core: int
    neurons_per_core: int
    preset: PerEventEnergy = field(repr=False)

    @property
    def total_cores(self) -> int:
        return sum(self.n_cores_per_tile)


# ---------------------------------------------------------------------------
# derive_arch_spec — pure Python
# ---------------------------------------------------------------------------


def derive_arch_spec(
    mapping: Any,
    *,
    preset_name: str,
    cores_per_tile: int = 0,
) -> ArchSpec:
    """Walk every neural segment of ``mapping`` and produce an ArchSpec.

    Parameters
    ----------
    mapping
        A ``HybridHardCoreMapping``-shaped object (tests pass a fake with
        the same attribute surface).
    preset_name
        Key into :data:`PRESETS`.  Raises ``ValueError`` if unknown.
    cores_per_tile
        ``0`` (default): pack all cores into one tile.  ``k > 0``: split
        cores into ``ceil(total / k)`` tiles, each holding up to ``k``
        cores.  The last tile may be smaller.
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown SANA-FE arch preset {preset_name!r}; "
            f"expected one of {sorted(PRESETS.keys())}"
        )
    preset = PRESETS[preset_name]

    # Collect all neural segments and walk their HardCores.
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

    # Pack cores into tiles.
    if cores_per_tile <= 0:
        n_tiles = 1
        n_cores_per_tile = [total_cores]
    else:
        n_tiles = (total_cores + cores_per_tile - 1) // cores_per_tile
        n_cores_per_tile = [cores_per_tile] * (n_tiles - 1)
        last = total_cores - cores_per_tile * (n_tiles - 1)
        n_cores_per_tile.append(last)

    name = f"mimarsinan_{preset_name}_{total_cores}core"
    return ArchSpec(
        name=name,
        n_tiles=n_tiles,
        n_cores_per_tile=n_cores_per_tile,
        axons_per_core=max_axons,
        neurons_per_core=max_neurons,
        preset=preset,
    )


# ---------------------------------------------------------------------------
# build_architecture — touches SANA-FE via _sanafe()
# ---------------------------------------------------------------------------


def _per_core_attrs(spec: ArchSpec) -> dict:
    """Per-core attribute kwargs forwarded to ``arch.create_core``.

    Names match the SANA-FE arch YAML attribute namespace so SANA-FE's
    ``create_core`` accepts them directly.  Per-event numbers come from
    the spec's preset, never from local literals.
    """
    p = spec.preset
    return {
        "max_neurons": spec.neurons_per_core,
        "max_axons": spec.axons_per_core,
        "synapse_energy_j":  p["synapse_energy_j"],
        "dendrite_energy_j": p["dendrite_energy_j"],
        "soma_energy_j":     p["soma_energy_j"],
        "network_energy_j":  p["network_energy_j"],
        "synapse_latency_s": p["synapse_latency_s"],
        "soma_latency_s":    p["soma_latency_s"],
        "network_latency_s": p["network_latency_s"],
    }


def build_architecture(
    spec: ArchSpec,
    *,
    custom_arch_path: Optional[str] = None,
) -> Any:
    """Construct (or load) a SANA-FE Architecture matching ``spec``.

    If ``custom_arch_path`` is provided, ``sanafe.load_arch`` is used
    instead of the programmatic builder.  The loaded architecture is
    validated against the spec's total core count.
    """
    sanafe = _sanafe()

    if custom_arch_path is not None:
        if not os.path.isfile(custom_arch_path):
            raise FileNotFoundError(
                f"SANA-FE custom arch YAML not found: {custom_arch_path}"
            )
        arch = sanafe.load_arch(custom_arch_path)
        # Validate: the loaded arch must have at least as many cores as we need.
        loaded_cores = sum(len(tile.cores) for tile in arch.tiles)
        if loaded_cores < spec.total_cores:
            raise ValueError(
                f"custom arch at {custom_arch_path} provides only "
                f"{loaded_cores} cores but the mapping needs {spec.total_cores}"
            )
        return arch

    arch = sanafe.Architecture(spec.name)
    core_attrs = _per_core_attrs(spec)
    for tile_idx, n_cores in enumerate(spec.n_cores_per_tile):
        tile = arch.create_tile(name=f"tile_{tile_idx}")
        for core_idx in range(n_cores):
            arch.create_core(
                parent_tile=tile,
                name=f"tile_{tile_idx}_core_{core_idx}",
                **core_attrs,
            )
    return arch
