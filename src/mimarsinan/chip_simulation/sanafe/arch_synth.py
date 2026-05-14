"""Synthesise a SANA-FE ``Architecture`` from a mimarsinan ``HybridHardCoreMapping``.

Two-stage pipeline:

1. :func:`derive_arch_spec` — pure-Python.  Walks all neural segments of
   the hybrid mapping, computes ``axons_per_core`` / ``neurons_per_core`` /
   total core count, and packs cores into tiles (default: one tile per
   segment-set; configurable via ``cores_per_tile``).  Returns an
   :class:`ArchSpec` carrying the geometry plus the chosen per-event
   preset.

2. :func:`build_architecture` — touches SANA-FE.  Either *synthesises* a
   YAML matching the spec and calls ``sanafe.load_arch``, or — when
   ``custom_arch_path`` is given — loads a user-supplied YAML directly.
   SANA-FE's hardware-unit registration (synapse / dendrite / soma) is
   YAML-only; the programmatic ``Architecture.create_core`` cannot reach
   it, so YAML synthesis is the canonical path.

The lazy ``_sanafe()`` accessor is the single import point; tests
monkey-patch it to keep the suite runnable without SANA-FE installed.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, List, Optional

from .presets import (
    AXON_IN_NAME, AXON_OUT_NAME, DENDRITE_NAME,
    PerEventEnergy, PRESETS, SOMA_INPUT_RANGE_NAME, SOMA_LIF_NAME, SYNAPSE_NAME,
)


# ---------------------------------------------------------------------------
# Plugin discovery
# ---------------------------------------------------------------------------


def _plugin_path(name: str) -> Optional[str]:
    """Return the absolute path to one of our compiled SANA-FE plugins.

    Looks for ``build/mimarsinan_sanafe_plugins/libmimarsinan_<name>.so``
    next to the mimarsinan project root.  Returns ``None`` if the plugin
    hasn't been built — callers should error or skip with a clear message
    pointing at ``scripts/bootstrap_sanafe.sh``.
    """
    # arch_synth.py lives at .../src/mimarsinan/chip_simulation/sanafe/.
    # Four ``..`` hops bring us back to the project root.
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
    candidate = os.path.join(
        project_root, "build", "mimarsinan_sanafe_plugins",
        f"libmimarsinan_{name}.so",
    )
    return candidate if os.path.isfile(candidate) else None


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
                "(or `pip install sanafe`) to enable the detailed-stats backend."
            ) from e
        _SANAFE_MODULE = sanafe
    return _SANAFE_MODULE


# ---------------------------------------------------------------------------
# ArchSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchSpec:
    """Geometry + preset describing a SANA-FE architecture to instantiate.

    Every SANA-FE core in the arch corresponds 1:1 to a ``HardCore`` in
    mimarsinan's HybridHardCoreMapping (in declaration order).  Each
    core declares its own ``inputs[0..axons_per_core-1]`` soma pool so
    input neurons live on the same core as the axons that consume them.

    The dendrite and LIF soma reference the mimarsinan-owned plugin
    shared libraries (``libmimarsinan_dendrite.so``,
    ``libmimarsinan_soma.so``) so the core has no Loihi-derived
    per-core neuron count cap.
    """

    name: str
    n_tiles: int
    n_cores_per_tile: List[int]
    axons_per_core: int
    neurons_per_core: int
    preset: PerEventEnergy = field(repr=False)
    # Absolute paths to the compiled mimarsinan plugins.  Resolved at
    # spec-construction time so failures surface early.
    dendrite_plugin_path: str = field(default="")
    soma_plugin_path: str = field(default="")
    # 2D NoC mesh layout — chip is ``mesh_width × mesh_height`` tiles.
    # ``derive_arch_spec`` picks a roughly-square layout by default
    # (``mesh_width = ceil(sqrt(n_tiles))``) so the GUI floorplan and the
    # SANA-FE NoC routing both treat the chip as a 2D mesh.
    mesh_width: int = 1
    mesh_height: int = 1
    # Final cores-per-tile (after auto-resolution of ``0``).  Stored so
    # ``net_synth`` can pack cores into tiles using the same arithmetic
    # the YAML synthesiser did — without this the runner and the arch
    # disagree on which tile each core lives in.
    cores_per_tile_resolved: int = 1

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

    # Plugin paths are resolved here so a missing build surfaces early
    # with a clear pointer to ``scripts/bootstrap_sanafe.sh``.
    dendrite_so = _plugin_path("dendrite")
    soma_so = _plugin_path("soma")
    if dendrite_so is None or soma_so is None:
        raise FileNotFoundError(
            "mimarsinan SANA-FE plugins are not built.  Run "
            "``scripts/bootstrap_sanafe.sh`` (with the project venv active) "
            "to build libmimarsinan_dendrite.so and libmimarsinan_soma.so."
        )

    # One SANA-FE core per HardCore — no extra input-host cores.
    # ``cores_per_tile <= 0`` means "auto": pick ``ceil(sqrt(total_cores))``
    # so the chip is a roughly-square 2D mesh, which makes the GUI
    # floorplan + NoC visualisations actually 2D (a 1×N row of cores
    # collapses both axes onto the same line and hides spatial structure).
    # Users who want a single-tile chip can pass ``cores_per_tile=total_cores``
    # explicitly.
    if cores_per_tile <= 0:
        cores_per_tile = max(1, math.isqrt(total_cores))
        if cores_per_tile * cores_per_tile < total_cores:
            cores_per_tile += 1
    n_tiles = (total_cores + cores_per_tile - 1) // cores_per_tile
    n_cores_per_tile = [cores_per_tile] * (n_tiles - 1)
    last = total_cores - cores_per_tile * (n_tiles - 1)
    n_cores_per_tile.append(last)

    # 2D mesh layout for the SANA-FE NoC.  ``mesh_width = ceil(sqrt(n_tiles))``
    # produces a roughly-square chip; remainder rows on the last row are fine
    # — SANA-FE accepts under-filled mesh rows.  Captured into ArchSpec so
    # the runner can stamp ``tile_index → (mesh_x, mesh_y)`` into
    # ``SanafeArchGeometry`` for the GUI floorplan.
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
        mesh_width=mesh_width,
        mesh_height=mesh_height,
        cores_per_tile_resolved=cores_per_tile,
    )


# ---------------------------------------------------------------------------
# YAML synthesis
# ---------------------------------------------------------------------------


def _render_arch_yaml(spec: ArchSpec) -> str:
    """Render a SANA-FE-compatible architecture YAML from the spec.

    The YAML embeds every hardware unit ``net_synth`` references by name
    (``SYNAPSE_NAME``, ``DENDRITE_NAME``, ``SOMA_LIF_NAME``,
    ``SOMA_INPUT_RANGE_NAME``, ``AXON_IN_NAME``, ``AXON_OUT_NAME``).
    Per-event numbers come from the spec's preset, never local literals.
    """
    p = spec.preset
    # Per-core ``inputs[0..N-1]`` pool — sized to this core's axon
    # capacity.  Each HardCore's input neurons live on the SANA-FE
    # core that consumes them (no global input host), so the pool only
    # needs to be ≥ ``axons_per_core``.
    n_input_somas = max(1, int(spec.axons_per_core))

    def _range_name(stem: str, count: int) -> str:
        return stem if count == 1 else f"{stem}[0..{count - 1}]"

    def _render_core_block(tile_idx: int, core_local_idx: int) -> str:
        """Render one core block.  We emit one block per core (no
        ``name[0..N-1]`` shorthand on cores) because SANA-FE 2.1.1's
        shorthand expansion does not propagate the inner ``inputs[0..K]``
        soma range correctly across multiple cores.

        The dendrite and the LIF soma both load mimarsinan-owned plugin
        shared libraries (``libmimarsinan_dendrite.so``,
        ``libmimarsinan_soma.so``).  This replaces SANA-FE's built-in
        ``accumulator`` and ``leaky_integrate_fire`` — both of which
        bake in Loihi's 1024-neuron-per-core cap — with versions whose
        per-neuron state grows dynamically.  See the plugin sources for
        the full rationale.
        """
        return f"""        - name: t{tile_idx}_c{core_local_idx}
          attributes:
            buffer_position: soma
            buffer_inside_unit: false
            max_neurons_supported: {max(8192, 2 * (spec.neurons_per_core + n_input_somas))}
          axon_in:
            - name: {AXON_IN_NAME}
              attributes:
                energy_message_in: {p["axon_in_energy_j"]}
                latency_message_in: {p["axon_in_latency_s"]}
          synapse:
            - name: {SYNAPSE_NAME}
              attributes:
                model: current_based
                energy_process_spike: {p["synapse_energy_j"]}
                latency_process_spike: {p["synapse_latency_s"]}
          dendrite:
            - name: {DENDRITE_NAME}
              attributes:
                plugin: {spec.dendrite_plugin_path}
                model: mimarsinan_dendrite
                update_every_timestep: true
                energy_update: {p["dendrite_energy_j"]}
                latency_update: {p["dendrite_latency_s"]}
          soma:
            - name: {SOMA_LIF_NAME}
              attributes:
                plugin: {spec.soma_plugin_path}
                model: mimarsinan_soma
                thresholding_mode: strict
                energy_access_neuron: {p["soma_access_energy_j"]}
                latency_access_neuron: {p["soma_access_latency_s"]}
                energy_update_neuron: {p["soma_update_energy_j"]}
                latency_update_neuron: {p["soma_update_latency_s"]}
                energy_spike_out: {p["soma_spike_out_energy_j"]}
                latency_spike_out: {p["soma_spike_out_latency_s"]}
            - name: {_range_name(SOMA_INPUT_RANGE_NAME, n_input_somas)}
              attributes:
                model: input
                energy_access_neuron: 0.0
                latency_access_neuron: 0.0
                energy_update_neuron: 0.0
                latency_update_neuron: 0.0
                energy_spike_out: 0.0
                latency_spike_out: 0.0
          axon_out:
            - name: {AXON_OUT_NAME}
              attributes:
                energy_message_out: {p["axon_out_energy_j"]}
                latency_message_out: {p["axon_out_latency_s"]}"""

    tile_lines: list[str] = []
    for tile_idx, n_cores in enumerate(spec.n_cores_per_tile):
        core_blocks = "\n".join(
            _render_core_block(tile_idx, c) for c in range(n_cores)
        )
        tile_lines.append(f"""    - name: tile{tile_idx}
      attributes:
        energy_north_hop: {p["tile_hop_energy_j"]}
        latency_north_hop: {p["tile_hop_latency_s"]}
        energy_east_hop: {p["tile_hop_energy_j"]}
        latency_east_hop: {p["tile_hop_latency_s"]}
        energy_south_hop: {p["tile_hop_energy_j"]}
        latency_south_hop: {p["tile_hop_latency_s"]}
        energy_west_hop: {p["tile_hop_energy_j"]}
        latency_west_hop: {p["tile_hop_latency_s"]}
      core:
{core_blocks}""")

    yaml_str = f"""architecture:
  name: {spec.name}
  attributes:
    topology: mesh
    width: {spec.mesh_width}
    height: {spec.mesh_height}
    link_buffer_size: 16
    sync_model: fixed
    latency_sync: 0.0
  tile:
""" + "\n".join(tile_lines) + "\n"
    return yaml_str


# ---------------------------------------------------------------------------
# build_architecture — touches SANA-FE via _sanafe()
# ---------------------------------------------------------------------------


def build_architecture(
    spec: ArchSpec,
    *,
    custom_arch_path: Optional[str] = None,
) -> Any:
    """Construct (or load) a SANA-FE Architecture matching ``spec``.

    If ``custom_arch_path`` is provided, ``sanafe.load_arch`` is used
    directly on that file; the loaded architecture is validated against
    the spec's total core count.  Otherwise an in-memory YAML is rendered
    from the spec, written to a tempfile, and loaded.
    """
    sanafe = _sanafe()

    if custom_arch_path is not None:
        if not os.path.isfile(custom_arch_path):
            raise FileNotFoundError(
                f"SANA-FE custom arch YAML not found: {custom_arch_path}"
            )
        arch = sanafe.load_arch(custom_arch_path)
        loaded_cores = sum(len(tile.cores) for tile in arch.tiles)
        if loaded_cores < spec.total_cores:
            raise ValueError(
                f"custom arch at {custom_arch_path} provides only "
                f"{loaded_cores} cores but the mapping needs {spec.total_cores}"
            )
        return arch

    yaml_str = _render_arch_yaml(spec)
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False,
        prefix=f"mimarsinan_sanafe_arch_{spec.name}_",
    ) as f:
        f.write(yaml_str)
        path = f.name
    try:
        return sanafe.load_arch(path)
    finally:
        # Keep the file around for post-mortem if SANA-FE failed to parse.
        # We don't unlink — let the OS reclaim /tmp.
        pass
