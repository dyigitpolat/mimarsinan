"""Synthesise a SANA-FE ``Architecture`` from a hybrid mapping."""

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


def _plugin_path(name: str) -> Optional[str]:
    """Absolute path to ``build/mimarsinan_sanafe_plugins/libmimarsinan_<name>.so``."""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
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
    if dendrite_so is None or soma_so is None:
        raise FileNotFoundError(
            "mimarsinan SANA-FE plugins are not built.  Run "
            "``scripts/bootstrap_sanafe.sh`` (with the project venv active) "
            "to build libmimarsinan_dendrite.so and libmimarsinan_soma.so."
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


def _render_arch_yaml(spec: ArchSpec, *, thresholding_mode: str = "<=") -> str:
    """Render SANA-FE architecture YAML from ``ArchSpec``."""
    p = spec.preset
    soma_thresholding = _thresholding_mode_to_soma_attr(thresholding_mode)
    n_input_somas = max(1, int(spec.axons_per_core))

    def _range_name(stem: str, count: int) -> str:
        return stem if count == 1 else f"{stem}[0..{count - 1}]"

    def _render_core_block(tile_idx: int, core_local_idx: int) -> str:
        """One core block (explicit per core — SANA-FE 2.1.1 range shorthand is broken)."""
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
                thresholding_mode: {soma_thresholding}
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


def build_architecture(
    spec: ArchSpec,
    *,
    custom_arch_path: Optional[str] = None,
    thresholding_mode: str = "<=",
) -> Any:
    """Load or synthesise a SANA-FE ``Architecture`` matching ``spec``."""
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

    yaml_str = _render_arch_yaml(spec, thresholding_mode=thresholding_mode)
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False,
        prefix=f"mimarsinan_sanafe_arch_{spec.name}_",
    ) as f:
        f.write(yaml_str)
        path = f.name
    return sanafe.load_arch(path)
