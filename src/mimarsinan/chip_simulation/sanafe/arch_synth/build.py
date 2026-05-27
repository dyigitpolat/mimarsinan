"""Build SANA-FE ``Architecture`` instances from ``ArchSpec``."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

import mimarsinan.chip_simulation.sanafe.arch_synth as _arch_synth
from mimarsinan.chip_simulation.sanafe.arch_synth.spec import (
    ArchSpec,
    _thresholding_mode_to_soma_attr,
)
from mimarsinan.chip_simulation.sanafe.presets import (
    AXON_IN_NAME,
    AXON_OUT_NAME,
    DENDRITE_NAME,
    SOMA_INPUT_RANGE_NAME,
    SOMA_LIF_NAME,
    SOMA_TTFS_CONTINUOUS_NAME,
    SOMA_TTFS_QUANTIZED_NAME,
    SYNAPSE_NAME,
)


def _render_arch_yaml(
    spec: ArchSpec,
    *,
    thresholding_mode: str = "<=",
    simulation_length: int = 1,
) -> str:
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
            - name: {SOMA_TTFS_CONTINUOUS_NAME}
              attributes:
                plugin: {spec.ttfs_continuous_plugin_path}
                model: mimarsinan_ttfs_continuous_soma
                energy_access_neuron: {p["soma_access_energy_j"]}
                latency_access_neuron: {p["soma_access_latency_s"]}
                energy_update_neuron: {p["soma_update_energy_j"]}
                latency_update_neuron: {p["soma_update_latency_s"]}
                energy_spike_out: {p["soma_spike_out_energy_j"]}
                latency_spike_out: {p["soma_spike_out_latency_s"]}
            - name: {SOMA_TTFS_QUANTIZED_NAME}
              attributes:
                plugin: {spec.ttfs_quantized_plugin_path}
                model: mimarsinan_ttfs_quantized_soma
                simulation_length: {simulation_length}
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
    simulation_length: int = 1,
) -> Any:
    """Load or synthesise a SANA-FE ``Architecture`` matching ``spec``."""
    sanafe = _arch_synth._sanafe()

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

    yaml_str = _render_arch_yaml(
        spec,
        thresholding_mode=thresholding_mode,
        simulation_length=simulation_length,
    )
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", mode="w", delete=False,
        prefix=f"mimarsinan_sanafe_arch_{spec.name}_",
    ) as f:
        f.write(yaml_str)
        path = f.name
    return sanafe.load_arch(path)
