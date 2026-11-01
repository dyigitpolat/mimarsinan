"""Build SANA-FE ``Architecture`` instances from ``ArchSpec``."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

import mimarsinan.chip_simulation.sanafe.arch_synth as _arch_synth
from mimarsinan.chip_simulation.sanafe.arch_synth.core_sizes import (
    declared_core_size,
    input_slots,
    neuron_capacity,
)
from mimarsinan.chip_simulation.sanafe.arch_synth.placed_region import placed_region
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
    SOMA_TTFS_CASCADE_NAME,
    SOMA_TTFS_CONTINUOUS_NAME,
    SOMA_TTFS_CYCLE_NAME,
    SOMA_TTFS_QUANTIZED_NAME,
    SYNAPSE_NAME,
)


def _render_arch_yaml(
    spec: ArchSpec,
    *,
    thresholding_mode: str = "<=",
    simulation_length: int = 1,
) -> str:
    """Render SANA-FE architecture YAML from ``ArchSpec``: each core at its declared size (``core_sizes``) [U-SF]."""
    p = spec.preset
    soma_thresholding = _thresholding_mode_to_soma_attr(thresholding_mode)

    def _render_core_block(tile_idx: int, core_local_idx: int) -> str:
        """One core block (explicit per core — SANA-FE 2.1.1 range shorthand is broken); every input slot ``inputs[k]``
        is named by its range, one slot included (the net builder binds input axon ``a`` to ``inputs[a]``)."""
        axons, neurons = declared_core_size(spec, tile_idx, core_local_idx)
        n_input_somas = input_slots(axons)
        return f"""        - name: t{tile_idx}_c{core_local_idx}
          attributes:
            buffer_position: soma
            buffer_inside_unit: false
            max_neurons_supported: {neuron_capacity(axons, neurons)}
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
            - name: {SOMA_TTFS_CYCLE_NAME}
              attributes:
                plugin: {spec.ttfs_cycle_plugin_path}
                model: mimarsinan_ttfs_cycle_soma
                simulation_length: {simulation_length}
                energy_access_neuron: {p["soma_access_energy_j"]}
                latency_access_neuron: {p["soma_access_latency_s"]}
                energy_update_neuron: {p["soma_update_energy_j"]}
                latency_update_neuron: {p["soma_update_latency_s"]}
                energy_spike_out: {p["soma_spike_out_energy_j"]}
                latency_spike_out: {p["soma_spike_out_latency_s"]}
            - name: {SOMA_TTFS_CASCADE_NAME}
              attributes:
                plugin: {spec.ttfs_cascade_plugin_path}
                model: mimarsinan_ttfs_cascade_soma
                thresholding_mode: {soma_thresholding}
                energy_access_neuron: {p["soma_access_energy_j"]}
                latency_access_neuron: {p["soma_access_latency_s"]}
                energy_update_neuron: {p["soma_update_energy_j"]}
                latency_update_neuron: {p["soma_update_latency_s"]}
                energy_spike_out: {p["soma_spike_out_energy_j"]}
                latency_spike_out: {p["soma_spike_out_latency_s"]}
            - name: {SOMA_INPUT_RANGE_NAME}[0..{n_input_somas - 1}]
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
    """Load or synthesise a SANA-FE ``Architecture`` matching ``spec``: a synthesized chip defines the declared
    floorplan's region holding the placed cores (``placed_region``) [U-TT T-1], each core at the size the program places
    on it (``core_sizes``) [U-SF]."""
    sanafe = _arch_synth._sanafe()

    if custom_arch_path is not None:
        if not os.path.isfile(custom_arch_path):
            raise FileNotFoundError(
                f"SANA-FE custom arch YAML not found: {custom_arch_path}"
            )
        arch = sanafe.load_arch(custom_arch_path)
        loaded_cores = sum(len(tile.cores) for tile in arch.tiles)
        # packed_cores is the mapping's real need; total_cores counts floorplan
        # slots (capacity) and over-demands when idle slots are defined.
        needed = spec.packed_cores if spec.packed_cores > 0 else spec.total_cores
        if loaded_cores < needed:
            raise ValueError(
                f"custom arch at {custom_arch_path} provides only "
                f"{loaded_cores} cores but the mapping needs {needed}"
            )
        return arch

    yaml_str = _render_arch_yaml(
        placed_region(spec),
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
