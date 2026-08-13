"""The closed vocabulary of declarable physics constants, as data.

Each row is ``(key, group, dimension, display_unit, multiplicand, doc)``. The
``multiplicand`` names the recorded quantity the constant is priced against — a bare
"energy" number is unusable, so the pairing is declared here rather than discovered at
a pricing site. Execution POLICY (overlap, residency-across-batch, schedule policy) is
deliberately absent: physics is banded numbers only, and policy already lives on the
platform-constraints surface.
"""

from __future__ import annotations

from typing import Mapping, Tuple

from mimarsinan.deployment_record.platform_physics.units import (
    AREA,
    DATA,
    DIMENSIONLESS,
    ENERGY,
    POWER,
    RATE,
    TIME,
)

ARRAY = "array"
PERIPHERY = "periphery"
NEURON = "neuron"
INTERCONNECT = "interconnect"
PROGRAMMING = "programming"
SYNC = "sync"
GLOBAL = "global"
HOST = "host"
AGGREGATE = "aggregate"

#: Declaration order is the wizard's panel order.
GROUPS: Tuple[str, ...] = (
    ARRAY,
    PERIPHERY,
    NEURON,
    INTERCONNECT,
    PROGRAMMING,
    SYNC,
    GLOBAL,
    HOST,
    AGGREGATE,
)

ROWS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    # --- array: the crossbar itself -------------------------------------------------
    ("area_per_cell", ARRAY, AREA, "um^2", "cells_physical",
     "Silicon area of one weight cell (one crossbar intersection), including its share "
     "of local wiring."),
    ("area_per_cell_per_weight_bit", ARRAY, AREA, "um^2",
     "cells_physical x weight_bits",
     "Incremental cell area per weight bit, for targets whose cell area scales with "
     "precision (digital SRAM arrays) rather than being fixed (analog conductance)."),
    ("e_mac", ARRAY, ENERGY, "pJ", "synaptic_events",
     "Array energy of one multiply-accumulate, excluding conversion and periphery. "
     "On an event-driven chip one synaptic event IS one MAC, so the event census "
     "is the multiplicand."),
    ("t_array_read", ARRAY, TIME, "ns", "timesteps",
     "Wall time of one array read (column integration) at the declared array size; "
     "one integration happens per timestep."),
    ("conductance_levels", ARRAY, DIMENSIONLESS, "levels", "declared, not multiplied",
     "Distinct programmable conductance states per cell; bounds representable weight "
     "precision."),
    ("write_sigma", ARRAY, DIMENSIONLESS, "fraction", "declared, not multiplied",
     "Relative programming noise (sigma/mu) of a written conductance."),
    ("read_sigma", ARRAY, DIMENSIONLESS, "fraction", "declared, not multiplied",
     "Relative read noise (sigma/mu) of a sensed column current."),
    # --- periphery: converters and drivers -------------------------------------------
    ("area_per_adc", PERIPHERY, AREA, "um^2", "adc_count",
     "Silicon area of one analog-to-digital converter instance."),
    ("adc_sharing_factor", PERIPHERY, DIMENSIONLESS, "columns", "declared, not multiplied",
     "Array columns multiplexed onto one ADC; sets how many converters a crossbar needs."),
    ("e_adc_conversion", PERIPHERY, ENERGY, "pJ", "adc_conversions",
     "Energy of one analog-to-digital conversion."),
    ("t_adc_conversion", PERIPHERY, TIME, "ns", "adc_conversions",
     "Wall time of one analog-to-digital conversion."),
    ("area_per_row_driver", PERIPHERY, AREA, "um^2", "axons_physical",
     "Silicon area of one row (axon) driver."),
    ("e_row_drive", PERIPHERY, ENERGY, "pJ", "boundary_events",
     "Energy to drive one row line for one activation."),
    # --- neuron: soma state and update -----------------------------------------------
    ("area_per_neuron_logic", NEURON, AREA, "um^2", "neurons_physical",
     "Silicon area of one neuron's update logic, excluding its state storage."),
    ("area_per_state_bit", NEURON, AREA, "um^2", "neurons_physical x membrane_bits",
     "Silicon area of one bit of per-neuron state storage."),
    ("membrane_bits", NEURON, DIMENSIONLESS, "bit", "declared, not multiplied",
     "Width of one neuron's membrane/accumulator state register."),
    ("e_neuron_update", NEURON, ENERGY, "pJ", "neurons_used x timesteps",
     "Energy of one neuron state update at one timestep."),
    ("e_leak_per_neuron_step", NEURON, ENERGY, "pJ", "neurons_used x timesteps",
     "Energy of one leak application per neuron per timestep, where leak is separable."),
    # --- interconnect: routers, tiles, hops -------------------------------------------
    ("area_per_router", INTERCONNECT, AREA, "um^2", "tiles",
     "Silicon area of one NoC router."),
    ("area_per_tile_fixed", INTERCONNECT, AREA, "um^2", "tiles",
     "Fixed per-tile area that is neither array, neuron logic nor router."),
    ("e_intra_tile_packet", INTERCONNECT, ENERGY, "pJ", "noc_intra_tile_packets",
     "Energy to deliver one spike packet within a tile."),
    ("e_inter_tile_hop", INTERCONNECT, ENERGY, "pJ", "noc_total_hops",
     "Energy of one packet traversal of one inter-tile link."),
    ("t_hop", INTERCONNECT, TIME, "ns", "noc_total_hops",
     "Wall time of one packet traversal of one inter-tile link."),
    # --- programming: getting weights and connectivity onto the chip ------------------
    ("e_dma_per_byte", PROGRAMMING, ENERGY, "pJ", "reprogrammed_bytes",
     "Energy to move one byte of programming payload to the chip."),
    ("bytes_per_connectivity_entry", PROGRAMMING, DATA, "B",
     "connectivity_entries",
     "Wire size of one connectivity (axon source span) entry in this target's format."),
    ("e_core_program", PROGRAMMING, ENERGY, "pJ", "reprogrammed_cores",
     "Fixed energy of programming one core, beyond its payload bytes."),
    ("e_core_init", PROGRAMMING, ENERGY, "pJ", "segment_cores",
     "Energy to reset one core's neuron state at the start of a pass."),
    ("t_program_per_byte", PROGRAMMING, TIME, "ns", "reprogrammed_bytes",
     "Wall time to move one byte of programming payload to the chip. Declared per byte "
     "rather than as a bandwidth so the band stays monotone under multiplication."),
    ("t_core_init", PROGRAMMING, TIME, "ns", "segment_cores",
     "Wall time to reset one core's neuron state at the start of a pass."),
    # --- sync: barriers between passes -------------------------------------------------
    ("e_sync_barrier", SYNC, ENERGY, "pJ", "sync_count",
     "Energy of one chip-wide synchronization barrier."),
    ("t_sync_barrier", SYNC, TIME, "ns", "sync_count",
     "Wall time of one chip-wide synchronization barrier."),
    # --- global: whole-chip constants ---------------------------------------------------
    ("t_cycle", GLOBAL, TIME, "ns", "latency_steps",
     "Wall time of one timestep. The single constant that converts the record's integer "
     "latency_steps into seconds, and therefore gates E2E latency and throughput."),
    ("area_global_fixed", GLOBAL, AREA, "mm^2", "declared, not multiplied",
     "Whole-chip area that scales with neither cores nor tiles (pads, PLLs, host "
     "interface)."),
    ("p_static_per_core", GLOBAL, POWER, "mW", "cores_physical x e2e_latency_s",
     "Static (leakage) power of one core while powered."),
    ("p_static_global", GLOBAL, POWER, "mW", "e2e_latency_s",
     "Static (leakage) power of the whole chip outside its cores."),
    # --- host: the other side of the NeuralOps/ComputeOps boundary ----------------------
    ("p_host", HOST, POWER, "W", "host_ops_s",
     "Power drawn by the host while it executes ComputeOps. Without it, moving work "
     "host-side is free and `subsume` wins every comparison by disappearing."),
    ("host_compute_rate", HOST, DIMENSIONLESS, "fraction", "host_ops_s",
     "Speed of the deployment host relative to the machine that measured host_ops_s; "
     "1.0 means the measuring machine is the deployment host."),
    ("host_macs_per_s", HOST, RATE, "G/s", "host_macs",
     "Sustained forward-MAC throughput of the deployment host, for pricing host time "
     "STATICALLY (host_macs / host_macs_per_s) when no measured wall exists — the "
     "candidate-time host model of owner decision 3. A DIVIDING constant: corners "
     "flip so the band stays monotone."),
    # --- aggregate: what a chip publishes when it publishes no decomposition ----------
    ("e_synaptic_event_total", AGGREGATE, ENERGY, "pJ", "synaptic_events",
     "Whole-system energy per synaptic event: measured chip power divided by the "
     "synaptic events that produced it, so it already contains array, neuron, routing "
     "and static contributions. Most published chips report this rather than a "
     "decomposition. It SUPERSEDES the decomposed terms — never sum both."),
    ("area_per_core_total", AGGREGATE, AREA, "um^2", "cores_physical",
     "Measured silicon footprint of one complete core — array, periphery, neuron "
     "logic, router and local memory together, as published core floorplans report "
     "it. SUPERSEDES the decomposed area constants; never sum both."),
)

#: An aggregate constant already contains the terms it supersedes; pricing must use the
#: aggregate OR the decomposition, never both, or every component is counted twice.
SUPERSEDES: Mapping[str, Tuple[str, ...]] = {
    "area_per_core_total": (
        "area_per_cell",
        "area_per_cell_per_weight_bit",
        "area_per_adc",
        "area_per_row_driver",
        "area_per_neuron_logic",
        "area_per_state_bit",
        "area_per_router",
        "area_per_tile_fixed",
    ),
    "e_synaptic_event_total": (
        "e_mac",
        "e_adc_conversion",
        "e_row_drive",
        "e_neuron_update",
        "e_leak_per_neuron_step",
        "e_intra_tile_packet",
        "e_inter_tile_hop",
        "p_static_per_core",
        "p_static_global",
    ),
}
