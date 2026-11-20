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

from mimarsinan.deployment_record.platform_physics.hosts.rows import HOST, HOST_ROWS
from mimarsinan.deployment_record.units import AREA, DATA, DIMENSIONLESS, ENERGY, POWER, TIME

ARRAY = "array"
PERIPHERY = "periphery"
NEURON = "neuron"
COMPUTE = "compute"
INTERCONNECT = "interconnect"
PROGRAMMING = "programming"
SYNC = "sync"
GLOBAL = "global"
AGGREGATE = "aggregate"

#: [U-ENERGY class 1] Every chip time a static power is charged over: the inference, the deployment's first load, the
#: reprogramming per round and the measured model's whole latency (one multiplicand alternative per wall).
_WALLS = ("e2e_latency_s", "latency_load_s", "latency_programming_s", "total_s")


def _over_walls(per: str = "") -> str:
    """The multiplicand of a static row: ``per`` (a count) x each wall, as alternatives."""
    return " | ".join(f"{per} x {wall}" if per else wall for wall in _WALLS)


#: Declaration order is the wizard's panel order.
GROUPS: Tuple[str, ...] = (
    ARRAY,
    PERIPHERY,
    NEURON,
    COMPUTE,
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
    # [U-C2 G-2] The time twin of e_mac: a chip without a fixed timestep prices its compute per synaptic operation.
    ("t_synaptic_op", ARRAY, TIME, "ns", "synaptic_events",
     "Wall time of one synaptic operation on an event-driven core that runs its operations one at a time: the time "
     "twin of e_mac, on the same event census (one operation per input event and neuron the core processes for it; a "
     "core that visits every neuron per input event counts its whole sweep). A chip without a fixed timestep declares "
     "it and leaves t_cycle undeclared; its compute time is t_synaptic_op x synaptic_events where one core runs every "
     "event and, on several concurrent cores, t_synaptic_op x synaptic_events_busiest_core (each pass at its busiest "
     "core, which the candidate plane states); refused by name without an event census or a declared activity_factor, "
     "or on several cores where a view states the census as one total (a sealed record). A declared t_cycle takes "
     "precedence: a timestep's wall holds its events."),
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
    # [U-ENERGY class 3] Core-local memories, read and write apart.
    ("e_local_mem_read_per_byte", PERIPHERY, ENERGY, "pJ", "local_mem_read_bytes",
     "Energy of one byte read from a core's local activation memories (its input buffer and the registers that "
     "stage a row's inputs into the array): every used row's input read into the array, per evaluation step."),
    ("e_local_mem_write_per_byte", PERIPHERY, ENERGY, "pJ", "local_mem_write_bytes",
     "Energy of one byte written into a core's local activation memories (its output buffer): every used column's "
     "output written out, per evaluation step."),
    # --- neuron: soma state and update -----------------------------------------------
    ("area_per_neuron_logic", NEURON, AREA, "um^2", "neurons_physical",
     "Silicon area of one neuron's update logic, excluding its state storage."),
    ("area_per_state_bit", NEURON, AREA, "um^2", "neurons_physical x membrane_bits",
     "Silicon area of one bit of per-neuron state storage."),
    ("membrane_bits", NEURON, DIMENSIONLESS, "bit", "declared, not multiplied",
     "Width of one neuron's membrane/accumulator state register."),
    ("e_neuron_update", NEURON, ENERGY, "pJ", "neurons_used x neuron_cycles_per_window",
     "Energy of one neuron state update at one timestep."),
    ("e_leak_per_neuron_step", NEURON, ENERGY, "pJ", "neurons_used x neuron_cycles_per_window",
     "Energy of one leak application per neuron per timestep, where leak is separable."),
    # --- compute: arithmetic lanes beside the array (on-chip capability levels) ---------
    ("area_per_lane", COMPUTE, AREA, "um^2", "lanes_per_tile x tiles",
     "Silicon area of one output arithmetic lane of a tile at the program's capability level."),
    ("e_lane_op", COMPUTE, ENERGY, "pJ", "lane_ops",
     "Energy of one element operation on an arithmetic lane."),
    ("t_lane_op", COMPUTE, TIME, "ns", "lane_ops",
     "Standalone wall of one element operation on one lane; a tile's lanes run in parallel, "
     "so the lane pass takes t_lane_op x lane_ops / (lanes_per_tile x tiles)."),
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
    ("e_noc_value_per_byte_hop", INTERCONNECT, ENERGY, "pJ", "noc_value_bytes",
     "Energy to move one byte of a value-domain activation across one inter-tile link: the "
     "value-domain twin of the spike-packet hop, priced on byte-hops rather than packets."),
    # --- programming: getting weights and connectivity onto the chip ------------------
    ("e_dma_per_byte", PROGRAMMING, ENERGY, "pJ",
     "reprogrammed_bytes | carry_in_bytes",
     "Energy to move one byte ONTO the chip over the target's DMA channel — "
     "programming payloads (per program load) and the host's re-injected "
     "pass-boundary trains (per inference) alike. [U-J4] The chip's backing-store "
     "row: beside the host-chip link (an edge host's rows) it prices only bytes from "
     "the chip's own store, never a byte that crosses the link (hosts.store); "
     "without link rows it is the chip's one channel, as before. [U-J5] Unless "
     "the chip description places it in a chip-internal buffer every arriving "
     "byte passes (transfer_rows, hosts.transfer): every byte entering that "
     "buffer pays it, crossing the link or not."),
    ("e_readout_per_byte", PROGRAMMING, ENERGY, "pJ", "carry_out_bytes",
     "Energy to move one byte OFF the chip: the pass-boundary emissions the "
     "host reads back. Its own constant because the readout path need not "
     "cost what the inbound DMA path costs; a target that does not declare it "
     "prices no readout term rather than borrowing the inbound number. [U-J4] "
     "Beside the host-chip link it prices only bytes written back to the chip's own "
     "store, never a byte that leaves over the link (hosts.store), unless [U-J5] "
     "the chip description places it in a chip-internal buffer (transfer_rows)."),
    ("e_carry_buffer_per_byte", PROGRAMMING, ENERGY, "pJ", "carry_onchip_bytes",
     "[U-F4] Energy of one byte access of a declared on-chip carry buffer (a carried byte is written once and read "
     "once; carry_onchip_bytes counts both accesses): the buffer's own figure, priced only where the chip declares "
     "carry_buffer_bytes; a target that does not declare it (with its source) refuses the energy by name rather "
     "than borrowing another store's figure. [U-G] An SRAM's access energy depends on its size: a profile may state "
     "it over capacity (its carry_buffer table's rows), read at the declared capacity, refused by name elsewhere."),
    ("p_carry_buffer_static_per_byte", PROGRAMMING, POWER, "nW", _over_walls("carry_buffer_capacity_bytes"),
     "[U-F4] Static (leakage) power of one declared byte of the on-chip carry buffer while the chip is powered, "
     "priced on the capacity the chip declares (carry_buffer_bytes), filled or not; a target declaring the buffer "
     "without it (with its source) refuses the energy by name."),
    ("area_carry_buffer_per_byte", PROGRAMMING, AREA, "um^2", "carry_buffer_capacity_bytes",
     "[U-F4] Silicon area of one declared byte of the on-chip carry buffer, beside the compute footprint on either "
     "area path; a target declaring the buffer without it (with its source) refuses the chip area by name."),
    ("e_program_per_byte", PROGRAMMING, ENERGY, "pJ", "reprogrammed_bytes",
     "Energy to COMMIT one byte of programming payload into the target's weight "
     "store — the write itself, beyond moving the byte onto the chip. Its own "
     "constant because the two costs differ by orders of magnitude on an NVM "
     "target (a program-verify write against a bus transfer) and because a "
     "carried activation crosses the DMA channel WITHOUT being written into the "
     "weight store: sharing one constant would charge every inference for a "
     "write that never happened. A target whose published per-byte figure "
     "already contains the write declares only one of the two."),
    ("bytes_per_connectivity_entry", PROGRAMMING, DATA, "B",
     "connectivity_entries",
     "Wire size of one connectivity (axon source span) entry in this target's format."),
    ("e_core_program", PROGRAMMING, ENERGY, "pJ", "reprogrammed_cores",
     "Fixed energy of programming one core, beyond its payload bytes."),
    # [U-C2 G-1] The time twin of e_core_program.
    ("t_core_program", PROGRAMMING, TIME, "us", "reprogrammed_cores",
     "Fixed wall time of programming one core, beyond its payload bytes: the time twin of e_core_program, read "
     "wherever the programming wall is (latency_programming_s, the write law's per-pass walls, so Pi, M and the "
     "write-bandwidth batch). An interface that writes a whole core image per reprogrammed core, whatever the "
     "program occupies, declares that image's wall here and no t_program_per_byte."),
    ("e_core_init", PROGRAMMING, ENERGY, "pJ", "segment_cores",
     "Energy to reset one core's neuron state at the start of a pass."),
    ("t_program_per_byte", PROGRAMMING, TIME, "ns", "reprogrammed_bytes",
     "Wall time to move one byte of programming payload to the chip. Declared per byte "
     "rather than as a bandwidth so the band stays monotone under multiplication."),
    ("t_core_init", PROGRAMMING, TIME, "ns", "segment_cores",
     "Wall time to reset one core's neuron state at the start of a pass."),
    # [CH-A1] The write law: what committing a cell during an inference costs and wears.
    ("e_write_per_cell", PROGRAMMING, ENERGY, "pJ", "runtime_written_cells x write_verify_iterations",
     "Energy of one write pulse into one weight cell (one program-verify iteration's write), "
     "for cells written during deployment: per period from the backing store, per inference "
     "from a producer."),
    ("e_verify_read_per_cell", PROGRAMMING, ENERGY, "pJ",
     "runtime_written_cells x write_verify_iterations",
     "Energy of the verify read after one write pulse into one cell."),
    ("write_verify_iterations", PROGRAMMING, DIMENSIONLESS, "iterations", "declared, not multiplied",
     "Mean write-verify iterations per written cell (1 for a write that needs no verify); a "
     "factor of the write energy and write time formulas."),
    ("t_write_per_unit", PROGRAMMING, TIME, "ns", "write_events x write_verify_iterations",
     "Wall time of one write step of the declared granularity (a cell, a row, a column or a "
     "core), per write-verify iteration."),
    ("t_write_setup_per_event", PROGRAMMING, TIME, "ns", "write_events",
     "Fixed wall time to open one write step (addressing, mode switch), beside its pulses."),
    ("endurance_cycles", PROGRAMMING, DIMENSIONLESS, "cycles", "declared, not multiplied",
     "Write cycles a cell sustains over its life; lifetime in inferences is endurance_cycles / "
     "(max_writes_per_cell_per_inference x pulses_per_write) (divided, not multiplied)."),
    # [U-C C-1] Endurance is counted in set and reset cycles, a write-verify write applies several pulses.
    ("pulses_per_write", PROGRAMMING, DIMENSIONLESS, "pulses", "declared, not multiplied",
     "Programming pulses one cell write applies to its cell, each counted as one cycle of endurance_cycles (a set "
     "or reset pulse): the wear of one write. Lifetime divides endurance_cycles by the most-written cell's writes "
     "per inference times pulses_per_write; a technology that declares none wears one pulse per write."),
    # --- sync: barriers between passes -------------------------------------------------
    ("e_sync_barrier", SYNC, ENERGY, "pJ", "sync_count",
     "Energy of one chip-wide synchronization barrier."),
    ("t_sync_barrier", SYNC, TIME, "ns", "sync_count",
     "Wall time of one chip-wide synchronization barrier."),
    # --- global: whole-chip constants ---------------------------------------------------
    ("t_cycle", GLOBAL, TIME, "ns", "latency_steps",
     "Wall time of one timestep. The single constant that converts the record's integer "
     "latency_steps into seconds, and therefore gates E2E latency and throughput. A chip "
     "without a fixed timestep (an event-driven core that runs its synaptic operations one "
     "at a time) declares t_synaptic_op instead and leaves t_cycle undeclared."),
    ("area_global_fixed", GLOBAL, AREA, "mm^2", "declared, not multiplied",
     "Whole-chip area that scales with neither cores nor tiles (host link and memory PHYs, controller, PLLs, "
     "pads): [U-A] the sum of the chip's declared chip_components at its node (area_components), else a sourced "
     "scalar; refused by name where neither exists (an estimate prices only on a projection)."),
    ("p_static_per_core", GLOBAL, POWER, "mW", _over_walls("cores_physical"),
     "Static (leakage) power of one core while powered: [U-ENERGY] over every priced chip time (the inference, "
     "the deployment's first load, the reprogramming per round, the measured model's whole latency)."),
    ("p_static_global", GLOBAL, POWER, "mW", _over_walls(),
     "Static (leakage) power of the whole chip outside its cores, over every priced chip time (as p_static_per_core)."),
) + HOST_ROWS + (
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
        "e_local_mem_read_per_byte",  # [U-ENERGY class 3] the whole-system figure holds a core's local memories
        "e_local_mem_write_per_byte",
        "e_neuron_update",
        "e_leak_per_neuron_step",
        "e_intra_tile_packet",
        "e_inter_tile_hop",
        "e_noc_value_per_byte_hop",
        "p_static_per_core",
        "p_static_global",
    ),
}

#: [U-C C-1] The decision class of a declaration of each constant (the registry's class names, data only). A constant
#: whose only reader is the lifetime the write law judges against ``min_lifetime_hours`` moves only which programs are
#: admitted: a capacity the built program is judged against without being shaped by it (the class rule), so it is
#: admissibility-bearing; every other constant is priced, so cost-only.
LIFETIME_JUDGED: Tuple[str, ...] = ("endurance_cycles", "pulses_per_write")
_JUDGED = ("admissibility_bearing", "read only by the lifetime the write law judges against min_lifetime_hours: a "
                                    "declaration moves which programs are admitted, never a priced term")
_PRICED = ("cost_only", "a constant the cost model prices with; nothing else reads it")


def decision_class_of(key: str) -> Tuple[str, str]:
    """(class, reason) of a declaration of the vocabulary constant ``key``; an unknown key refuses (KeyError)."""
    if key not in {row[0] for row in ROWS}:
        raise KeyError(f"{key!r} is not a declarable physics constant")
    return _JUDGED if key in LIFETIME_JUDGED else _PRICED
