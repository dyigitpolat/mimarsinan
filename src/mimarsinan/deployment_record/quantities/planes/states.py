"""[U-ENERGY class 2] What each plane never counts and what a platform's declarations make zero, with reasons.

Every factor of every component table is, on each plane, produced (a value, a known zero among them), withheld with
the view's reason, not produced on the plane with the plane's reason (named beside the estimate, never a refusal), or
zero by the platform's declaration (its reason printed in the basis). The tables below are data: a key a view
produces or withholds stays so, and a key a declaration zeroes is not "not produced"."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

from mimarsinan.deployment_record.quantities.spec import Quantities, QuantityValue

_NO_PASSES = "the candidate states no pass structure"
_NO_GEOMETRY = "the platform declares no core geometry"
_NO_CONVERTER = "the description's conversion model derives no converter count here"
_NO_CARRY = "the candidate states no carry census (no wire census, discipline or crossing wire)"
#: What the candidate plane never counts, each with its reason.
CANDIDATE_UNPRODUCED: Mapping[str, str] = {
    "boundary_events": "the candidate plane counts no row drives per inference (a sealed record's boundary traffic "
                       "does, a bit-sliced dataflow derives them)",
    "noc_intra_tile_packets": "the spike-packet census needs a declared activity_factor and the wire census",
    "noc_total_hops": "the spike-packet census needs a declared activity_factor and the wire census",
    "neurons_used": "the pass programs state no used columns per core",
    "local_mem_read_bytes": "the pass programs state no used rows per core",
    "local_mem_write_bytes": "the pass programs state no used columns per core",
    "adc_conversions": _NO_CONVERTER, "adc_count": _NO_CONVERTER,
    "lane_ops": "no lane block is declared, or no lane model estimates its work",
    "lanes_per_tile": "no lane block is declared",
    "carry_onchip_bytes": "no on-chip carry buffer holds this program's carry",
    "carry_buffer_capacity_bytes": "no on-chip carry buffer is declared",
    "connectivity_entries": "the candidate plane counts no connectivity entries (a sealed record does)",
    "noc_value_bytes": "the value NoC census needs the value domain and the wire census",
    "carry_in_bytes": _NO_CARRY, "carry_out_bytes": _NO_CARRY,
    "synaptic_events": "the event census needs the value rule or a declared activity_factor",
    "timesteps": "the platform declares no simulation_steps",
    "sync_count": "the candidate states no schedule census",
    "segment_cores": _NO_PASSES, "reprogrammed_cores": _NO_PASSES, "cores_allocated": _NO_PASSES,
    "reprogrammed_bytes": "the platform declares no weight width",
    "cores_physical": "the platform declares no cores",
    "cells_physical": _NO_GEOMETRY, "axons_physical": _NO_GEOMETRY, "neurons_physical": _NO_GEOMETRY,
    "weight_bits": "the platform declares no weight width",
    "tiles": "the platform resolves no tile grid",
}
_NO_NOC = "the record seals no NoC traffic census"
#: [U-ENERGY m-r2-9] What the record plane never counts, or counts only where the record seals it, with the reason.
RECORD_UNPRODUCED: Mapping[str, str] = {
    "adc_conversions": "a record seals no converter count (the description's conversion model derives it)",
    "adc_count": "a record seals no converter count (the description's conversion model derives it)",
    "boundary_events": "the record seals no boundary traffic census",
    "noc_total_packets": _NO_NOC, "noc_intra_tile_packets": _NO_NOC, "noc_inter_tile_packets": _NO_NOC,
    "noc_total_hops": _NO_NOC,
    "synaptic_events": "the record seals no event census",
    "total_spikes": "the record seals no spike count",
    "synaptic_events_busiest_core": "a sealed event census is one total per inference, never per core",
    "noc_value_bytes": "the record states no value NoC census",
    "host_ops_s": "the record seals no measured host wall",
    "segment_in_bytes": "the record plane states no host-chip boundary census",
    "segment_out_bytes": "the record plane states no host-chip boundary census",
    "host_link_transfers": "the record plane states no host-chip boundary census",
    "host_memory_bound_bytes": "the record plane states no host-chip boundary census",
    "cores_physical": "the platform declares no cores", "cells_physical": "the platform declares no cores",
    "axons_physical": "the platform declares no cores", "neurons_physical": "the platform declares no cores",
    "onchip_params": "the record seals no partition census", "host_params": "the record seals no partition census",
    "total_params": "the record seals no partition census", "onchip_macs": "the record seals no partition census",
    "host_macs": "the record seals no partition census", "total_macs": "the record seals no partition census",
}
#: Why the traffic between a capability level's cores has no size (the level plane's mesh census).
NO_LEVEL_MESH = "the level census places no core on the chip's mesh, so the traffic between its cores has no size"
_NO_LEVEL_EXTENTS = "the requirement placement states no used rows or columns per core"
_NO_LEVEL_BOUNDARY = "the level plane states no host-chip boundary census"
#: Why the level plane's load census states no connectivity entries.
LEVEL_NO_ENTRIES = "the level plane counts no connectivity entries"
#: [U-ENERGY class 2, the class swept] What the capability level's plane never counts, with the reason.
LEVEL_UNPRODUCED: Mapping[str, str] = {
    "neurons_used": _NO_LEVEL_EXTENTS, "local_mem_read_bytes": _NO_LEVEL_EXTENTS,
    "local_mem_write_bytes": _NO_LEVEL_EXTENTS,
    "boundary_events": "the level plane counts no row drives per inference (a bit-sliced dataflow derives them)",
    "adc_conversions": _NO_CONVERTER, "adc_count": _NO_CONVERTER,
    **dict.fromkeys(("noc_total_packets", "noc_intra_tile_packets", "noc_inter_tile_packets", "noc_total_hops",
                     "noc_value_bytes"), NO_LEVEL_MESH),
    "connectivity_entries": LEVEL_NO_ENTRIES,
    "carry_onchip_bytes": "the level plane states no on-chip carry census",
    "lane_ops": "the level census states no lane work", "lanes_per_tile": "the level census states no lane work",
    "synaptic_events_busiest_core": "the level plane states no per-core event census",
    "total_spikes": "the level plane counts no spikes",
    **dict.fromkeys(("host_ops_s", "segment_in_bytes", "segment_out_bytes", "host_link_transfers",
                     "host_memory_bound_bytes"), _NO_LEVEL_BOUNDARY),
}
#: A value-domain program sends values, never spikes: its spike-packet census is a known zero.
VALUE_DOMAIN_ZEROS = ("noc_total_packets", "noc_intra_tile_packets", "noc_inter_tile_packets", "noc_total_hops")


def declared_zeros(lane_block: bool, carry_buffer: bool, value_domain: Optional[bool] = None) -> Dict[str, str]:
    """The work a platform's declarations make zero, with the reason (one rule on both planes): no lane block, no
    on-chip carry buffer, and a platform that names no value domain sends spikes, never values."""
    zeros: Dict[str, str] = {}
    if value_domain is False:
        zeros["noc_value_bytes"] = "the platform names no value domain: its cores send spikes, never values"
    if not lane_block:
        zeros.update({k: "the chip declares no lane block" for k in ("lane_ops", "lanes_per_tile")})
    if not carry_buffer:
        zeros.update({k: "the chip declares no on-chip carry buffer"
                      for k in ("carry_onchip_bytes", "carry_buffer_capacity_bytes")})
    return zeros


def plane_view(values: Dict[str, QuantityValue], withheld: Mapping[str, str], unproduced: Mapping[str, str],
               zeros: Mapping[str, str]) -> Quantities:
    """A plane's view in the four states: what it produces, withholds, zeroes by declaration and never counts (each
    key in exactly one state; ``Quantities`` keeps the first that holds it, in that order)."""
    return Quantities(values, withheld={k: r for k, r in withheld.items() if k not in values},
                      unproduced=unproduced, known_zero=zeros)
