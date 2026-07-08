"""SanafeStepReport — pipeline-persisted aggregate of ``SanafeRunRecord``s for the GUI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .records import SanafeEnergyBreakdown, SanafeRunRecord


def _eb_to_dict(eb: SanafeEnergyBreakdown) -> Dict[str, float]:
    return {
        "synapse": float(eb.synapse_j),
        "dendrite": float(eb.dendrite_j),
        "soma": float(eb.soma_j),
        "network": float(eb.network_j),
        "total": float(eb.total_j),
    }


def _segment_to_dict(seg) -> Dict[str, Any]:
    return {
        "stage_index": int(seg.stage_index),
        "stage_name": str(seg.stage_name),
        "energy_j": float(seg.energy.total_j),
        "energy_breakdown_j": _eb_to_dict(seg.energy),
        "sim_time_s": float(seg.sim_time_s),
        "timesteps_executed": int(seg.timesteps_executed),
        "spikes": int(seg.spikes),
        "packets_sent": int(seg.packets_sent),
        "neurons_updated": int(seg.neurons_updated),
        "neurons_fired": int(seg.neurons_fired),
        "per_tile": [
            {
                "tile_index": int(t.tile_index),
                "cores": [int(c) for c in t.cores],
                "energy_j": float(t.energy.total_j),
                "energy_breakdown_j": _eb_to_dict(t.energy),
                "spikes_fired": int(t.spikes_fired),
                "packets_sent": int(t.packets_sent),
                "mesh_x": int(t.mesh_x),
                "mesh_y": int(t.mesh_y),
            }
            for t in seg.per_tile
        ],
        "arch_geometry": (
            {
                "width": int(seg.arch_geometry.width),
                "height": int(seg.arch_geometry.height),
                "tiles_xy": [[int(x), int(y)] for x, y in seg.arch_geometry.tiles_xy],
            }
            if seg.arch_geometry is not None else None
        ),
        "noc_links": [
            {
                "src_tile": int(L.src_tile),
                "dst_tile": int(L.dst_tile),
                "src_x": int(L.src_x), "src_y": int(L.src_y),
                "dst_x": int(L.dst_x), "dst_y": int(L.dst_y),
                "packet_count": int(L.packet_count),
                "spike_count": int(L.spike_count),
                "total_hops": int(L.total_hops),
            }
            for L in seg.noc_links
        ],
        "noc_link_load": [
            {
                "from_x": int(L.from_x), "from_y": int(L.from_y),
                "to_x": int(L.to_x), "to_y": int(L.to_y),
                "packet_count": int(L.packet_count),
            }
            for L in seg.noc_link_load
        ],
        "cycle_energy": [
            {
                "cycle": int(p.cycle),
                "synapse_j": float(p.synapse_j),
                "dendrite_j": float(p.dendrite_j),
                "soma_j": float(p.soma_j),
                "network_j": float(p.network_j),
                "total_j": float(p.total_j),
            }
            for p in seg.cycle_energy
        ],
        "cascade": [
            {"cycle": int(p.cycle), "depth": int(p.depth), "firings": int(p.firings)}
            for p in seg.cascade
        ],
        "critical_cores": [
            {"cycle": int(p.cycle), "core_index": int(p.core_index),
             "event_count": int(p.event_count)}
            for p in seg.critical_cores
        ],
        "connectivity": [
            {"src_core": int(e.src_core), "dst_core": int(e.dst_core),
             "weight_sum_abs": float(e.weight_sum_abs),
             "fan_count": int(e.fan_count)}
            for e in seg.connectivity
        ],
        "hcm_diff": [
            {"core_index": int(d.core_index),
             "input_delta_sum": int(d.input_delta_sum),
             "output_delta_sum": int(d.output_delta_sum)}
            for d in seg.hcm_diff
        ],
        "noc_traffic_per_cycle": [
            [list(map(int, q)) for q in cycle]
            for cycle in seg.noc_traffic_per_cycle
        ],
        "noc_link_load_per_cycle": [
            [list(map(int, q)) for q in cycle]
            for cycle in getattr(seg, "noc_link_load_per_cycle", ())
        ],
        "per_core": [
            {
                "core_index": int(c.core_index),
                "n_neurons": int(c.n_neurons),
                "n_axons_used": int(c.n_axons_used),
                "core_latency": int(c.core_latency),
                "has_hardware_bias": bool(c.has_hardware_bias),
                "n_always_on_axons": int(c.n_always_on_axons),
                "spikes_fired": int(c.spikes_fired),
                "input_neuron_spikes_fired": int(
                    getattr(c, "input_neuron_spikes_fired", 0),
                ),
                "energy_j": float(c.energy.total_j),
                "energy_breakdown_j": _eb_to_dict(c.energy),
                "spike_raster": (
                    None if c.spike_raster is None else c.spike_raster.tolist()
                ),
            }
            for c in seg.per_core
        ],
        "has_neuron_spike_trace": seg.per_neuron_spike_trace is not None,
        "has_neuron_potential_trace": seg.per_neuron_potential_trace is not None,
        "has_message_trace": seg.message_trace is not None,
        "inter_tile_packets": int(getattr(seg, "inter_tile_packets", 0)),
        "intra_tile_packets": int(getattr(seg, "intra_tile_packets", 0)),
        "input_path_packets": int(getattr(seg, "input_path_packets", 0)),
        "cross_tile_connectivity_edges": int(
            getattr(seg, "cross_tile_connectivity_edges", 0),
        ),
        "chip_spike_count": int(getattr(seg, "chip_spike_count", seg.spikes)),
        "lif_spike_count": int(getattr(seg, "lif_spike_count", 0)),
        "spike_trace_parse_skipped": int(
            getattr(seg, "spike_trace_parse_skipped", 0),
        ),
        "spike_capture_warning": getattr(seg, "spike_capture_warning", None),
        "mapped_cross_tile_axons": int(getattr(seg, "mapped_cross_tile_axons", 0)),
        "ttfs_contract_active_cores": int(
            getattr(seg, "ttfs_contract_active_cores", 0),
        ),
        "ttfs_hardware_active_cores": int(
            getattr(seg, "ttfs_hardware_active_cores", 0),
        ),
        "ttfs_event_active_cores": int(getattr(seg, "ttfs_event_active_cores", 0)),
        "ttfs_activation_event_mismatch_count": int(
            getattr(seg, "ttfs_activation_event_mismatch_count", 0),
        ),
        "tile_packets_per_cycle": [
            {str(k): int(v) for k, v in cycle.items()}
            for cycle in getattr(seg, "tile_packets_per_cycle", ())
        ],
    }


def _record_summary(rec: SanafeRunRecord) -> Dict[str, Any]:
    return {
        "sample_index": int(rec.sample_index),
        "T": int(rec.T),
        "arch_name": str(rec.arch_name),
        "total_energy_j": float(rec.aggregate_energy.total_j),
        "energy_breakdown_j": _eb_to_dict(rec.aggregate_energy),
        "max_sim_time_s": float(rec.aggregate_sim_time_s),
        "total_spikes": int(rec.total_spikes),
        "total_packets": int(rec.total_packets),
        "segments": [_segment_to_dict(seg) for _, seg in sorted(rec.segments.items())],
    }


@dataclass
class SanafeStepReport:
    """All SANA-FE output the step persists to the pipeline cache.

    ``per_sample`` keeps the full :class:`SanafeRunRecord`s so downstream
    code can re-extract rich data without rerunning the simulator.
    ``aggregate`` is a precomputed headline dict for fast UI rendering.
    """

    arch_preset: str
    sample_indices: List[int]
    per_sample: List[SanafeRunRecord]
    aggregate: Dict[str, Any] = field(default_factory=dict)


    @classmethod
    def from_records(
        cls, arch_preset: str, records: Sequence[SanafeRunRecord]
    ) -> "SanafeStepReport":
        recs = list(records)
        agg_e = SanafeEnergyBreakdown.zero()
        max_sim_time_s = 0.0
        total_spikes = 0
        total_packets = 0
        for r in recs:
            agg_e = agg_e.add(r.aggregate_energy)
            if r.aggregate_sim_time_s > max_sim_time_s:
                max_sim_time_s = r.aggregate_sim_time_s
            total_spikes += int(r.total_spikes)
            total_packets += int(r.total_packets)
        aggregate = {
            "sample_count": len(recs),
            "total_energy_j": float(agg_e.total_j),
            "total_energy_mj": float(agg_e.total_j) * 1000.0,
            "energy_breakdown_j": _eb_to_dict(agg_e),
            "max_sim_time_s": float(max_sim_time_s),
            "total_spikes": int(total_spikes),
            "total_packets": int(total_packets),
        }
        return cls(
            arch_preset=arch_preset,
            sample_indices=[int(r.sample_index) for r in recs],
            per_sample=recs,
            aggregate=aggregate,
        )


    def to_snapshot_dict(self) -> Dict[str, Any]:
        """JSON-safe dict consumed by ``gui/snapshot/builders.py``."""
        return {
            "arch_preset": str(self.arch_preset),
            "sample_indices": [int(i) for i in self.sample_indices],
            "aggregate": dict(self.aggregate),
            "per_sample": [_record_summary(r) for r in self.per_sample],
        }
