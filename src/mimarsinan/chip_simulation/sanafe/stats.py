"""SanafeStepReport — what the pipeline step persists and the GUI consumes.

A report aggregates one or more :class:`SanafeRunRecord`s into:

* a flat ``aggregate`` dict of headline numbers for the summary cards,
* a per-sample list of compact summaries the GUI can render directly.

The full rich records remain on the report (``per_sample`` attribute) so
downstream code or notebooks can drill into per-tile / per-neuron data
without re-running SANA-FE.  Only the JSON-safe projection is shipped to
the frontend.
"""

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
            }
            for t in seg.per_tile
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
                "energy_j": float(c.energy.total_j),
                "energy_breakdown_j": _eb_to_dict(c.energy),
            }
            for c in seg.per_core
        ],
        "has_neuron_spike_trace": seg.per_neuron_spike_trace is not None,
        "has_neuron_potential_trace": seg.per_neuron_potential_trace is not None,
        "has_message_trace": seg.message_trace is not None,
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

    # ------------------------------------------------------------------ ctor

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

    # --------------------------------------------------------------- snapshot

    def to_snapshot_dict(self) -> Dict[str, Any]:
        """JSON-safe dict consumed by ``gui/snapshot/builders.py``."""
        return {
            "arch_preset": str(self.arch_preset),
            "sample_indices": [int(i) for i in self.sample_indices],
            "aggregate": dict(self.aggregate),  # shallow copy; values already primitives
            "per_sample": [_record_summary(r) for r in self.per_sample],
        }
