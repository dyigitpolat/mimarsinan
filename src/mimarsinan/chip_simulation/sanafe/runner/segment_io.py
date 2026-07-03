"""SANA-FE per-core input/output spike accounting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation.sanafe.analysis import _group_name
from mimarsinan.chip_simulation.sanafe.records import SanafeCoreRecord, SanafeEnergyBreakdown, SanafeTileRecord
from mimarsinan.mapping.support.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources


class SanafeSegmentIOMixin:
    """Derive segment spike counts and tile aggregates."""

    def _derive_per_core_input_counts(
        self,
        *,
        core: Any,
        seg_input_encoded: np.ndarray,
        group_spike_counts: Dict[str, np.ndarray],
        core_to_group: Dict[int, Any],
        seg_raster: Optional[np.ndarray] = None,
        group_row_offsets: Optional[Dict[str, int]] = None,
        consumer_latency: int = 0,
        hcm: Any = None,
    ) -> Tuple[np.ndarray, int]:
        """Per-axon input spike counts for one core (HCM window semantics)."""
        used_ax = _used_axons(core)
        if used_ax <= 0:
            return np.zeros(0, dtype=np.int64), 0
        counts = np.zeros(used_ax, dtype=np.int64)
        n_always_on = 0
        T = self.T
        read_start = max(consumer_latency, 0)
        read_end = consumer_latency + T
        n_read = read_end - read_start
        for a in range(used_ax):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_always_on_", False):
                from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

                counts[a] = 1 if is_cascaded_ttfs(
                    self.spiking_mode, self.ttfs_cycle_schedule
                ) else T
                n_always_on += 1
                continue
            if getattr(src, "is_input_", False):
                k = int(src.neuron_)
                lo = min(consumer_latency, T)
                hi = min(consumer_latency + T, T)
                if hi > lo:
                    counts[a] = int(seg_input_encoded[0, k, lo:hi].sum())
                continue
            src_core = int(src.core_)
            src_neuron = int(src.neuron_)
            if seg_raster is None or group_row_offsets is None or hcm is None:
                src_group = core_to_group.get(src_core)
                if src_group is None:
                    continue
                gsc = group_spike_counts.get(_group_name(src_group))
                if gsc is None or src_neuron >= len(gsc):
                    continue
                counts[a] = int(gsc[src_neuron])
                continue
            src_group = core_to_group.get(src_core)
            if src_group is None:
                continue
            row_off = group_row_offsets.get(_group_name(src_group))
            if row_off is None:
                continue
            row_idx = row_off + src_neuron
            if row_idx >= seg_raster.shape[0]:
                continue
            src_core_obj = hcm.cores[src_core] if hasattr(hcm, "cores") else None
            src_lat = (
                int(src_core_obj.latency)
                if src_core_obj is not None and src_core_obj.latency is not None
                else 0
            )
            src_active_start = src_lat + 1
            src_active_end = src_lat + T + 1
            src_last_active_idx = src_lat + T
            total = 0
            for k in range(n_read):
                t = read_start + k
                if t < src_active_start:
                    continue
                if t < src_active_end:
                    src_t = t
                else:
                    src_t = src_last_active_idx
                if 0 <= src_t < seg_raster.shape[1]:
                    total += int(seg_raster[row_idx, src_t])
            counts[a] = total
        return counts, n_always_on


    def _collect_first_fire_cycles(
        self,
        spike_trace: list,
        *,
        core_to_group: Dict[int, Any],
        hcm: Any,
    ) -> Dict[int, np.ndarray]:
        """Per-core array of each neuron's FIRST fire cycle (``-1`` if silent).

        Single-spike TTFS encodes the value in the fire time, so the host
        reconstructs the ramp value ``T_eff − fire_cycle`` from this."""
        out: Dict[int, np.ndarray] = {}
        if not spike_trace:
            return out
        for core_idx, group in core_to_group.items():
            group_name = _group_name(group)
            n = len(group)
            first = np.full(n, -1, dtype=np.int64)
            for cycle, events in enumerate(spike_trace):
                for event in events:
                    s = str(event)
                    if "." not in s:
                        continue
                    ev_group, idx_str = s.rsplit(".", 1)
                    if ev_group != group_name:
                        continue
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    if 0 <= idx < n and first[idx] < 0:
                        first[idx] = cycle
            out[core_idx] = first
        return out

    def _single_spike_ramp_outputs(
        self,
        spike_trace: list,
        *,
        core_to_group: Dict[int, Any],
        hcm: Any,
        T_eff: int,
    ) -> Dict[int, np.ndarray]:
        """Per-core ramp value per neuron from fire timing within the core's own window:
        ``(core.latency + 1 + T) − first_fire`` clamped to ``[0, T]`` (0 if silent).
        Per-source windowing is essential — a global ``T_eff − fire`` overcounts shallow sources."""
        first = self._collect_first_fire_cycles(
            spike_trace, core_to_group=core_to_group, hcm=hcm,
        )
        ramps: Dict[int, np.ndarray] = {}
        for core_idx, fires in first.items():
            core = hcm.cores[core_idx]
            core_lat = int(getattr(core, "latency", 0) or 0)
            window_end = core_lat + 1 + int(self.T)
            ramp = np.where(
                fires >= 0,
                np.clip(window_end - fires, 0, int(self.T)),
                0,
            ).astype(np.int64)
            ramps[core_idx] = ramp
        return ramps

    def _compute_seg_output_spike_count(
        self,
        output_map: List[Any],
        per_core_records: List[SanafeCoreRecord],
        *,
        output_sources: np.ndarray,
        T: int,
        hcm: Any,
        last_active_fires: Dict[int, np.ndarray],
        core_output_override: Dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Gather per-output spike counts. ``core_output_override`` (single-spike
        cascade) supplies the reconstructed ramp value per core neuron, used in
        place of the raw per-core spike count for the gather."""
        flat_sources = (
            list(output_sources.flatten())
            if output_sources is not None and hasattr(output_sources, "flatten")
            else (list(output_sources) if output_sources else [])
        )
        n_out = len(flat_sources)
        if n_out == 0:
            if not output_map:
                return np.zeros(0, dtype=np.int64)
            total_size = max((s.offset + s.size for s in output_map), default=0)
            out = np.zeros(total_size, dtype=np.int64)
            flat = (np.concatenate([rec.output_spike_count for rec in per_core_records],
                                    axis=0)
                    if per_core_records else np.zeros(0, dtype=np.int64))
            cursor = 0
            for slot in output_map:
                take = min(slot.size, max(flat.size - cursor, 0))
                if take > 0:
                    out[slot.offset:slot.offset + take] = flat[cursor:cursor + take]
                cursor += slot.size
            return out
        out = np.zeros(n_out, dtype=np.int64)

        core_outputs: Dict[int, np.ndarray] = (
            core_output_override if core_output_override is not None
            else {rec.core_index: rec.output_spike_count for rec in per_core_records}
        )

        spans = compress_spike_sources(flat_sources)
        for sp in spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                out[d0:d1] = int(T)
                continue
            if sp.kind == "input":
                continue
            buf = core_outputs.get(int(sp.src_core))
            if buf is None:
                continue
            s0 = int(sp.src_start)
            length = int(sp.length)
            take = min(length, max(buf.size - s0, 0))
            if take <= 0:
                continue
            out[d0:d0 + take] = buf[s0:s0 + take]
        return out

    def _collect_last_active_fires(
        self,
        spike_trace: list,
        *,
        core_to_group: Dict[int, Any],
        hcm: Any,
    ) -> Dict[int, np.ndarray]:
        """Per-core bitmap: neuron fired on its last active cycle."""
        out: Dict[int, np.ndarray] = {}
        if not spike_trace:
            return out
        T_eff = len(spike_trace)
        for core_idx, group in core_to_group.items():
            core = hcm.cores[core_idx]
            core_lat = int(getattr(core, "latency", 0) or 0)
            sf_last_cycle = self.T + core_lat
            if sf_last_cycle < 0 or sf_last_cycle >= T_eff:
                continue
            group_name = _group_name(group)
            n = len(group)
            bits = np.zeros(n, dtype=np.int64)
            for event in spike_trace[sf_last_cycle]:
                s = str(event)
                if "." not in s:
                    continue
                ev_group, idx_str = s.rsplit(".", 1)
                if ev_group != group_name:
                    continue
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if 0 <= idx < n:
                    bits[idx] = 1
            out[core_idx] = bits
        return out


    def _aggregate_per_tile(
        self,
        per_core_records: List[SanafeCoreRecord],
        results: Dict[str, Any],
        *,
        message_trace: Any = None,
    ) -> List[SanafeTileRecord]:
        """Roll per-core records up into SANA-FE tiles."""
        if not per_core_records:
            return []
        cpt = max(int(self.cores_per_tile), 0)
        if cpt == 0:
            cpt = len(per_core_records)
        geom = self._arch_geometry
        tiles: Dict[int, List[SanafeCoreRecord]] = {}
        for pos, rec in enumerate(per_core_records):
            tile_idx = pos // cpt
            tiles.setdefault(tile_idx, []).append(rec)

        hops_per_tile: Dict[int, int] = {}
        pkts_per_tile: Dict[int, int] = {}
        if message_trace:
            for events in message_trace:
                for ev in events:
                    if not isinstance(ev, dict) or ev.get("placeholder"):
                        continue
                    dt = int(ev.get("dest_tile_id", -1))
                    if dt < 0:
                        continue
                    hops_per_tile[dt] = (
                        hops_per_tile.get(dt, 0) + int(ev.get("hops", 0) or 0)
                    )
                    pkts_per_tile[dt] = pkts_per_tile.get(dt, 0) + 1
        hop_energy = float(self._preset.get("tile_hop_energy_j", 0.0))

        out: List[SanafeTileRecord] = []
        for tile_idx in sorted(tiles.keys()):
            cores = tiles[tile_idx]
            tile_energy = SanafeEnergyBreakdown.zero()
            for c in cores:
                tile_energy = tile_energy.add(c.energy)
            tile_hops = int(hops_per_tile.get(tile_idx, 0))
            hop_j = hop_energy * tile_hops
            if hop_j > 0.0:
                tile_energy = SanafeEnergyBreakdown(
                    synapse_j=tile_energy.synapse_j,
                    dendrite_j=tile_energy.dendrite_j,
                    soma_j=tile_energy.soma_j,
                    network_j=tile_energy.network_j + hop_j,
                    total_j=tile_energy.total_j + hop_j,
                )
            mesh_x, mesh_y = -1, -1
            if geom and 0 <= tile_idx < len(geom.tiles_xy):
                mesh_x, mesh_y = geom.tiles_xy[tile_idx]
            out.append(SanafeTileRecord(
                tile_index=tile_idx,
                cores=[c.core_index for c in cores],
                energy=tile_energy,
                spikes_fired=int(sum(c.spikes_fired for c in cores)),
                packets_sent=int(pkts_per_tile.get(tile_idx, 0)),
                mesh_x=int(mesh_x), mesh_y=int(mesh_y),
            ))
        return out
