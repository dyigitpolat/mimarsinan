"""SANA-FE backend driver; sole caller of ``sanafe.SpikingChip``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation._spike_encoding import uniform_rate_encode
from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    store_segment_output_numpy,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.core_geometry import used_neurons as _used_neurons
from mimarsinan.mapping.spike_source_spans import compress_spike_sources

from .arch_synth import _sanafe, build_architecture, derive_arch_spec
from .net_synth import (
    build_network_for_segment, set_always_on_spike_trains, set_input_spike_trains,
)
from .presets import PRESETS
from .records import (
    SanafeArchGeometry,
    SanafeCascadePoint,
    SanafeConnectivityEdge,
    SanafeCoreRecord,
    SanafeCriticalCore,
    SanafeCycleEnergyPoint,
    SanafeEnergyBreakdown,
    SanafeNocLink,
    SanafeNocLinkLoad,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)


_RAW_INPUT_NODE_ID = -2

# float64 matches HCM; float32 drifts ±1 spike at rate-encoding boundaries.
_COMPUTE_DTYPE: np.dtype = np.float64


class SanafeRunner:
    """Run one hybrid-mapping sample through SANA-FE."""

    def __init__(
        self,
        mapping: Any,
        simulation_length: int,
        *,
        arch_preset: str = "loihi",
        custom_arch_path: Optional[str] = None,
        thresholding_mode: str = "<=",
        spiking_mode: str = "lif",
        log_potential_trace: bool = False,
        log_message_trace: bool = True,
        cores_per_tile: int = 0,
    ):
        if spiking_mode != "lif":
            raise ValueError(
                f"SanafeRunner requires spiking_mode='lif'; got {spiking_mode!r}"
            )
        if arch_preset not in PRESETS:
            raise ValueError(
                f"unknown SANA-FE arch preset {arch_preset!r}; "
                f"expected one of {sorted(PRESETS.keys())}"
            )

        self.mapping = mapping
        self._preset = PRESETS[arch_preset]
        self.T = int(simulation_length)
        self.arch_preset = arch_preset
        self.custom_arch_path = custom_arch_path
        self.thresholding_mode = thresholding_mode
        self.log_potential_trace = log_potential_trace
        self.log_message_trace = log_message_trace
        self.cores_per_tile = cores_per_tile

        self._arch: Optional[Any] = None
        self._arch_name: str = "<unbuilt>"
        self._arch_geometry: Optional[SanafeArchGeometry] = None
        self._last_chip: Optional[Any] = None  # test hook


    def run(self, sample_input: np.ndarray, sample_index: int) -> SanafeRunRecord:
        """Run one sample through every hybrid stage."""
        if sample_input.ndim != 2 or sample_input.shape[0] != 1:
            raise ValueError(
                f"sample_input must have shape (1, D); got {sample_input.shape}"
            )

        has_neural = any(s.kind == "neural" for s in self.mapping.stages)
        if has_neural:
            sanafe = _sanafe()
            self._ensure_arch()
        else:
            sanafe = None

        state_buffer: Dict[int, np.ndarray] = {_RAW_INPUT_NODE_ID: sample_input}
        segments: Dict[int, SanafeSegmentRecord] = {}
        compute_outputs: Dict[int, np.ndarray] = {}

        out_scales = getattr(self.mapping, "node_activation_scales", {}) or {}
        in_scales = getattr(self.mapping, "node_input_activation_scales", out_scales) or {}

        from mimarsinan.chip_simulation.hybrid_stage_runner import run_hybrid_stages

        def _on_neural(stage_index, stage, state_buffer):
            segments[stage_index] = self._run_neural_stage(
                sanafe=sanafe,
                stage=stage,
                stage_index=stage_index,
                state_buffer=state_buffer,
            )

        def _on_compute(_stage_index, stage, state_buffer):
            op = stage.compute_op
            in_scale = in_scales.get(op.id, 1.0)
            out_scale = out_scales.get(op.id, 1.0)
            result = execute_compute_op_numpy(
                op, sample_input, state_buffer,
                in_scale=in_scale, out_scale=out_scale,
                dtype=_COMPUTE_DTYPE,
            )
            if hasattr(result, "detach"):
                result = result.detach().cpu().numpy()
            state_buffer[op.id] = np.asarray(result, dtype=_COMPUTE_DTYPE)
            compute_outputs[op.id] = state_buffer[op.id]

        run_hybrid_stages(
            self.mapping,
            state_buffer,
            on_neural=_on_neural,
            on_compute=_on_compute,
        )

        agg_e = SanafeEnergyBreakdown.zero()
        max_sim_time = 0.0
        total_spikes = 0
        total_packets = 0
        for seg in segments.values():
            agg_e = agg_e.add(seg.energy)
            if seg.sim_time_s > max_sim_time:
                max_sim_time = seg.sim_time_s
            total_spikes += seg.spikes
            total_packets += seg.packets_sent

        return SanafeRunRecord(
            arch_preset=self.arch_preset,
            arch_name=self._arch_name,
            sample_index=int(sample_index),
            T=self.T,
            segments=segments,
            compute_outputs=compute_outputs,
            aggregate_energy=agg_e,
            aggregate_sim_time_s=max_sim_time,
            total_spikes=total_spikes,
            total_packets=total_packets,
        )


    def _ensure_arch(self) -> None:
        """Lazily build the shared SANA-FE architecture."""
        if self._arch is not None:
            return
        spec = derive_arch_spec(
            self.mapping,
            preset_name=self.arch_preset,
            cores_per_tile=self.cores_per_tile,
        )
        self._arch_name = spec.name
        self._arch = build_architecture(
            spec,
            custom_arch_path=self.custom_arch_path,
            thresholding_mode=self.thresholding_mode,
        )
        self.cores_per_tile = int(spec.cores_per_tile_resolved)
        # Column-major tile coords: x = tile_id // mesh_height, y = tile_id % mesh_height.
        n_tiles = int(spec.n_tiles)
        mw = max(int(spec.mesh_width), 1)
        mh = max(int(spec.mesh_height), 1)
        tiles_xy = [[i // mh, i % mh] for i in range(n_tiles)]
        self._arch_geometry = SanafeArchGeometry(
            width=mw, height=mh, tiles_xy=tiles_xy,
        )


    def _run_neural_stage(
        self,
        *,
        sanafe: Any,
        stage: Any,
        stage_index: int,
        state_buffer: Dict[int, np.ndarray],
    ) -> SanafeSegmentRecord:
        hcm = stage.hard_core_mapping
        _output_sources = getattr(hcm, "output_sources", None)
        if _output_sources is not None and len(_output_sources) > 0:
            try:
                ChipLatency(hcm).calculate()
            except (RecursionError, ValueError):
                pass

        seg_input_rates = assemble_segment_input_numpy(
            stage.input_map, state_buffer, num_samples=1,
            dtype=_COMPUTE_DTYPE,
        )
        seg_in_size = int(seg_input_rates.shape[1])

        (net, core_to_group, core_input_neurons,
         core_always_on_neurons) = build_network_for_segment(
            self._arch, hcm,
            tile_offset=0, core_offset=0,
            cores_per_tile=self.cores_per_tile,
            simulation_length=self.T,
        )

        encoded = uniform_rate_encode(seg_input_rates, self.T)

        max_latency = max(
            (int(c.latency) if getattr(c, "latency", None) is not None else 0)
            for c in hcm.cores
        ) if hcm.cores else 0
        # +1: SANA-FE applies input spikes one cycle after emission.
        T_eff = self.T + max_latency + 1
        encoded_padded = encoded
        if T_eff > encoded.shape[2]:
            pad = np.zeros(
                (encoded.shape[0], encoded.shape[1], T_eff - encoded.shape[2]),
                dtype=encoded.dtype,
            )
            encoded_padded = np.concatenate([encoded, pad], axis=2)

        set_input_spike_trains(core_input_neurons, hcm, encoded_padded)
        set_always_on_spike_trains(core_always_on_neurons, T_eff)

        chip = sanafe.SpikingChip(self._arch)
        chip.load(net)
        results = chip.sim(
            T_eff,
            spike_trace=True,
            potential_trace=self.log_potential_trace,
            message_trace=self.log_message_trace,
        )
        self._last_chip = chip

        group_spike_counts = _spike_trace_to_group_counts(
            results.get("spike_trace", []),
            group_sizes=_group_name_to_size(net),
        )
        seg_raster = _pack_spike_trace_matrix(
            results.get("spike_trace", []), net.groups,
        )
        group_row_offsets = _group_row_offsets(net.groups)
        pkts_in, pkts_out = _per_core_packet_counts(
            results.get("message_trace"), n_cores=len(hcm.cores),
            cores_per_tile=self.cores_per_tile,
        )

        per_core_records: List[SanafeCoreRecord] = []
        for core_idx, core in enumerate(hcm.cores):
            used_neu = _used_neurons(core)
            used_ax = _used_axons(core)
            if used_neu <= 0:
                continue
            group = core_to_group.get(core_idx)
            if group is None:
                output_count = np.zeros(used_neu, dtype=np.int64)
            else:
                gsc = group_spike_counts.get(_group_name(group))
                output_count = (np.zeros(used_neu, dtype=np.int64)
                                if gsc is None
                                else np.asarray(gsc, dtype=np.int64)[:used_neu])
            input_count, n_always_on = self._derive_per_core_input_counts(
                core=core, seg_input_encoded=encoded,
                group_spike_counts=group_spike_counts,
                core_to_group=core_to_group,
                seg_raster=seg_raster,
                group_row_offsets=group_row_offsets,
                consumer_latency=int(core.latency) if core.latency is not None else 0,
                hcm=hcm,
            )
            core_raster: Optional[np.ndarray] = None
            if seg_raster is not None:
                group_name = _group_name(group) if group is not None else f"core{core_idx}"
                off = group_row_offsets.get(group_name)
                if off is not None and used_neu > 0:
                    core_raster = seg_raster[off:off + used_neu]
            per_core_records.append(SanafeCoreRecord(
                core_index=core_idx,
                n_neurons=used_neu,
                n_axons_used=used_ax,
                core_latency=int(getattr(core, "latency", 0) or 0),
                has_hardware_bias=core.hardware_bias is not None,
                n_always_on_axons=n_always_on,
                spikes_fired=int(output_count.sum()),
                input_spike_count=input_count,
                output_spike_count=output_count,
                energy=_per_core_energy_sanafe(
                    preset=self._preset,
                    n_neurons=used_neu,
                    T_active=self.T,
                    T_eff=T_eff,
                    incoming_spikes=int(input_count.sum()) if input_count.size else 0,
                    firings=int(output_count.sum()),
                    packets_in=int(pkts_in[core_idx]),
                    packets_out=int(pkts_out[core_idx]),
                ),
                spike_raster=core_raster,
            ))

        per_tile_records = self._aggregate_per_tile(
            per_core_records, results,
            message_trace=results.get("message_trace"),
        )

        last_active_fires = self._collect_last_active_fires(
            results.get("spike_trace", []),
            core_to_group=core_to_group,
            hcm=hcm,
        )

        seg_out_count = self._compute_seg_output_spike_count(
            stage.output_map, per_core_records,
            output_sources=getattr(hcm, "output_sources", None),
            T=self.T,
            hcm=hcm,
            last_active_fires=last_active_fires,
        )
        seg_in_count = encoded[0].sum(axis=1).astype(np.int64)

        seg_record = SanafeSegmentRecord(
            stage_index=stage_index,
            stage_name=stage.name,
            schedule_segment_index=getattr(stage, "schedule_segment_index", None),
            schedule_pass_index=getattr(stage, "schedule_pass_index", None),
            timesteps_executed=int(results.get("timesteps_executed", self.T)),
            sim_time_s=float(results.get("sim_time", 0.0)),
            energy=SanafeEnergyBreakdown.from_sanafe_dict(results["energy"]),
            spikes=int(results.get("spikes", 0)),
            packets_sent=int(results.get("packets_sent", 0)),
            neurons_updated=int(results.get("neurons_updated", 0)),
            neurons_fired=int(results.get("neurons_fired", 0)),
            seg_input_rates=seg_input_rates.astype(np.float32, copy=False),
            seg_input_spike_count=seg_in_count,
            seg_output_spike_count=seg_out_count,
            per_core=per_core_records,
            per_tile=per_tile_records,
            per_neuron_spike_trace=_pack_spike_trace_matrix(
                results.get("spike_trace", []), net.groups,
            ),
            per_neuron_potential_trace=_pack_potential_trace(
                results.get("potential_trace"),
            ),
            message_trace=_flatten_message_trace(results.get("message_trace")),
            arch_geometry=self._arch_geometry,
            noc_links=_aggregate_noc_links(
                results.get("message_trace"), self._arch_geometry,
            ),
            noc_link_load=_aggregate_noc_link_load(
                results.get("message_trace"), self._arch_geometry,
            ),
            cycle_energy=_compute_cycle_energy_breakdown(
                results.get("message_trace"),
                results.get("spike_trace", []),
                self._preset, hcm,
            ),
            cascade=_compute_cascade_timeline(
                results.get("spike_trace", []),
                net=net, hcm=hcm,
            ),
            critical_cores=_compute_critical_cores(
                results.get("spike_trace", []),
                results.get("message_trace"),
                net=net, hcm=hcm,
            ),
            connectivity=_compute_connectivity_edges(hcm),
            noc_traffic_per_cycle=_compute_noc_traffic_per_cycle(
                results.get("message_trace"),
            ),
        )

        seg_output_rates = (
            seg_out_count.astype(_COMPUTE_DTYPE) / np.asarray(self.T, dtype=_COMPUTE_DTYPE)
        ).reshape(1, -1)
        store_segment_output_numpy(stage.output_map, state_buffer, seg_output_rates)
        return seg_record


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
                counts[a] = T
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


    def _compute_seg_output_spike_count(
        self,
        output_map: List[Any],
        per_core_records: List[SanafeCoreRecord],
        *,
        output_sources: np.ndarray,
        T: int,
        hcm: Any,
        last_active_fires: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Gather per-output spike counts (in-window only, matching HCM)."""
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

        core_outputs: Dict[int, np.ndarray] = {
            rec.core_index: rec.output_spike_count for rec in per_core_records
        }

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
            cpt = len(per_core_records)  # one tile per segment
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


def _group_name(group: Any) -> str:
    """Return a NeuronGroup name (``get_name()`` or ``.name``)."""
    if hasattr(group, "get_name"):
        return group.get_name()
    return group.name


def _group_name_to_size(net: Any) -> Dict[str, int]:
    """Build ``{group_name: size}`` from dict- or list-shaped ``net.groups``."""
    groups = net.groups
    if isinstance(groups, dict):
        return {name: len(g) for name, g in groups.items()}
    return {_group_name(g): len(g) for g in groups}


def _per_core_energy_sanafe(
    *,
    preset: Dict[str, float],
    n_neurons: int,
    T_active: int,
    T_eff: int,
    incoming_spikes: int,
    firings: int,
    packets_in: int,
    packets_out: int,
) -> SanafeEnergyBreakdown:
    """Per-core energy mirroring SANA-FE ``sim_calculate_core_energy``."""
    if n_neurons <= 0:
        return SanafeEnergyBreakdown.zero()
    syn = float(preset.get("synapse_energy_j", 0.0)) * int(incoming_spikes)
    dend = float(preset.get("dendrite_energy_j", 0.0)) * n_neurons * int(T_eff)
    soma = (
        float(preset.get("soma_access_energy_j", 0.0)) * n_neurons * int(T_eff)
        + float(preset.get("soma_update_energy_j", 0.0)) * n_neurons * int(T_active)
        + float(preset.get("soma_spike_out_energy_j", 0.0)) * int(firings)
    )
    net = (
        float(preset.get("axon_in_energy_j", 0.0)) * int(packets_in)
        + float(preset.get("axon_out_energy_j", 0.0)) * int(packets_out)
    )
    return SanafeEnergyBreakdown(
        synapse_j=syn, dendrite_j=dend, soma_j=soma, network_j=net,
        total_j=syn + dend + soma + net,
    )


def _per_core_packet_counts(
    message_trace: Any, *, n_cores: int, cores_per_tile: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-core packet in/out counts from the message trace."""
    pkts_in = np.zeros(max(n_cores, 1), dtype=np.int64)
    pkts_out = np.zeros(max(n_cores, 1), dtype=np.int64)
    if not message_trace:
        return pkts_in, pkts_out
    cpt = max(int(cores_per_tile), 1)
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            st = int(ev.get("src_tile_id", -1))
            sc = int(ev.get("src_core_id", -1))
            dt = int(ev.get("dest_tile_id", -1))
            dc = int(ev.get("dest_core_id", -1))
            if st >= 0 and sc >= 0:
                gid = st * cpt + sc
                if 0 <= gid < n_cores:
                    pkts_out[gid] += 1
            if dt >= 0 and dc >= 0:
                gid = dt * cpt + dc
                if 0 <= gid < n_cores:
                    pkts_in[gid] += 1
    return pkts_in, pkts_out


def _energy_share(total: SanafeEnergyBreakdown, *, n_cores: int) -> SanafeEnergyBreakdown:
    """Split run-wide energy evenly across cores."""
    if n_cores <= 0:
        return SanafeEnergyBreakdown.zero()
    f = 1.0 / float(n_cores)
    return SanafeEnergyBreakdown(
        synapse_j=total.synapse_j * f,
        dendrite_j=total.dendrite_j * f,
        soma_j=total.soma_j * f,
        network_j=total.network_j * f,
        total_j=total.total_j * f,
    )


def _spike_trace_to_group_counts(
    spike_trace: list,
    *,
    group_sizes: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """Tally ``"<group>.<neuron>"`` spike-trace strings into per-group counts."""
    counts: Dict[str, np.ndarray] = {
        name: np.zeros(size, dtype=np.int64) for name, size in group_sizes.items()
    }
    for events in spike_trace:
        for event in events:
            s = str(event)
            if "." not in s:
                continue
            group_name, idx_str = s.rsplit(".", 1)
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            arr = counts.get(group_name)
            if arr is None or idx >= arr.size:
                continue
            arr[idx] += 1
    return counts


def _pack_spike_trace_matrix(
    spike_trace: list, groups: Any,
) -> Optional[np.ndarray]:
    """Spike trace → ``(sum_neurons, T)`` matrix; ``None`` if empty."""
    if not spike_trace:
        return None
    offsets: Dict[str, int] = {}
    cursor = 0
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        offsets[name] = cursor
        cursor += len(g)
    if cursor == 0:
        return None
    T = len(spike_trace)
    mat = np.zeros((cursor, T), dtype=np.uint8)
    for t, events in enumerate(spike_trace):
        for event in events:
            s = str(event)
            if "." not in s:
                continue
            group_name, idx_str = s.rsplit(".", 1)
            if group_name not in offsets:
                continue
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            row = offsets[group_name] + idx
            if 0 <= row < cursor:
                mat[row, t] = 1
    return mat


def _pack_potential_trace(potential_trace: Any) -> Optional[np.ndarray]:
    """Potential trace → ``(n_logged, T)``; ``None`` if empty or malformed."""
    if not potential_trace:
        return None
    arr = np.asarray(potential_trace, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr.T


def _flatten_message_trace(message_trace: Any) -> Optional[List[dict]]:
    """Flatten per-cycle message lists; drop placeholder entries."""
    if not message_trace:
        return None
    flat: List[dict] = []
    for events in message_trace:
        for ev in events:
            if isinstance(ev, dict):
                if ev.get("placeholder"):
                    continue
                flat.append({k: (float(v) if isinstance(v, float) else v)
                             for k, v in ev.items()})
    return flat or None


def _aggregate_noc_links(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLink]:
    """Aggregate cross-tile message trace into directed NoC links."""
    if not message_trace:
        return []
    bins: Dict[Tuple[int, int], Dict[str, int]] = {}
    src_coord: Dict[int, Tuple[int, int]] = {}
    dst_coord: Dict[int, Tuple[int, int]] = {}
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            src_t = int(ev.get("src_tile_id", -1))
            dst_t = int(ev.get("dest_tile_id", -1))
            if src_t < 0 or dst_t < 0 or src_t == dst_t:
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            src_coord[src_t] = (sx, sy)
            dst_coord[dst_t] = (dx, dy)
            slot = bins.setdefault(
                (src_t, dst_t),
                {"packets": 0, "spikes": 0, "hops": 0},
            )
            slot["packets"] += 1
            slot["spikes"] += int(ev.get("spikes", 0) or 0)
            slot["hops"] += int(ev.get("hops", 0) or 0)
    out: List[SanafeNocLink] = []
    for (src_t, dst_t), b in sorted(bins.items()):
        sx, sy = src_coord.get(src_t, (-1, -1))
        dx, dy = dst_coord.get(dst_t, (-1, -1))
        if geom is not None:
            if (sx < 0 or sy < 0) and 0 <= src_t < len(geom.tiles_xy):
                sx, sy = geom.tiles_xy[src_t]
            if (dx < 0 or dy < 0) and 0 <= dst_t < len(geom.tiles_xy):
                dx, dy = geom.tiles_xy[dst_t]
        out.append(SanafeNocLink(
            src_tile=src_t, dst_tile=dst_t,
            src_x=int(sx), src_y=int(sy),
            dst_x=int(dx), dst_y=int(dy),
            packet_count=int(b["packets"]),
            spike_count=int(b["spikes"]),
            total_hops=int(b["hops"]),
        ))
    return out


def _aggregate_noc_link_load(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLinkLoad]:
    """Per-mesh-edge packet load via XY routing."""
    if not message_trace:
        return []
    counts: Dict[Tuple[int, int, int, int], int] = {}
    for events in message_trace:
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                continue
            cx, cy = sx, sy
            step_x = 1 if dx > sx else -1 if dx < sx else 0
            step_y = 1 if dy > sy else -1 if dy < sy else 0
            while cx != dx:
                nx = cx + step_x
                k = (cx, cy, nx, cy)
                counts[k] = counts.get(k, 0) + 1
                cx = nx
            while cy != dy:
                ny = cy + step_y
                k = (cx, cy, cx, ny)
                counts[k] = counts.get(k, 0) + 1
                cy = ny
    out: List[SanafeNocLinkLoad] = []
    for (fx, fy, tx, ty), n in sorted(counts.items()):
        out.append(SanafeNocLinkLoad(
            from_x=fx, from_y=fy, to_x=tx, to_y=ty, packet_count=int(n),
        ))
    return out


def _compute_cycle_energy_breakdown(
    message_trace: Any,
    spike_trace: list,
    preset: Dict[str, float],
    hcm: Any,
) -> List[SanafeCycleEnergyPoint]:
    """Reconstruct per-cycle energy split from spike/message traces."""
    if not spike_trace and not message_trace:
        return []
    T_eff = max(
        len(spike_trace) if spike_trace else 0,
        len(message_trace) if message_trace else 0,
    )
    if T_eff <= 0:
        return []
    firings_per_cycle = np.zeros(T_eff, dtype=np.int64)
    if spike_trace:
        for c, evs in enumerate(spike_trace[:T_eff]):
            firings_per_cycle[c] = len(evs)
    pkt_per_cycle = np.zeros(T_eff, dtype=np.int64)
    hop_per_cycle = np.zeros(T_eff, dtype=np.int64)
    syn_per_cycle = np.zeros(T_eff, dtype=np.int64)
    dend_targets: List[set] = [set() for _ in range(T_eff)]
    if message_trace:
        for c, evs in enumerate(message_trace[:T_eff]):
            for ev in evs:
                if not isinstance(ev, dict) or ev.get("placeholder"):
                    continue
                pkt_per_cycle[c] += 1
                hop_per_cycle[c] += int(ev.get("hops", 0) or 0)
                syn_per_cycle[c] += int(ev.get("spikes", 0) or 0)
                dst_core = int(ev.get("dest_core_id", -1))
                dst_neuron = int(ev.get("dest_neuron_offset",
                                         ev.get("dest_axon_id", -1)))
                if dst_core >= 0 and dst_neuron >= 0:
                    dend_targets[c].add((dst_core, dst_neuron))
    total_live_neurons = 0
    if hcm is not None and getattr(hcm, "cores", None):
        for c in hcm.cores:
            np_used = int(c.neurons_per_core) - int(getattr(c, "available_neurons", 0))
            if np_used > 0:
                total_live_neurons += np_used
    out: List[SanafeCycleEnergyPoint] = []
    syn_e = float(preset.get("synapse_energy_j", 0.0))
    dend_e = float(preset.get("dendrite_energy_j", 0.0))
    soma_access_e = float(preset.get("soma_access_energy_j", 0.0))
    soma_update_e = float(preset.get("soma_update_energy_j", 0.0))
    soma_spike_e = float(preset.get("soma_spike_out_energy_j", 0.0))
    axon_in_e = float(preset.get("axon_in_energy_j", 0.0))
    axon_out_e = float(preset.get("axon_out_energy_j", 0.0))
    hop_e = float(preset.get("tile_hop_energy_j", 0.0))
    for c in range(T_eff):
        synapse_j = syn_e * int(syn_per_cycle[c])
        dendrite_j = dend_e * len(dend_targets[c])
        soma_j = (
            soma_access_e * total_live_neurons
            + soma_update_e * total_live_neurons
            + soma_spike_e * int(firings_per_cycle[c])
        )
        network_j = (
            axon_in_e * int(pkt_per_cycle[c])
            + axon_out_e * int(pkt_per_cycle[c])
            + hop_e * int(hop_per_cycle[c])
        )
        total = synapse_j + dendrite_j + soma_j + network_j
        out.append(SanafeCycleEnergyPoint(
            cycle=c,
            synapse_j=synapse_j, dendrite_j=dendrite_j,
            soma_j=soma_j, network_j=network_j, total_j=total,
        ))
    return out


def _build_neuron_to_core_map(
    net: Any, hcm: Any,
) -> Tuple[Dict[str, int], List[int]]:
    """Map spike-trace group names and global rows to HardCore indices."""
    group_to_core: Dict[str, int] = {}
    for core_idx, core in enumerate(hcm.cores):
        group_to_core[f"core{core_idx}"] = core_idx
    row_to_core: List[int] = []
    groups = net.groups
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        cidx = group_to_core.get(name, -1)
        row_to_core.extend([cidx] * len(g))
    return group_to_core, row_to_core


def _compute_cascade_timeline(
    spike_trace: list, *, net: Any, hcm: Any,
) -> List[SanafeCascadePoint]:
    """Bucket per-cycle firings by core latency depth."""
    if not spike_trace or hcm is None or not getattr(hcm, "cores", None):
        return []
    core_latency = [
        int(c.latency) if getattr(c, "latency", None) is not None else 0
        for c in hcm.cores
    ]
    group_to_core, _ = _build_neuron_to_core_map(net, hcm)
    out: List[SanafeCascadePoint] = []
    for cycle, evs in enumerate(spike_trace):
        bucket: Dict[int, int] = {}
        for ev in evs:
            s = str(ev)
            if "." not in s:
                continue
            gname, _ = s.rsplit(".", 1)
            core_idx = group_to_core.get(gname, -1)
            if core_idx < 0 or core_idx >= len(core_latency):
                continue
            d = core_latency[core_idx]
            bucket[d] = bucket.get(d, 0) + 1
        for d, n in sorted(bucket.items()):
            out.append(SanafeCascadePoint(cycle=int(cycle), depth=int(d), firings=int(n)))
    return out


def _compute_critical_cores(
    spike_trace: list, message_trace: Any, *, net: Any, hcm: Any,
) -> List[SanafeCriticalCore]:
    """Per-cycle core with highest firings + incoming spike load."""
    if hcm is None or not getattr(hcm, "cores", None):
        return []
    n_cores = len(hcm.cores)
    T_eff = max(
        len(spike_trace) if spike_trace else 0,
        len(message_trace) if message_trace else 0,
    )
    if T_eff <= 0 or n_cores == 0:
        return []
    group_to_core, _ = _build_neuron_to_core_map(net, hcm)
    fires = np.zeros((n_cores, T_eff), dtype=np.int64)
    if spike_trace:
        for cycle, evs in enumerate(spike_trace[:T_eff]):
            for ev in evs:
                s = str(ev)
                if "." not in s:
                    continue
                gname, _ = s.rsplit(".", 1)
                core_idx = group_to_core.get(gname, -1)
                if 0 <= core_idx < n_cores:
                    fires[core_idx, cycle] += 1
    incoming = np.zeros((n_cores, T_eff), dtype=np.int64)
    if message_trace:
        for cycle, evs in enumerate(message_trace[:T_eff]):
            for ev in evs:
                if not isinstance(ev, dict) or ev.get("placeholder"):
                    continue
                dt = int(ev.get("dest_tile_id", -1))
                dc = int(ev.get("dest_core_id", -1))
                if dt < 0 or dc < 0:
                    continue
                idx = dt * (1 + dc) + dc
                if 0 <= idx < n_cores:
                    incoming[idx, cycle] += int(ev.get("spikes", 0) or 0)
    score = fires + incoming
    out: List[SanafeCriticalCore] = []
    for cycle in range(T_eff):
        col = score[:, cycle]
        if col.sum() == 0:
            continue
        core_idx = int(col.argmax())
        out.append(SanafeCriticalCore(
            cycle=int(cycle), core_index=core_idx,
            event_count=int(col[core_idx]),
        ))
    return out


def _group_row_offsets(groups: Any) -> Dict[str, int]:
    """Group name → starting row (same order as ``_pack_spike_trace_matrix``)."""
    offsets: Dict[str, int] = {}
    cursor = 0
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        offsets[name] = cursor
        cursor += len(g)
    return offsets


def _compute_noc_traffic_per_cycle(message_trace: Any) -> List[List[List[int]]]:
    """Per-cycle ``[src_x, src_y, dst_x, dst_y, count]`` quintuples."""
    if not message_trace:
        return []
    out: List[List[List[int]]] = []
    for events in message_trace:
        bins: Dict[Tuple[int, int, int, int], int] = {}
        for ev in events:
            if not isinstance(ev, dict) or ev.get("placeholder"):
                continue
            sx = int(ev.get("src_x", -1))
            sy = int(ev.get("src_y", -1))
            dx = int(ev.get("dest_x", -1))
            dy = int(ev.get("dest_y", -1))
            if sx < 0 or sy < 0 or dx < 0 or dy < 0:
                continue
            if sx == dx and sy == dy:
                continue
            k = (sx, sy, dx, dy)
            bins[k] = bins.get(k, 0) + 1
        out.append([[fx, fy, tx, ty, n] for (fx, fy, tx, ty), n in bins.items()])
    return out


def _compute_connectivity_edges(hcm: Any) -> List[SanafeConnectivityEdge]:
    """Sum ``|weight|`` per ``(src_core, dst_core)`` from axon wiring."""
    if hcm is None or not getattr(hcm, "cores", None):
        return []
    bins: Dict[Tuple[int, int], Dict[str, float]] = {}
    for dst_idx, core in enumerate(hcm.cores):
        ax_per_core = int(core.axons_per_core)
        avail = int(getattr(core, "available_axons", 0))
        used_ax = max(ax_per_core - avail, 0)
        cm = getattr(core, "core_matrix", None)
        if cm is None or used_ax <= 0:
            continue
        for a in range(used_ax):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_input_", False) or getattr(src, "is_always_on_", False):
                continue
            src_core = int(src.core_)
            try:
                w_col = cm[a, :]
            except Exception:
                continue
            w_abs = float(np.abs(np.asarray(w_col, dtype=np.float64)).sum())
            if w_abs == 0.0:
                continue
            slot = bins.setdefault(
                (src_core, dst_idx), {"w": 0.0, "n": 0},
            )
            slot["w"] += w_abs
            slot["n"] += 1
    out: List[SanafeConnectivityEdge] = []
    for (src, dst), b in sorted(bins.items()):
        out.append(SanafeConnectivityEdge(
            src_core=int(src), dst_core=int(dst),
            weight_sum_abs=float(b["w"]), fan_count=int(b["n"]),
        ))
    return out
