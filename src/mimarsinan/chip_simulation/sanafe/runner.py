"""SANA-FE backend driver; sole caller of ``sanafe.SpikingChip``."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mimarsinan.chip_simulation._spike_encoding import uniform_rate_encode
from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    store_segment_output_numpy,
)
from mimarsinan.chip_simulation.hybrid_semantics import (
    NeuralSegmentResult,
    is_ttfs_spiking_mode,
    lif_inter_stage_from_spike_counts,
    store_neural_segment_output,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.chip_latency import ChipLatency
from mimarsinan.mapping.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.core_geometry import used_neurons as _used_neurons
from mimarsinan.mapping.spike_source_spans import compress_spike_sources

from .arch_synth import _sanafe, build_architecture, derive_arch_spec
from .net_synth import (
    apply_ttfs_preset_membranes,
    build_network_for_segment,
    set_always_on_spike_trains,
    set_input_spike_trains,
    set_ttfs_input_spike_trains,
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
        firing_mode: str = "Default",
        log_potential_trace: bool = False,
        log_message_trace: bool = True,
        cores_per_tile: int = 0,
    ):
        self.spiking_mode = str(spiking_mode)
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
        self.firing_mode = str(firing_mode)
        from mimarsinan.chip_simulation.firing_strategy import FiringStrategyFactory

        from mimarsinan.chip_simulation.spiking_semantics import require_spiking_mode_supported

        require_spiking_mode_supported(
            self.spiking_mode, backend="sanafe", context="SanafeRunner",
        )
        FiringStrategyFactory.from_config(
            {
                "firing_mode": self.firing_mode,
                "thresholding_mode": thresholding_mode,
                "spiking_mode": self.spiking_mode,
            }
        ).require_backend("sanafe")
        self.log_potential_trace = log_potential_trace
        self.log_message_trace = log_message_trace
        self.cores_per_tile = cores_per_tile

        self._arch: Optional[Any] = None
        self._arch_built_for_T: Optional[int] = None
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

        from mimarsinan.chip_simulation.hybrid_execution import resolve_stage_compute_scales
        from mimarsinan.chip_simulation.hybrid_stage_runner import run_hybrid_stages

        def _on_neural(stage_index, stage, state_buffer):
            segments[stage_index] = self._run_neural_stage(
                sanafe=sanafe,
                stage=stage,
                stage_index=stage_index,
                state_buffer=state_buffer,
            )

        def _on_compute(_stage_index, stage, state_buffer):
            from mimarsinan.chip_simulation.ttfs_executor import (
                run_ttfs_contract_compute_stage,
            )

            op = stage.compute_op
            assert op is not None
            if is_ttfs_spiking_mode(self.spiking_mode):
                result = run_ttfs_contract_compute_stage(
                    self.mapping, stage, state_buffer, sample_input,
                )
                compute_outputs[result.op_id] = result.output
            else:
                in_scale, out_scale = resolve_stage_compute_scales(
                    self.mapping, op.id, apply_ttfs=True,
                )
                result = execute_compute_op_numpy(
                    op, sample_input, state_buffer,
                    in_scale=in_scale, out_scale=out_scale,
                    dtype=_COMPUTE_DTYPE,
                )
                if hasattr(result, "detach"):
                    result = result.detach().cpu().numpy()
                out = np.asarray(result, dtype=_COMPUTE_DTYPE)
                state_buffer[op.id] = out
                compute_outputs[op.id] = out

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
        need_T = self.spiking_mode == "ttfs_quantized"
        if self._arch is not None:
            if not need_T or self._arch_built_for_T == self.T:
                return
            self._arch = None
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
            simulation_length=self.T,
        )
        self._arch_built_for_T = self.T if need_T else None
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
            firing_mode=self.firing_mode,
            spiking_mode=self.spiking_mode,
        )

        is_ttfs = self.spiking_mode in ("ttfs", "ttfs_quantized")

        max_latency = max(
            (int(c.latency) if getattr(c, "latency", None) is not None else 0)
            for c in hcm.cores
        ) if hcm.cores else 0
        # +1: SANA-FE applies input spikes one cycle after emission.
        T_eff = self.T + max_latency + 1

        contract_ttfs_cores: List[Any] = []
        contract_ttfs_seg_output: Optional[np.ndarray] = None
        logical_ttfs_result: Optional[NeuralSegmentResult] = None

        if is_ttfs:
            from mimarsinan.chip_simulation.ttfs_executor import (
                run_ttfs_contract_neural_stage,
            )

            contract_stage = run_ttfs_contract_neural_stage(
                self.mapping,
                stage,
                stage_index,
                state_buffer,
                simulation_length=self.T,
                spiking_mode=self.spiking_mode,
            )
            logical_ttfs_result = contract_stage.neural_result
            contract_ttfs_cores = list(contract_stage.segment_record.cores)
            contract_ttfs_seg_output = contract_stage.segment_record.seg_output
            membrane_V = contract_stage.membrane_voltages
            seg_input_rates = contract_stage.seg_input
            apply_ttfs_preset_membranes(
                core_to_group, hcm, membrane_V,
                spiking_mode=self.spiking_mode,
                simulation_length=self.T,
                firing_mode=self.firing_mode,
            )
            set_ttfs_input_spike_trains(
                core_input_neurons, hcm, seg_input_rates, self.T,
            )
            from mimarsinan.chip_simulation.ttfs_encoding import ttfs_latched_spike_train

            encoded = ttfs_latched_spike_train(
                seg_input_rates.astype(np.float64), self.T,
            ).astype(np.float32)
            encoded_padded = encoded
            if T_eff > encoded.shape[2]:
                pad = np.zeros(
                    (encoded.shape[0], encoded.shape[1], T_eff - encoded.shape[2]),
                    dtype=encoded.dtype,
                )
                encoded_padded = np.concatenate([encoded, pad], axis=2)
        else:
            encoded = uniform_rate_encode(seg_input_rates, self.T)
            encoded_padded = encoded
            if T_eff > encoded.shape[2]:
                pad = np.zeros(
                    (encoded.shape[0], encoded.shape[1], T_eff - encoded.shape[2]),
                    dtype=encoded.dtype,
                )
                encoded_padded = np.concatenate([encoded, pad], axis=2)
            set_input_spike_trains(core_input_neurons, hcm, encoded_padded)
        set_always_on_spike_trains(
            core_always_on_neurons, T_eff, spiking_mode=self.spiking_mode,
        )

        chip = sanafe.SpikingChip(self._arch)
        chip.load(net)
        need_potential_trace = is_ttfs or self.log_potential_trace
        results = chip.sim(
            T_eff,
            spike_trace=True,
            potential_trace=need_potential_trace,
            message_trace=self.log_message_trace,
        )
        self._last_chip = chip

        group_spike_counts, spike_parse_skipped = _spike_trace_to_group_counts(
            results.get("spike_trace", []),
            group_sizes=_group_name_to_size(net),
        )
        lif_spike_count, _input_spike_total = _lif_and_input_spike_totals(
            group_spike_counts,
        )
        input_spikes_by_core = _input_spikes_per_core(group_spike_counts)
        msg_summary = _summarize_message_trace(results.get("message_trace"))
        connectivity_edges = _compute_connectivity_edges(hcm)
        cross_tile_conn = _count_cross_tile_connectivity_edges(
            connectivity_edges, cores_per_tile=self.cores_per_tile,
        )
        chip_spike_count = int(results.get("spikes", 0))
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
            output_activation = None
            if is_ttfs and group is not None:
                output_activation = _read_ttfs_core_activations(
                    chip, core_idx, used_neu, results,
                )

            per_core_records.append(SanafeCoreRecord(
                core_index=core_idx,
                n_neurons=used_neu,
                n_axons_used=used_ax,
                core_latency=int(getattr(core, "latency", 0) or 0),
                has_hardware_bias=core.hardware_bias is not None,
                n_always_on_axons=n_always_on,
                spikes_fired=int(output_count.sum()),
                input_neuron_spikes_fired=int(
                    input_spikes_by_core.get(core_idx, 0),
                ),
                input_spike_count=input_count,
                output_spike_count=output_count,
                output_activation=output_activation,
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

        ttfs_diag = (
            _compute_ttfs_activity_diagnostics(
                contract_ttfs_cores, per_core_records,
            )
            if is_ttfs else {
                "ttfs_contract_active_cores": 0,
                "ttfs_hardware_active_cores": 0,
                "ttfs_event_active_cores": 0,
                "ttfs_activation_event_mismatch_count": 0,
            }
        )
        spike_warning = _build_spike_capture_warning(
            chip_spike_count=chip_spike_count,
            lif_spike_count=lif_spike_count,
            input_path_packets=msg_summary["input_path_packets"],
            spike_trace_parse_skipped=spike_parse_skipped,
            ttfs_hardware_active=ttfs_diag["ttfs_hardware_active_cores"],
            ttfs_event_active=ttfs_diag["ttfs_event_active_cores"],
            ttfs_mismatch_count=ttfs_diag["ttfs_activation_event_mismatch_count"],
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
            connectivity=connectivity_edges,
            noc_traffic_per_cycle=_compute_noc_traffic_per_cycle(
                results.get("message_trace"),
            ),
            tile_packets_per_cycle=_compute_tile_packets_per_cycle(
                results.get("message_trace"),
            ),
            inter_tile_packets=msg_summary["inter_tile_packets"],
            intra_tile_packets=msg_summary["intra_tile_packets"],
            input_path_packets=msg_summary["input_path_packets"],
            cross_tile_connectivity_edges=cross_tile_conn,
            chip_spike_count=chip_spike_count,
            lif_spike_count=lif_spike_count,
            spike_trace_parse_skipped=spike_parse_skipped,
            spike_capture_warning=spike_warning,
            mapped_cross_tile_axons=cross_tile_conn,
            ttfs_contract_active_cores=ttfs_diag["ttfs_contract_active_cores"],
            ttfs_hardware_active_cores=ttfs_diag["ttfs_hardware_active_cores"],
            ttfs_event_active_cores=ttfs_diag["ttfs_event_active_cores"],
            ttfs_activation_event_mismatch_count=ttfs_diag[
                "ttfs_activation_event_mismatch_count"
            ],
            contract_ttfs_cores=contract_ttfs_cores,
            contract_ttfs_seg_output=contract_ttfs_seg_output,
        )

        if not is_ttfs_spiking_mode(self.spiking_mode):
            seg_output_rates = lif_inter_stage_from_spike_counts(
                seg_out_count, self.T, dtype=_COMPUTE_DTYPE,
            )
            store_neural_segment_output(
                self.spiking_mode,
                stage.output_map,
                state_buffer,
                NeuralSegmentResult(inter_stage=seg_output_rates),
            )
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


_CORE_GROUP_RE = re.compile(r"^core(\d+)$")
_LIF_SPIKE_GROUP_RE = re.compile(r"^core(\d+)$")
_INPUT_SPIKE_GROUP_RE = re.compile(r"^core(\d+)_(in|on)$")

from mimarsinan.chip_simulation.sanafe.runner_analysis import *  # noqa: F403
