"""Assemble SanafeSegmentRecord from chip.sim() results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from mimarsinan.chip_simulation.sanafe.presets import PerEventEnergy
    from mimarsinan.chip_simulation.sanafe.records import (
        SanafeArchGeometry,
        SanafeTileRecord,
    )

import mimarsinan.chip_simulation.sanafe.runner as _runner
from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import (
    NeuralSegmentResult,
    store_neural_segment_output,
)
from mimarsinan.spiking.segment_boundary import decode_segment_output
from mimarsinan.chip_simulation.sanafe.analysis import (
    _aggregate_noc_link_load,
    _aggregate_noc_links,
    _build_spike_capture_warning,
    _compute_cascade_timeline,
    _compute_critical_cores,
    _compute_cycle_energy_breakdown,
    _compute_noc_traffic_per_cycle,
    _compute_tile_packets_per_cycle,
    _compute_ttfs_activity_diagnostics,
    _flatten_message_trace,
    _group_name,
    _pack_potential_trace,
    _pack_spike_trace_matrix,
    _per_core_energy_sanafe,
    _read_ttfs_core_activations,
)
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeSegmentRecord,
)
from mimarsinan.mapping.support.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons

from .constants import _COMPUTE_DTYPE


class SanafeNeuralStageRecordMixin:
    if TYPE_CHECKING:
        # Host contract supplied by SanafeRunner + sibling mixins (declaration-only).
        _preset: PerEventEnergy
        _arch_geometry: Optional[SanafeArchGeometry]
        T: int
        spiking_mode: str
        ttfs_cycle_schedule: str

        def _derive_per_core_input_counts(self, *args: Any, **kwargs: Any) -> Tuple[np.ndarray, int]: ...
        def _aggregate_per_tile(self, *args: Any, **kwargs: Any) -> List[SanafeTileRecord]: ...
        def _collect_last_active_fires(self, *args: Any, **kwargs: Any) -> Dict[int, np.ndarray]: ...
        def _single_spike_ramp_outputs(self, *args: Any, **kwargs: Any) -> Dict[int, np.ndarray]: ...
        def _compute_seg_output_spike_count(self, *args: Any, **kwargs: Any) -> np.ndarray: ...

    def _finalize_neural_stage_record(
        self,
        *,
        stage,
        stage_index: int,
        state_buffer: Dict[int, np.ndarray],
        hcm,
        net,
        core_to_group,
        chip,
        results,
        seg_input_rates,
        encoded,
        T_eff: int,
        is_ttfs: bool,
        contract_ttfs_cores,
        contract_ttfs_seg_output,
        group_spike_counts,
        spike_parse_skipped,
        lif_spike_count,
        input_spikes_by_core,
        msg_summary,
        connectivity_edges,
        cross_tile_conn,
        chip_spike_count,
        seg_raster,
        group_row_offsets,
        pkts_in,
        pkts_out,
    ) -> SanafeSegmentRecord:
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

        from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

        cascade_override = None
        if is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule):
            cascade_override = self._single_spike_ramp_outputs(
                results.get("spike_trace", []),
                core_to_group=core_to_group, hcm=hcm, T_eff=T_eff,
            )

        seg_out_count = self._compute_seg_output_spike_count(
            stage.output_map, per_core_records,
            output_sources=getattr(hcm, "output_sources", None),
            T=self.T,
            hcm=hcm,
            last_active_fires=last_active_fires,
            core_output_override=cascade_override,
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

        from mimarsinan.chip_simulation.spiking_semantics import is_cascaded_ttfs

        cascade = is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)
        if cascade or not _runner.is_ttfs_spiking_mode(self.spiking_mode):
            # Cross-backend contract: cascaded counts may exceed T (lossy +latency),
            # matching HCM/nevresim — consistency across backends, not analytical agreement.
            seg_output_rates = decode_segment_output(
                seg_out_count, self.T, dtype=_COMPUTE_DTYPE,
            )
            store_neural_segment_output(
                "lif" if cascade else self.spiking_mode,
                stage.output_map,
                state_buffer,
                NeuralSegmentResult(inter_stage=seg_output_rates),
            )
        return seg_record

