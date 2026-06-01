"""SANA-FE neural stage execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import mimarsinan.chip_simulation.sanafe.runner as _runner
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import assemble_segment_input_numpy
from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import (
    NeuralSegmentResult,
    lif_inter_stage_from_spike_counts,
    store_neural_segment_output,
)
from mimarsinan.chip_simulation.sanafe.analysis import (
    _build_spike_capture_warning,
    _compute_connectivity_edges,
    _compute_critical_cores,
    _compute_cycle_energy_breakdown,
    _compute_noc_traffic_per_cycle,
    _compute_ttfs_activity_diagnostics,
    _count_cross_tile_connectivity_edges,
    _flatten_message_trace,
    _group_name,
    _group_name_to_size,
    _group_row_offsets,
    _lif_and_input_spike_totals,
    _input_spikes_per_core,
    _pack_potential_trace,
    _pack_spike_trace_matrix,
    _per_core_energy_sanafe,
    _per_core_packet_counts,
    _read_ttfs_core_activations,
    _spike_trace_to_group_counts,
    _summarize_message_trace,
    _aggregate_noc_link_load,
    _aggregate_noc_links,
    _compute_cascade_timeline,
)
from mimarsinan.chip_simulation.sanafe.net_synth import apply_ttfs_preset_membranes
from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)
from mimarsinan.mapping.latency.chip import ChipLatency
from mimarsinan.mapping.support.core_geometry import used_axons as _used_axons
from mimarsinan.mapping.support.core_geometry import used_neurons as _used_neurons

from .constants import _COMPUTE_DTYPE


class SanafeNeuralStageMixin:
    """Run one neural hybrid stage through SANA-FE."""

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
         core_always_on_neurons) = _runner.build_network_for_segment(
            self._arch, hcm,
            tile_offset=0, core_offset=0,
            cores_per_tile=self.cores_per_tile,
            simulation_length=self.T,
            firing_mode=self.firing_mode,
            spiking_mode=self.spiking_mode,
        )

        from mimarsinan.chip_simulation.spiking_semantics import requires_ttfs_firing

        is_ttfs = requires_ttfs_firing(self.spiking_mode)

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
            from mimarsinan.chip_simulation.ttfs.ttfs_executor import (
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
            _runner.set_ttfs_input_spike_trains(
                core_input_neurons, hcm, seg_input_rates, self.T,
            )
            from mimarsinan.chip_simulation.ttfs.ttfs_encoding import ttfs_latched_spike_train

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
            encoded = self._behavior.encode_segment_input(seg_input_rates, self.T)
            encoded_padded = encoded
            if T_eff > encoded.shape[2]:
                pad = np.zeros(
                    (encoded.shape[0], encoded.shape[1], T_eff - encoded.shape[2]),
                    dtype=encoded.dtype,
                )
                encoded_padded = np.concatenate([encoded, pad], axis=2)
            _runner.set_input_spike_trains(core_input_neurons, hcm, encoded_padded)
        _runner.set_always_on_spike_trains(
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

        return self._finalize_neural_stage_record(
            stage=stage,
            stage_index=stage_index,
            state_buffer=state_buffer,
            hcm=hcm,
            net=net,
            core_to_group=core_to_group,
            chip=chip,
            results=results,
            seg_input_rates=seg_input_rates,
            encoded=encoded,
            T_eff=T_eff,
            is_ttfs=is_ttfs,
            contract_ttfs_cores=contract_ttfs_cores,
            contract_ttfs_seg_output=contract_ttfs_seg_output,
            group_spike_counts=group_spike_counts,
            spike_parse_skipped=spike_parse_skipped,
            lif_spike_count=lif_spike_count,
            input_spikes_by_core=input_spikes_by_core,
            msg_summary=msg_summary,
            connectivity_edges=connectivity_edges,
            cross_tile_conn=cross_tile_conn,
            chip_spike_count=chip_spike_count,
            seg_raster=seg_raster,
            group_row_offsets=group_row_offsets,
            pkts_in=pkts_in,
            pkts_out=pkts_out,
        )
