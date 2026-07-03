"""SANA-FE neural stage execution."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

import mimarsinan.chip_simulation.sanafe.runner as _runner
from mimarsinan.chip_simulation.hybrid_run.hybrid_execution import assemble_segment_input_numpy
from mimarsinan.chip_simulation.hybrid_run.hybrid_semantics import NeuralSegmentResult
from mimarsinan.chip_simulation.sanafe.analysis import (
    _compute_connectivity_edges,
    _count_cross_tile_connectivity_edges,
    _group_name_to_size,
    _group_row_offsets,
    _lif_and_input_spike_totals,
    _input_spikes_per_core,
    _pack_spike_trace_matrix,
    _per_core_packet_counts,
    _spike_trace_to_group_counts,
    _summarize_message_trace,
)
from mimarsinan.chip_simulation.sanafe.net_synth import apply_ttfs_preset_membranes
from mimarsinan.chip_simulation.sanafe.records import SanafeSegmentRecord
from mimarsinan.mapping.latency.chip import ChipLatency

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
            ttfs_cycle_schedule=self.ttfs_cycle_schedule,
        )

        from mimarsinan.chip_simulation.spiking_semantics import (
            is_cascaded_ttfs,
            is_synchronized_ttfs,
            requires_ttfs_firing,
        )

        is_cascade = is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)
        is_ttfs = requires_ttfs_firing(self.spiking_mode) and not is_cascade
        is_cycle = is_synchronized_ttfs(self.spiking_mode, self.ttfs_cycle_schedule)

        max_latency = max(
            (int(c.latency) if getattr(c, "latency", None) is not None else 0)
            for c in hcm.cores
        ) if hcm.cores else 0
        if is_cycle:
            from mimarsinan.chip_simulation.ttfs.ttfs_cycle_genuine import latency_groups

            num_groups, _ = latency_groups(
                [getattr(c, "latency", None) for c in hcm.cores]
            )
            T_eff = (num_groups + 1) * self.T
        elif is_cascade:
            # Cascaded T_eff must equal HCM's ``ChipLatency + T`` (full ChipLatency,
            # not max core latency) + 1 for SANA-FE's one-cycle input delivery delay.
            try:
                chip_latency = int(ChipLatency(hcm).calculate())
            except (RecursionError, ValueError):
                chip_latency = max_latency
            T_eff = self.T + chip_latency + 1
        else:
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
                quantize_input_to_ttfs_grid=is_cycle,
            )
            logical_ttfs_result = contract_stage.neural_result
            contract_ttfs_cores = list(contract_stage.segment_record.cores)
            contract_ttfs_seg_output = contract_stage.segment_record.seg_output
            membrane_V = contract_stage.membrane_voltages
            seg_input_rates = contract_stage.seg_input
            if not is_cycle:
                apply_ttfs_preset_membranes(
                    core_to_group, hcm, membrane_V,
                    spiking_mode=self.spiking_mode,
                    simulation_length=self.T,
                    firing_mode=self.firing_mode,
                )
            if not is_cycle:
                _runner.set_ttfs_input_spike_trains(
                    core_input_neurons, hcm, seg_input_rates, self.T,
                )
            from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
                ttfs_latched_spike_train,
                ttfs_single_spike_train,
            )

            _encode = ttfs_single_spike_train if is_cycle else ttfs_latched_spike_train
            encoded = _encode(
                seg_input_rates.astype(np.float64), self.T,
            ).astype(np.float32)
            encoded_padded = encoded
            if T_eff > encoded.shape[2]:
                pad = np.zeros(
                    (encoded.shape[0], encoded.shape[1], T_eff - encoded.shape[2]),
                    dtype=encoded.dtype,
                )
                encoded_padded = np.concatenate([encoded, pad], axis=2)
            if is_cycle:
                _runner.set_input_spike_trains(core_input_neurons, hcm, encoded_padded)
        elif is_cascade:
            from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
                ttfs_single_spike_train,
            )

            encoded = ttfs_single_spike_train(
                seg_input_rates.astype(np.float64), self.T,
            ).astype(np.float32)
            encoded_padded = encoded
            if T_eff > encoded.shape[2]:
                pad = np.zeros(
                    (encoded.shape[0], encoded.shape[1], T_eff - encoded.shape[2]),
                    dtype=encoded.dtype,
                )
                encoded_padded = np.concatenate([encoded, pad], axis=2)
            _runner.set_input_spike_trains(core_input_neurons, hcm, encoded_padded)
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
        core_latencies = {
            i: (int(c.latency) if getattr(c, "latency", None) is not None else 0)
            for i, c in enumerate(hcm.cores)
        } if is_cascade else None
        _runner.set_always_on_spike_trains(
            core_always_on_neurons, T_eff, spiking_mode=self.spiking_mode,
            core_latencies=core_latencies,
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
