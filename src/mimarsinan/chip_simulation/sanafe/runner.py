"""SANA-FE backend driver.

``SanafeRunner.run(sample_input, sample_index)`` walks a
``HybridHardCoreMapping`` end-to-end: each neural stage is built as a
SANA-FE ``Network`` on the shared synthesised ``Architecture``, run
through ``SpikingChip.sim()``, and reduced to a
:class:`SanafeSegmentRecord`; each compute stage runs host-side via
``hybrid_execution.execute_compute_op_numpy``.

The runner is the only module that touches ``sanafe.SpikingChip``.

Single-sample only at this stage: the parity-gate use case feeds one
deterministic test sample at a time.  Higher sample counts are handled
by the pipeline step looping over samples.
"""

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

# Match ``hybrid_core_flow._COMPUTE_DTYPE``.  Compute-op arithmetic and the
# segment-input assembly run in float64 so the rate-encoder sees the same
# value HCM does — float32 here causes ±1 spike-count drift at quantization
# boundaries and breaks the per-axon parity check.
_COMPUTE_DTYPE: np.dtype = np.float64


class SanafeRunner:
    """Run one hybrid mapping sample through SANA-FE and collect rich stats."""

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
        # Captured at ``_ensure_arch`` time and threaded into every
        # ``SanafeSegmentRecord`` so the GUI floorplan can place cores
        # on the actual mesh without re-deriving it from the YAML.
        self._arch_geometry: Optional[SanafeArchGeometry] = None
        # White-box hook for tests; runner sets this on each chip.sim() call.
        self._last_chip: Optional[Any] = None

    # ------------------------------------------------------------------ run

    def run(self, sample_input: np.ndarray, sample_index: int) -> SanafeRunRecord:
        """Run one sample through every stage, returning a rich record."""
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

        for stage_index, stage in enumerate(self.mapping.stages):
            if stage.kind == "neural":
                segments[stage_index] = self._run_neural_stage(
                    sanafe=sanafe,
                    stage=stage,
                    stage_index=stage_index,
                    state_buffer=state_buffer,
                )
            elif stage.kind == "compute":
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
                # Preserve float64 in the state buffer — HCM keeps compute-op
                # outputs at ``_COMPUTE_DTYPE`` precision, and downstream
                # rate encoding is sensitive to ±1 ulp differences at
                # quantization boundaries.
                state_buffer[op.id] = np.asarray(result, dtype=_COMPUTE_DTYPE)
                compute_outputs[op.id] = state_buffer[op.id]
            else:
                raise ValueError(f"Unknown hybrid stage kind: {stage.kind!r}")

        # Aggregate.
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

    # ------------------------------------------------------------------ arch

    def _ensure_arch(self) -> None:
        """Lazily synthesise the shared SANA-FE architecture from the mapping."""
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
        # Lift the auto-resolved ``cores_per_tile`` from arch_synth so
        # ``net_synth.build_network_for_segment`` packs cores into tiles
        # consistently with what the YAML synthesiser produced (passing
        # ``0`` to ``_pack_tile_index`` would route every core to tile 0
        # and IndexError once the spec actually has >1 tile).
        self.cores_per_tile = int(spec.cores_per_tile_resolved)
        # Capture geometry: ``arch_synth`` lays tiles out on a 2D mesh
        # of ``mesh_width × mesh_height``.  **Column-major** tile
        # ordering matches SANA-FE's own ``Architecture::calculate_tile_coordinates``
        # (``sana_fe/src/arch.cpp:84``), which assigns
        # ``x = tile_id // noc_height_in_tiles, y = tile_id %
        # noc_height_in_tiles``.  Getting this wrong is why the NoC
        # overlay was misaligned — message_trace.src_x/src_y carried
        # SANA-FE's column-major coords while the GUI rendered tiles
        # in row-major order.
        n_tiles = int(spec.n_tiles)
        mw = max(int(spec.mesh_width), 1)
        mh = max(int(spec.mesh_height), 1)
        tiles_xy = [[i // mh, i % mh] for i in range(n_tiles)]
        self._arch_geometry = SanafeArchGeometry(
            width=mw, height=mh, tiles_xy=tiles_xy,
        )

    # --------------------------------------------------------------- segment

    def _run_neural_stage(
        self,
        *,
        sanafe: Any,
        stage: Any,
        stage_index: int,
        state_buffer: Dict[int, np.ndarray],
    ) -> SanafeSegmentRecord:
        hcm = stage.hard_core_mapping
        # Normalise per-core latencies to the local (per-segment) depth.
        # ``build_hybrid_hard_core_mapping`` stores GLOBAL depths from the
        # combined IR — but HCM's ``_run_neural_segment_rate`` calls
        # ``ChipLatency(mapping).calculate()`` first thing (line 504),
        # which overwrites each core's ``latency`` with the depth measured
        # from THIS segment's outputs.  We must do the same so our active
        # window and ``max_latency`` see the local values HCM windowed on.
        # The test fakes use ``SimpleNamespace`` HCMs without
        # ``output_sources`` (or with synthetic cores that contain cycles
        # in their cross-core references — ``ChipLatency`` recurses
        # infinitely on those).  Skip the recalc in either case; the
        # tests set ``latency`` directly on their fake cores.
        _output_sources = getattr(hcm, "output_sources", None)
        if _output_sources is not None and len(_output_sources) > 0:
            try:
                ChipLatency(hcm).calculate()
            except (RecursionError, ValueError):
                # Synthetic test fixtures may produce dependency cycles;
                # fall back to whatever latencies the fixture supplied.
                pass

        seg_input_rates = assemble_segment_input_numpy(
            stage.input_map, state_buffer, num_samples=1,
            dtype=_COMPUTE_DTYPE,
        )
        seg_in_size = int(seg_input_rates.shape[1])

        # Build the per-segment SANA-FE network.  Each HardCore in the HCM
        # maps to one SANA-FE core; input neurons live on the consuming
        # core (see ``sanafe_per_core_input_neurons`` memory).
        # Passing ``simulation_length=self.T`` enables the per-neuron
        # ``active_start`` / ``active_length`` gate so each core only
        # integrates for exactly ``T`` cycles inside its
        # ``[core.latency, T + core.latency)`` window — matching HCM's
        # per-core recording window even when the chip runs longer.
        (net, core_to_group, core_input_neurons,
         core_always_on_neurons) = build_network_for_segment(
            self._arch, hcm,
            tile_offset=0, core_offset=0,
            cores_per_tile=self.cores_per_tile,
            simulation_length=self.T,
        )

        # Rate-encode the segment input as a (1, D, T) binary tensor.  Each
        # (core_idx, axon_idx) input neuron gets the spike train for its
        # logical input index ``hcm.cores[core_idx].axon_sources[axon_idx].neuron_``.
        encoded = uniform_rate_encode(seg_input_rates, self.T)            # (1, D, T)

        # ----- Per-HardCore latency cascade -----
        # HCM simulates ``T + max_core_latency`` cycles per segment so that
        # multi-depth cascades have time to flush all the way through
        # (``_run_neural_segment_rate`` line 504-505).  Each core records
        # spikes only inside its active window ``[core.latency, T +
        # core.latency)``.
        #
        # SANA-FE is synchronous: each tick advances every core by one
        # cycle, with cross-core sources read from the previous tick's
        # buffer.  That naturally implements the per-depth latency cascade
        # — a depth-d core only sees non-zero input starting on tick d —
        # but it still needs ``T + max_latency`` ticks to give the
        # deepest cores their full ``T`` integration window.
        #
        # We therefore extend the chip simulation to ``T + max_latency``
        # ticks and pad the input spike trains with trailing zeros.
        # Always-on neurons keep firing every tick (they're constant
        # bias sources, not gated).
        max_latency = max(
            (int(c.latency) if getattr(c, "latency", None) is not None else 0)
            for c in hcm.cores
        ) if hcm.cores else 0
        # +1 accounts for SANA-FE's input→synapse pipeline delay: an input
        # neuron's spike at sim_time t is processed by the consumer's
        # synapse only at sim_time t+1, so the first downstream integration
        # happens one cycle later than in HCM.  Without this offset, the
        # depth-0 cores miss their last input cycle and underfire.
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

        # Build and run the chip.
        chip = sanafe.SpikingChip(self._arch)
        chip.load(net)
        results = chip.sim(
            T_eff,
            spike_trace=True,
            potential_trace=self.log_potential_trace,
            message_trace=self.log_message_trace,
        )
        self._last_chip = chip

        # Parse spike_trace → per-group spike counts.  Each timestep entry
        # is a list of "group_name.neuron_idx" strings; we tally per group.
        # SANA-FE 2.1.1's ``net.groups`` is a dict[name, NeuronGroup]; the
        # fake-network test surface mirrors that as a list of objects
        # exposing ``get_name()``.
        group_spike_counts = _spike_trace_to_group_counts(
            results.get("spike_trace", []),
            group_sizes=_group_name_to_size(net),
        )
        # Pre-compute the segment-wide spike raster once; per-core
        # records slice their LIF rows out of it for the click-to-raster
        # mini-view (#8).  ``None`` when the trace was empty.
        seg_raster = _pack_spike_trace_matrix(
            results.get("spike_trace", []), net.groups,
        )
        group_row_offsets = _group_row_offsets(net.groups)
        # Per-core packet counts straight from the message trace — feed
        # into SANA-FE's exact ``axon_in_energy`` / ``axon_out_energy``
        # accounting (one packet × per-event constant, same formula as
        # ``chip.cpp:sim_calculate_core_energy``).
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
            # Slice this core's LIF rows out of the segment raster.
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
                # Reproduce SANA-FE's exact per-core energy formula
                # (``chip.cpp:sim_calculate_core_energy`` +
                # ``pipeline.hpp:calculate_*_default_energy_latency``):
                # synapse, dendrite, soma each multiply their per-event
                # YAML constant by the count of process() calls.  Counts
                # come from the per-core counters we already track or
                # derive directly from the trace (axon packets).
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

        # Build a per-(core, neuron) lookup of "did this neuron fire on the
        # last cycle of its core's active window?" — needed by HCM-replica
        # seg_output gather to add stale-buffer accumulations.  See the
        # docstring of ``_compute_seg_output_spike_count`` for the why.
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

        # Store segment output rates into the state buffer for downstream
        # stages.  Keep at ``_COMPUTE_DTYPE`` (float64) so a downstream
        # compute op gathers the same precision HCM does — otherwise
        # ``np.round(rate * T)`` in the rate encoder can drift by ±1 ulp
        # at quantization boundaries and break per-axon parity.
        seg_output_rates = (
            seg_out_count.astype(_COMPUTE_DTYPE) / np.asarray(self.T, dtype=_COMPUTE_DTYPE)
        ).reshape(1, -1)
        store_segment_output_numpy(stage.output_map, state_buffer, seg_output_rates)
        return seg_record

    # ----------------------------------------------------- per-core input

    def _derive_per_core_input_counts(
        self,
        *,
        core: Any,
        seg_input_encoded: np.ndarray,            # (1, seg_in_size, T)
        group_spike_counts: Dict[str, np.ndarray],
        core_to_group: Dict[int, Any],
        seg_raster: Optional[np.ndarray] = None,  # (sum_neurons, T_eff) per-cycle firing
        group_row_offsets: Optional[Dict[str, int]] = None,
        consumer_latency: int = 0,
        hcm: Any = None,
    ) -> Tuple[np.ndarray, int]:
        """Walk this core's ``axon_sources``; build (input_spike_count, n_always_on).

        For cross-core axons, the count must match HCM's
        ``record_in_t`` accumulator (``hybrid_core_flow.
        _run_neural_segment_rate``): HCM sums ``input_signals`` only
        during the *consumer's* active window ``[cons_lat, cons_lat+T)``,
        where ``input_signals`` reads ``buffers[src_core]`` — and HCM's
        ``buffers`` follows the chip's axon-held semantics:

          * cycles before the source's active window: ``0`` (buffer
            never written)
          * cycles inside ``[src_lat, src_lat+T)``: the source's actual
            firing for that cycle (the buffer was just written)
          * cycles past ``src_lat+T-1``: the source's *last* active
            firing, held until next reset

        The previous implementation used the source's total firing
        count via ``group_spike_counts``, ignoring time entirely. This
        diverged from HCM whenever the consumer's read window
        ``[cons_lat-1, cons_lat+T-1)`` didn't fully cover the source's
        active window — most visibly when a shallow consumer reads a
        deep source (consumer_lat < src_lat), where HCM gives 0 and
        the buggy SANA-FE recording attributed the source's full
        in-window firing count instead.
        """
        used_ax = _used_axons(core)
        if used_ax <= 0:
            return np.zeros(0, dtype=np.int64), 0
        counts = np.zeros(used_ax, dtype=np.int64)
        n_always_on = 0
        T = self.T
        # SANA-FE's ``seg_raster`` time axis is shifted +1 from HCM cycles
        # (``T_eff = T + max_latency + 1`` accounts for the input→synapse
        # pipeline delay). A neuron at HCM ``latency=L`` therefore fires
        # during SANA-FE sim_times ``[L+1, L+T+1)``. Consumer integration at
        # HCM cycle ``cons_lat+k`` corresponds to SANA-FE sim_time
        # ``cons_lat+k+1`` reading the source's state at sim_time
        # ``cons_lat+k``. The consumer's read window in SANA-FE time is
        # therefore ``[cons_lat, cons_lat+T)`` — T reads.
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
                # Input neurons emit at SANA-FE sim_times ``[0, T)``;
                # the consumer's read window in SANA-FE time covers
                # ``[cons_lat, cons_lat+T)``. Count input spikes in the
                # intersection. For shallow consumers (cons_lat == 0)
                # this reduces to the full input row over ``[0, T)``,
                # matching the pre-refactor behaviour.
                k = int(src.neuron_)
                lo = min(consumer_latency, T)
                hi = min(consumer_latency + T, T)
                if hi > lo:
                    counts[a] = int(seg_input_encoded[0, k, lo:hi].sum())
                continue
            # Cross-core: count firings the consumer's axon actually
            # observes across the consumer's read window
            # ``[cons_lat-1, cons_lat+T-1)``, mirroring HCM's per-cycle
            # ``buffers[src]`` accounting (with axon-held semantics past
            # the source's active window).
            src_core = int(src.core_)
            src_neuron = int(src.neuron_)
            if seg_raster is None or group_row_offsets is None or hcm is None:
                # Fallback: pre-fix behaviour for harness tests that
                # don't supply the trace. Real pipeline runs always
                # pass these.
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
            # Source at HCM ``latency=L`` fires during SANA-FE sim_times
            # ``[L+1, L+T+1)`` (the +1 offset that ``T_eff`` accounts for).
            # HCM's buffer holds the source's last active firing past that
            # window; replicate that here by reading the last-active
            # sim_time ``L+T`` for any consumer cycle ``>= L+T+1``.
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

    # --------------------------------------------------------- segment output

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
        """Gather per-output spike counts matching HCM's per-source window.

        HCM (``hybrid_core_flow._run_neural_segment_rate``) walks
        ``compress_spike_sources(mapping.output_sources)`` once per
        simulation cycle and accumulates ``buffers[src_core][neuron]``
        into ``output_counts`` — but **only when the source core is
        inside its own active window** ``[core.latency, T + core.latency)``.
        That gate keeps the per-source contribution at exactly
        ``in_window_fires`` regardless of how far the segment's chain
        extends past that source, so a shallow output source feeding a
        deep segment doesn't get its last-cycle firing replayed for
        every stale cycle.  ``last_active_fires`` is no longer needed
        here, but the parameter is kept for ABI compatibility with
        existing callers in this module.
        """
        flat_sources = (
            list(output_sources.flatten())
            if output_sources is not None and hasattr(output_sources, "flatten")
            else (list(output_sources) if output_sources else [])
        )
        n_out = len(flat_sources)
        # Fallback for test fakes that don't carry output_sources: lay
        # out per-core counts contiguously into the output_map slots so
        # the unit tests in ``tests/unit/chip_simulation/test_sanafe_runner.py``
        # and ``tests/integration/test_sanafe_synthetic_end_to_end.py``
        # keep their pre-refactor expectations.
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
                # ``on`` axons contribute exactly ``T`` spikes per output —
                # one per input-train cycle.  HCM gates the always-on
                # accumulator on ``cycle < T`` (see
                # ``hybrid_core_flow._run_neural_segment_rate``).
                out[d0:d1] = int(T)
                continue
            if sp.kind == "input":
                # Input-typed output_sources are an unusual re-export
                # path; not currently exercised by the parity scenarios
                # and would need the per-cycle encoded train to
                # reproduce HCM's accumulation exactly.
                continue
            buf = core_outputs.get(int(sp.src_core))
            if buf is None:
                continue
            s0 = int(sp.src_start)
            length = int(sp.length)
            take = min(length, max(buf.size - s0, 0))
            if take <= 0:
                continue
            # In-window total only — HCM gates the accumulator on the
            # source's own active window so ``buf`` already holds
            # exactly ``in_window_fires`` per neuron.
            out[d0:d0 + take] = buf[s0:s0 + take]
        return out

    def _collect_last_active_fires(
        self,
        spike_trace: list,
        *,
        core_to_group: Dict[int, Any],
        hcm: Any,
    ) -> Dict[int, np.ndarray]:
        """Per-core bitmap: "did this neuron fire on its last active cycle?"

        Used by :func:`_compute_seg_output_spike_count` to add stale-buffer
        contributions matching HCM's per-cycle gather (see that method's
        docstring).  Returned bitmap is indexed by core's
        ``output_spike_count`` neuron order.
        """
        out: Dict[int, np.ndarray] = {}
        if not spike_trace:
            return out
        T_eff = len(spike_trace)
        # Per-cycle firing tally keyed by group name → (n_neurons, T_eff)
        # bitmap.  We only care about the last active cycle per core, so
        # build a single-column bitmap per group instead of the full
        # matrix.
        for core_idx, group in core_to_group.items():
            core = hcm.cores[core_idx]
            core_lat = int(getattr(core, "latency", 0) or 0)
            # SF cycle (0-indexed) for HCM's last active cycle = T+core_lat
            # (with the +1 pipeline shift baked into our active window).
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

    # --------------------------------------------------------- tile aggregation

    def _aggregate_per_tile(
        self,
        per_core_records: List[SanafeCoreRecord],
        results: Dict[str, Any],
        *,
        message_trace: Any = None,
    ) -> List[SanafeTileRecord]:
        """Partition cores into SANA-FE tiles + roll up per-tile energy.

        Distribution mirrors ``arch_synth.derive_arch_spec`` /
        ``net_synth.build_network_for_segment`` exactly:

            tile_idx = (core_offset + core_position) // cores_per_tile

        where ``cores_per_tile == 0`` means "one big tile holds everything".
        Energy is roughly partitioned by the tile's share of total per-core
        energy (the YAML-reported run total is preserved at the segment
        level; per-tile is a split for the GUI breakdown).
        """
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

        # Per-tile hop counts replicating SANA-FE's XY routing
        # (``chip.cpp:1118-1158``).  Hops on the **destination** tile per
        # direction are tallied; ``sim_calculate_tile_energy`` then
        # multiplies each direction-hop count by its YAML constant.  We
        # don't have separate per-direction YAML constants in the
        # preset (the arch_synth emits the same number on all four
        # cardinal directions), so per-direction counts collapse to a
        # single ``total_hops × hop_energy`` term.
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
            # Add the tile's hop budget into the network bucket
            # (``ts.network_energy += total_hop_energy`` at
            # ``chip.cpp:1190``).
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _group_name(group: Any) -> str:
    """Get a SANA-FE NeuronGroup's name (real API exposes ``get_name()``)."""
    if hasattr(group, "get_name"):
        return group.get_name()
    return group.name


def _group_name_to_size(net: Any) -> Dict[str, int]:
    """Build ``{group_name: size}`` from either real or fake Network shape.

    Real SANA-FE: ``net.groups`` is ``dict[name, NeuronGroup]``.
    Test fakes:   ``net.groups`` is ``list[FakeGroup]``.
    """
    groups = net.groups
    if isinstance(groups, dict):
        return {name: len(g) for name, g in groups.items()}
    return {_group_name(g): len(g) for g in groups}


def _used_axons(core: Any) -> int:
    """Number of live axons — matches HCM's ``axons_per_core - available_axons``
    accounting.  Real ``HardCore.axon_sources`` is padded to capacity with
    trailing ``is_off_`` markers, so we must NOT use ``len(axon_sources)``
    (that would give the padded length and break parity with HCM's
    ``CoreSpikeCounts.input_spike_count`` shape).
    """
    return int(core.axons_per_core) - int(getattr(core, "available_axons", 0))


def _used_neurons(core: Any) -> int:
    return int(core.neurons_per_core) - int(getattr(core, "available_neurons", 0))


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
    """Per-core energy via SANA-FE's exact accounting.

    Mirrors ``sim_calculate_core_energy`` (``sana_fe/src/chip.cpp:1202``)
    + the per-event functions in
    ``sana_fe/src/pipeline.hpp:calculate_synapse_default_energy_latency``,
    ``calculate_dendrite_default_energy_latency``,
    ``calculate_soma_default_energy_latency``:

        axon_in_energy  = packets_in × energy_message_in
        axon_out_energy = packets_out × energy_message_out
        synapse_energy  = (synapse process() calls)
                        × energy_process_spike
                        = incoming_spikes × energy_process_spike
        dendrite_energy = (dendrite process() calls)
                        × energy_update
                        = (n_neurons × T_eff) × energy_update
                          (dendrite is update_every_timestep=true here,
                           so SANA-FE invokes process() once per neuron
                           per chip cycle; see arch_synth._render_arch_yaml)
        soma_energy     = energy_access_neuron × (every process call)
                        + energy_update_neuron × (updated | fired)
                        + energy_spike_out      × fired

    For our soma plugin: ``process()`` runs every chip cycle (n_neurons
    × T_eff calls), returning ``idle`` outside ``[core.latency,
    core.latency+T_active)`` and ``updated``/``fired`` inside.  So
    SANA-FE charges access on every call but only adds update on the
    in-window calls — that's ``n_neurons × T_active`` ``updated|fired``
    events.  ``fired`` events == per-core firings.

    Hop energy is excluded — SANA-FE accounts hops at the *destination
    tile* (``chip.cpp:1154``), not per source core.  The tile rollup
    below applies it where it belongs.
    """
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
    """``(packets_in, packets_out)`` arrays indexed by global core id.

    Counted directly from the message trace — one entry per real
    (non-placeholder) message, matching SANA-FE's own per-message
    ``spike_messages_in`` / ``packets_out`` counters at
    ``chip.cpp:1207`` / ``1238``.  When ``cores_per_tile == 1`` we
    can use ``dest_core_id`` / ``src_core_id`` directly; with
    multi-core tiles we recover the global core id as
    ``tile_id * cores_per_tile + core_id_within_tile``.
    """
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
    """Per-core energy estimate when SANA-FE only reports run-wide totals."""
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
    """Tally SANA-FE's per-timestep firing list into per-group spike counts.

    SANA-FE 2.1.1 emits each timestep as a list of strings of the form
    ``"<group_name>.<neuron_index>"``.  We tally those into ``(n_neurons,)``
    int arrays, one per group.  Trailing groups absent from the trace
    are mapped to zero arrays.
    """
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
    """Convert SANA-FE's per-timestep firing strings into a (sum_neurons, T) matrix.

    Accepts both real SANA-FE's ``net.groups`` (dict) and test fakes (list).
    Returns ``None`` when the trace is empty or shapeless so the snapshot
    builder can opt out of per-neuron raster rendering cleanly.
    """
    if not spike_trace:
        return None
    # Build a name → (offset, size) map so we can index into a single matrix.
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
    """Convert SANA-FE's per-timestep potential list to a (n_logged, T) matrix.

    Returns ``None`` when the trace is empty or has inconsistent shape.
    """
    if not potential_trace:
        return None
    arr = np.asarray(potential_trace, dtype=np.float32)
    if arr.ndim != 2:
        return None
    # SANA-FE returns shape (T, n_logged); transpose to (n_logged, T).
    return arr.T


def _flatten_message_trace(message_trace: Any) -> Optional[List[dict]]:
    """Flatten SANA-FE's per-timestep message list into a single list of dicts."""
    if not message_trace:
        return None
    flat: List[dict] = []
    for events in message_trace:
        for ev in events:
            if isinstance(ev, dict):
                # Filter out placeholder entries (no real spike).
                if ev.get("placeholder"):
                    continue
                flat.append({k: (float(v) if isinstance(v, float) else v)
                             for k, v in ev.items()})
    return flat or None


def _aggregate_noc_links(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLink]:
    """Aggregate per-cycle messages into directed (src_tile, dst_tile) links.

    Powers the NoC-traffic overlay in the GUI floorplan view.  Only
    real (non-placeholder) cross-tile messages are counted; intra-tile
    spikes don't traverse the NoC.  Returns an empty list when SANA-FE
    didn't record a trace (``log_message_trace=False``) or the trace is
    structurally empty.
    """
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
                # Skip self-tile traffic — it doesn't cross the NoC.
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
        # Fall back to ``arch_geometry.tiles_xy`` if the message trace
        # didn't carry coordinates (older SANA-FE builds).
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


# ---------------------------------------------------------------------------
# New rich-visualisation aggregators (1-7, 9)
# ---------------------------------------------------------------------------


def _aggregate_noc_link_load(
    message_trace: Any,
    geom: Optional[SanafeArchGeometry],
) -> List[SanafeNocLinkLoad]:
    """Per-mesh-edge packet count using XY routing.

    SANA-FE routes packets through the mesh first along x then along y
    (XY routing — standard for 2D-mesh NoCs).  We approximate the
    per-edge load by walking that path for every recorded message and
    incrementing the count on each intermediate edge.  Distinct from
    ``_aggregate_noc_links`` (which collapses to (src, dst) pairs) —
    this captures *intermediate* tiles' edge load, which is what makes
    the heatmap actually useful for finding NoC hotspots.
    """
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
            # XY: walk x first, then y.  Skip self-traffic.
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
    """Reconstruct per-cycle event-driven energy split from raw traces.

    SANA-FE doesn't report per-cycle energy breakdowns natively — it
    only reports the run total.  We have the per-cycle event counts
    (firings from ``spike_trace``, packets + hops + processed spikes
    from ``message_trace``) and the per-event constants from the
    preset, so we can faithfully reconstruct the breakdown for the
    energy-waterfall view.

    Categories:
      synapse_j  = synapse_energy_j × spikes_processed (sum over packets)
      dendrite_j = dendrite_energy_j × unique_target_neurons_per_cycle
      soma_j     = soma_access + soma_update + soma_spike_out
                   × (active_neurons / firings)
      network_j  = (axon_in + axon_out + hop × tile_hop) × packet count
    """
    if not spike_trace and not message_trace:
        return []
    T_eff = max(
        len(spike_trace) if spike_trace else 0,
        len(message_trace) if message_trace else 0,
    )
    if T_eff <= 0:
        return []
    # Per-cycle counts.
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
    # Per-cycle soma access count = number of neurons touched this
    # cycle.  ``update_every_timestep`` means every live neuron is
    # accessed every cycle of its window; we approximate with the
    # total live neurons in the segment (an upper bound — small bias).
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
    """Return ``({group_name: core_index}, [core_index per global neuron row])``.

    The global row order matches ``_pack_spike_trace_matrix``: groups
    walked in dict/iteration order, concatenated.  This map lets
    cascade-timeline / critical-core helpers turn a spike-trace event
    string ``"<group>.<idx>"`` into the originating HardCore index
    (which carries the latency / etc.) without a second walk.
    """
    group_to_core: Dict[str, int] = {}
    # The runner names each core's LIF group ``core{idx}`` —
    # ``net_synth.build_network_for_segment`` does that explicitly.
    for core_idx, core in enumerate(hcm.cores):
        group_to_core[f"core{core_idx}"] = core_idx
    # Per-row core map (size = sum of group sizes) for spike_trace
    # row→core lookup.  Unknown groups (e.g. ``core{i}_in`` input
    # neurons) map to ``-1``.
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
    """Bucket per-cycle firings into ``core.latency`` layers.

    Output is sparse: only non-zero (cycle, depth) pairs are emitted,
    so a long quiet network produces a tiny payload.  ``depth`` is
    the segment-local core latency that HCM windows on; depth-0 is
    the input pool, deeper layers are downstream cascades.
    """
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
    """Per-cycle critical-core: the core with the highest event load.

    Score per core per cycle = (firings + incoming spikes).  The core
    with the max score is treated as that cycle's critical core (the
    one that dominated ``sim_time = max(neuron_processing,
    message_processing)``).  Returns one entry per cycle.
    """
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
                # ``dest_core_id`` is tile-local in SANA-FE; we want
                # the HardCore index.  When the runner places one core
                # per tile the two coincide; when ``cores_per_tile>1``
                # we need ``dest_tile_id * cores_per_tile + dest_core_id``.
                # For the critical-core view this is fine to approximate
                # by global core index = ``dest_tile_id * cpt + dest_core_id``.
                dt = int(ev.get("dest_tile_id", -1))
                dc = int(ev.get("dest_core_id", -1))
                if dt < 0 or dc < 0:
                    continue
                # Heuristic: walk hcm.cores in order, find first core that
                # belongs to (dt, dc).  Cheap because n_cores is small.
                idx = dt * (1 + dc) + dc  # rough first guess; bounds-check below
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
    """Return ``{group_name: starting_row_in_segment_raster}``.

    Same iteration as :func:`_pack_spike_trace_matrix` so per-core
    slicing stays consistent with the segment-wide raster's row layout.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    iterable = (groups.items() if isinstance(groups, dict)
                else ((_group_name(g), g) for g in groups))
    for name, g in iterable:
        offsets[name] = cursor
        cursor += len(g)
    return offsets


def _compute_noc_traffic_per_cycle(message_trace: Any) -> List[List[List[int]]]:
    """Per-cycle compact NoC traffic list for the animated playback view.

    Each cycle is a list of ``[src_x, src_y, dst_x, dst_y, count]``
    quintuples — duplicate (src, dst) packets within the same cycle
    are collapsed to one entry with a count.  Empty cycles produce an
    empty list (keeps the cycle index aligned with the spike trace).
    """
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
    """Sum ``|weight|`` over each ``(src_core, dst_core)`` pair.

    Activity-independent — drives the "live connectivity" overlay
    that shows routing complexity regardless of run state.  Walks
    every core's axon_sources × core_matrix once; pairs with all-zero
    weight columns are skipped.
    """
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
