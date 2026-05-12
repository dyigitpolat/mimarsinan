"""SANA-FE backend driver.

``SanafeRunner.run(sample_input, sample_index)`` walks a
``HybridHardCoreMapping`` end-to-end: each neural stage is built as a
SANA-FE network on a shared ``Architecture``, run through
``SpikingChip.sim()``, and reduced to a :class:`SanafeSegmentRecord`;
each compute stage runs host-side via ``hybrid_execution.execute_compute_op_numpy``.

The runner is the only module that touches ``sanafe.SpikingChip``.  All
SANA-FE imports are gated behind ``arch_synth._sanafe()`` so the rest of
mimarsinan never grows a GPL-3.0 dependency at import time.

Single-sample only at this stage: the parity-gate use case feeds one
deterministic test sample at a time.  Higher sample counts are handled
by the pipeline step looping over samples.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from mimarsinan.chip_simulation._spike_encoding import uniform_rate_encode
from mimarsinan.chip_simulation.hybrid_execution import (
    assemble_segment_input_numpy,
    execute_compute_op_numpy,
    gather_final_output_numpy,
    store_segment_output_numpy,
)
from mimarsinan.code_generation.cpp_chip_model import SpikeSource

from .arch_synth import _sanafe, build_architecture, derive_arch_spec
from .net_synth import build_network_for_segment
from .neuron_model import resolve_plugin_path
from .presets import PRESETS
from .records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
    SanafeRunRecord,
    SanafeSegmentRecord,
    SanafeTileRecord,
)


_RAW_INPUT_NODE_ID = -2


class SanafeRunner:
    """Run one hybrid mapping sample through SANA-FE and collect rich stats."""

    def __init__(
        self,
        mapping: Any,
        simulation_length: int,
        *,
        arch_preset: str = "loihi",
        custom_arch_path: Optional[str] = None,
        thresholding_mode: str = "<",
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
        self.T = int(simulation_length)
        self.arch_preset = arch_preset
        self.custom_arch_path = custom_arch_path
        self.thresholding_mode = thresholding_mode
        self.log_potential_trace = log_potential_trace
        self.log_message_trace = log_message_trace
        self.cores_per_tile = cores_per_tile

        # Cached on first SANA-FE touch (after run() begins).
        self._arch: Optional[Any] = None
        self._arch_name: str = "<unbuilt>"
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
                )
                # Materialize to numpy if torch tensor.
                if hasattr(result, "detach"):
                    result = result.detach().cpu().numpy()
                state_buffer[op.id] = np.asarray(result)
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
        self._arch = build_architecture(spec, custom_arch_path=self.custom_arch_path)
        # Optional Strategy-B plugin path resolution (no-op when plugin absent).
        plugin_path = resolve_plugin_path()
        if plugin_path and hasattr(self._arch, "load_soma_plugin"):  # pragma: no cover
            self._arch.load_soma_plugin(plugin_path, "mimarsinan_subtractive_lif")

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
        seg_input_rates = assemble_segment_input_numpy(
            stage.input_map, state_buffer, num_samples=1,
        )
        seg_in_size = int(seg_input_rates.shape[1])

        # Build the per-segment SANA-FE network.
        net, core_to_group, input_group, always_on_group = build_network_for_segment(
            self._arch, hcm,
            tile_offset=0, core_offset=0,    # one tile per segment for now
            seg_in_size=seg_in_size,
            cores_per_tile=self.cores_per_tile,
        )

        # Rate-encode the input.
        encoded = uniform_rate_encode(seg_input_rates, self.T)   # (1, D, T)

        # Build and run the chip.
        chip = sanafe.SpikingChip(self._arch)
        chip.load(net)
        results = chip.sim(
            self.T,
            spike_trace=True,
            potential_trace=self.log_potential_trace,
            input_spikes=encoded,
        )
        self._last_chip = chip

        # Extract per-core stats.
        group_spike_counts = results.get("group_spike_counts", {})
        per_core_records: List[SanafeCoreRecord] = []
        seg_output_pieces: Dict[int, np.ndarray] = {}
        for core_idx, core in enumerate(hcm.cores):
            used_neu = _used_neurons(core)
            used_ax = _used_axons(core)
            if used_neu <= 0:
                continue
            group = core_to_group.get(core_idx)
            if group is None:
                output_count = np.zeros(used_neu, dtype=np.int64)
            else:
                gsc = group_spike_counts.get(group.name)
                if gsc is None:
                    output_count = np.zeros(used_neu, dtype=np.int64)
                else:
                    output_count = np.asarray(gsc, dtype=np.int64)[:used_neu]
            input_count, n_always_on = self._derive_per_core_input_counts(
                core=core,
                seg_input_encoded=encoded,
                state_buffer_spikes={},   # Cross-core spike trains not modelled in
                                          # this single-segment slice yet; refined in
                                          # the slow integration test.
            )
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
                energy=_energy_share(
                    SanafeEnergyBreakdown.from_sanafe_dict(results["energy"]),
                    n_cores=len([c for c in hcm.cores if _used_neurons(c) > 0]),
                ),
            ))

        # Per-tile aggregation (one tile per segment under default packing).
        per_tile_records = self._aggregate_per_tile(per_core_records, results)

        # Segment-level output spike count: concatenate from output_map slices.
        seg_out_count = self._compute_seg_output_spike_count(
            stage.output_map, per_core_records, hcm, state_buffer,
        )
        seg_in_count = uniform_rate_encode(seg_input_rates, self.T)[0].sum(axis=1).astype(np.int64)

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
            per_neuron_spike_trace=results.get("spike_trace"),
            per_neuron_potential_trace=results.get("potential_trace"),
            message_trace=results.get("message_trace"),
        )

        # Store segment output rates into the state buffer for downstream stages.
        seg_output_rates = _seg_output_spike_count_to_rates(
            seg_out_count, self.T,
        ).reshape(1, -1)
        store_segment_output_numpy(
            stage.output_map, state_buffer, seg_output_rates,
        )
        return seg_record

    # ----------------------------------------------------- per-core input

    def _derive_per_core_input_counts(
        self,
        *,
        core: Any,
        seg_input_encoded: np.ndarray,   # (1, seg_in_size, T)
        state_buffer_spikes: Dict[int, np.ndarray],
    ) -> tuple[np.ndarray, int]:
        """Walk this core's ``axon_sources``; build (input_spike_count, n_always_on)."""
        used_ax = _used_axons(core)
        if used_ax <= 0:
            return np.zeros(0, dtype=np.int64), 0
        counts = np.zeros(used_ax, dtype=np.int64)
        n_always_on = 0
        for a in range(used_ax):
            src = core.axon_sources[a]
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_always_on_", False):
                counts[a] = self.T
                n_always_on += 1
                continue
            if getattr(src, "is_input_", False):
                k = int(src.neuron_)
                counts[a] = int(seg_input_encoded[0, k, :].sum())
                continue
            # Cross-core spikes — populated host-side from upstream output
            # traces in the slow integration path. Default to 0 here.
            counts[a] = 0
        return counts, n_always_on

    # --------------------------------------------------------- segment output

    def _compute_seg_output_spike_count(
        self,
        output_map: List[Any],
        per_core_records: List[SanafeCoreRecord],
        hcm: Any,
        state_buffer: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Concatenate per-core spike-count slices in ``output_map`` order.

        Each ``output_map`` slice tells us which sub-range of which source's
        outputs go into the segment-level output buffer.  For HCM-mapped
        neural segments, the source node_id (e.g. the IR node id) is
        resolved by the runner's caller — here we approximate by summing
        per-core output counts across cores in declaration order, then
        slicing per the output_map.  This shape is verified by the parity
        gate at the segment-output diff layer (layer 4).
        """
        # Flatten all per-core output spike counts in core order; cap length
        # at the max output_map endpoint.
        if not output_map:
            # No outputs declared — return empty.
            return np.zeros(0, dtype=np.int64)

        total_size = max((s.offset + s.size for s in output_map), default=0)
        out = np.zeros(total_size, dtype=np.int64)
        flat = np.concatenate(
            [rec.output_spike_count for rec in per_core_records],
            axis=0,
        ) if per_core_records else np.zeros(0, dtype=np.int64)
        # First output_map slice consumes leading core outputs by convention.
        cursor = 0
        for slot in output_map:
            take = min(slot.size, max(flat.size - cursor, 0))
            if take > 0:
                out[slot.offset:slot.offset + take] = flat[cursor:cursor + take]
            cursor += slot.size
        return out

    # --------------------------------------------------------- tile aggregation

    def _aggregate_per_tile(
        self,
        per_core_records: List[SanafeCoreRecord],
        results: Dict[str, Any],
    ) -> List[SanafeTileRecord]:
        """One tile per segment under the default cores_per_tile=0 packing."""
        if not per_core_records:
            return []
        return [
            SanafeTileRecord(
                tile_index=0,
                cores=[c.core_index for c in per_core_records],
                energy=SanafeEnergyBreakdown.from_sanafe_dict(results["energy"]),
                spikes_fired=int(sum(c.spikes_fired for c in per_core_records)),
                packets_sent=int(results.get("packets_sent", 0)),
            )
        ]


# ---------------------------------------------------------------------------
# Module-level helpers (mirror net_synth's "used" counters)
# ---------------------------------------------------------------------------


def _used_axons(core: Any) -> int:
    if hasattr(core, "axon_sources"):
        return len(core.axon_sources)
    return int(core.axons_per_core) - int(getattr(core, "available_axons", 0))


def _used_neurons(core: Any) -> int:
    return int(core.neurons_per_core) - int(getattr(core, "available_neurons", 0))


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


def _seg_output_spike_count_to_rates(counts: np.ndarray, T: int) -> np.ndarray:
    if T <= 0:
        return counts.astype(np.float32, copy=True)
    return (counts.astype(np.float32) / float(T))
