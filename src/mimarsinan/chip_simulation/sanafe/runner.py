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

from .arch_synth import _sanafe, build_architecture, derive_arch_spec
from .net_synth import (
    build_network_for_segment, set_always_on_spike_trains, set_input_spike_trains,
)
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

        # Build the per-segment SANA-FE network.  Each HardCore in the HCM
        # maps to one SANA-FE core; input neurons live on the consuming
        # core (see ``sanafe_per_core_input_neurons`` memory).
        (net, core_to_group, core_input_neurons,
         core_always_on_neurons) = build_network_for_segment(
            self._arch, hcm,
            tile_offset=0, core_offset=0,
            cores_per_tile=self.cores_per_tile,
        )

        # Rate-encode the segment input as a (1, D, T) binary tensor.  Each
        # (core_idx, axon_idx) input neuron gets the spike train for its
        # logical input index ``hcm.cores[core_idx].axon_sources[axon_idx].neuron_``.
        encoded = uniform_rate_encode(seg_input_rates, self.T)            # (1, D, T)
        set_input_spike_trains(core_input_neurons, hcm, encoded)
        set_always_on_spike_trains(core_always_on_neurons, self.T)

        # Build and run the chip.
        chip = sanafe.SpikingChip(self._arch)
        chip.load(net)
        results = chip.sim(
            self.T,
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

        per_tile_records = self._aggregate_per_tile(per_core_records, results)

        seg_out_count = self._compute_seg_output_spike_count(
            stage.output_map, per_core_records,
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
        )

        # Store segment output rates into the state buffer for downstream stages.
        seg_output_rates = (seg_out_count.astype(np.float32) / float(self.T)
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
    ) -> Tuple[np.ndarray, int]:
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
            # Cross-core: pull the source core's per-neuron spike count
            # from the spike trace.
            src_core = int(src.core_)
            src_neuron = int(src.neuron_)
            src_group = core_to_group.get(src_core)
            if src_group is None:
                continue
            gsc = group_spike_counts.get(_group_name(src_group))
            if gsc is None or src_neuron >= len(gsc):
                continue
            counts[a] = int(gsc[src_neuron])
        return counts, n_always_on

    # --------------------------------------------------------- segment output

    def _compute_seg_output_spike_count(
        self,
        output_map: List[Any],
        per_core_records: List[SanafeCoreRecord],
    ) -> np.ndarray:
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

    # --------------------------------------------------------- tile aggregation

    def _aggregate_per_tile(
        self,
        per_core_records: List[SanafeCoreRecord],
        results: Dict[str, Any],
    ) -> List[SanafeTileRecord]:
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
