from __future__ import annotations

import os
import time as _time
from typing import Dict

import numpy as np

from mimarsinan.chip_simulation.recording.spike_recorder import CoreSpikeCounts, SegmentSpikeRecord
from mimarsinan.mapping.packing.softcore import HardCoreMapping
from mimarsinan.mapping.support.core_geometry import used_axons, used_neurons
from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
from mimarsinan.chip_simulation.lava_loihi.timing import _LAVA_DTYPE, _SegmentTiming


class LavaSegmentMixin:
    def _run_neural_segment_scheduled(
        self,
        seg: HardCoreMapping,
        seg_input_rates: np.ndarray,
        *,
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> np.ndarray:
        """Execute a segment with host-scheduled routing and Lava per-core LIF."""
        T = self.T
        N = seg_input_rates.shape[0]
        seg_in_size = seg_input_rates.shape[1]
        timing = _SegmentTiming.from_mapping(seg, T)

        seg_input_spikes = self._behavior.encode_segment_input(seg_input_rates, T)
        seg_input_logical = np.zeros(
            (seg_in_size, N, timing.sample_stride), dtype=_LAVA_DTYPE,
        )
        seg_input_logical[:, :, :T] = seg_input_spikes.transpose(1, 0, 2)

        core_output_spikes: Dict[int, np.ndarray] = {}
        core_buffer_spikes: Dict[int, np.ndarray] = {}

        deps = {
            idx: sorted(
                {
                    int(sp.src_core)
                    for sp in core.get_axon_source_spans()
                    if sp.kind == "core"
                }
            )
            for idx, core in enumerate(seg.cores)
        }
        topo_order: list[int] = []
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(idx: int) -> None:
            if idx in visited:
                return
            if idx in visiting:
                raise RuntimeError(f"Cycle detected in neural segment at core {idx}")
            visiting.add(idx)
            for dep in deps[idx]:
                visit(dep)
            visiting.remove(idx)
            visited.add(idx)
            topo_order.append(idx)

        for idx in sorted(
            range(len(seg.cores)),
            key=lambda i: (timing.core_latency(seg.cores[i]), i),
        ):
            visit(idx)

        t0 = _time.time()
        verbose = os.environ.get("MIMARSINAN_LOIHI_QUIET") != "1"
        n_cores = len(topo_order)
        if verbose:
            print(
                f"  [LavaLoihiRunner] segment with {n_cores} cores; T={T}, N={N} "
                f"— per-core timing follows",
                flush=True,
            )
        for step_idx, core_idx in enumerate(topo_order):
            t_core = _time.time()
            core = seg.cores[core_idx]
            latency = timing.core_latency(core)
            used_ax = used_axons(core, min_one=True)
            used_neu = used_neurons(core, min_one=True)
            active_input = np.zeros((used_ax, N, T), dtype=_LAVA_DTYPE)

            for sp in core.get_axon_source_spans():
                d0 = int(sp.dst_start)
                if d0 >= used_ax:
                    continue
                end = min(int(sp.dst_end), used_ax)
                take = end - d0
                if sp.kind == "off":
                    continue
                if sp.kind == "on":
                    active_input[d0:end, :, :] = 1.0
                    continue
                if sp.kind == "input":
                    s0 = int(sp.src_start)
                    active_input[d0:end, :, :] = seg_input_logical[
                        s0:s0 + take, :, latency:latency + T,
                    ]
                    continue

                src_core_id = int(sp.src_core)
                if src_core_id not in core_buffer_spikes:
                    raise RuntimeError(
                        f"Core {core_idx} depends on core {src_core_id}, "
                        "but the source has not been scheduled yet."
                    )
                s0 = int(sp.src_start)
                for local_cycle in range(T):
                    src_cycle = latency + local_cycle - 1
                    if src_cycle < 0:
                        continue
                    active_input[d0:end, :, local_cycle] = core_buffer_spikes[
                        src_core_id
                    ][s0:s0 + take, :, src_cycle]

            weights = np.asarray(
                core.core_matrix[:used_ax, :used_neu], dtype=_LAVA_DTYPE,
            ).T
            hardware_bias = (
                np.asarray(core.hardware_bias[:used_neu], dtype=_LAVA_DTYPE)
                if getattr(core, "hardware_bias", None) is not None
                else None
            )
            t_lava = _time.time()
            active_output = self._run_core_lava(
                weights=weights,
                threshold=float(core.threshold),
                hardware_bias=hardware_bias,
                input_spikes=active_input.reshape(used_ax, N * T),
            ).reshape(used_neu, N, T)
            lava_dt = _time.time() - t_lava

            full_output = np.zeros(
                (int(core.neurons_per_core), N, timing.sample_stride),
                dtype=_LAVA_DTYPE,
            )
            full_output[:used_neu, :, latency:latency + T] = active_output
            core_output_spikes[core_idx] = full_output

            buffered = full_output.copy()
            hold_start = latency + T
            if hold_start < timing.sample_stride:
                buffered[:, :, hold_start:] = buffered[:, :, hold_start - 1:hold_start]
            core_buffer_spikes[core_idx] = buffered
            if verbose:
                print(
                    f"    core {step_idx + 1:>4}/{n_cores} "
                    f"(idx={core_idx}, ax={used_ax}, neu={used_neu}, "
                    f"lat={latency}): lava={lava_dt*1000:7.1f}ms  "
                    f"core_total={(_time.time() - t_core)*1000:7.1f}ms",
                    flush=True,
                )

        print(
            f"  [LavaLoihiRunner] scheduled segment run: {len(seg.cores)} cores, "
            f"{timing.total_steps(N)} logical cycles, {_time.time() - t0:.1f}s",
            flush=True,
        )

        out_size = len(seg.output_sources)
        seg_out_spikes = np.zeros(
            (out_size, N, timing.sample_stride), dtype=_LAVA_DTYPE,
        )
        out_spans = compress_spike_sources(seg.output_sources)
        for sp in out_spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                # Always-on: gate to cycle < T to match HCM spike counts.
                seg_out_spikes[d0:d1, :, :T] = 1.0
                continue
            if sp.kind == "input":
                seg_out_spikes[d0:d1, :, :] = seg_input_logical[
                    int(sp.src_start) : int(sp.src_end), :, :
                ]
                continue
            seg_out_spikes[d0:d1, :, :] = core_output_spikes[int(sp.src_core)][
                int(sp.src_start) : int(sp.src_end), :, :
            ]

        seg_out_counts = seg_out_spikes.sum(axis=2).T
        seg_out_rates = seg_out_counts / float(T)

        if recorder_seg is not None:
            assert N == 1, "Spike recording requires a single sample (N == 1)"
            recorder_seg.seg_output_spike_count = seg_out_counts[0].astype(np.int64)
            for core_idx, core in enumerate(seg.cores):
                used_ax = used_axons(core, min_one=True)
                used_neu = used_neurons(core, min_one=True)
                latency = timing.core_latency(core)
                active_slice = slice(latency, latency + T)

                out_count = core_output_spikes[core_idx][
                    :used_neu, 0, active_slice
                ].sum(axis=1).astype(np.int64)

                in_count = np.zeros(used_ax, dtype=np.int64)
                n_always_on = 0
                for sp in core.get_axon_source_spans():
                    d0 = int(sp.dst_start)
                    if d0 >= used_ax:
                        continue
                    end = min(int(sp.dst_end), used_ax)
                    take = end - d0
                    if sp.kind == "off":
                        continue
                    if sp.kind == "on":
                        n_always_on += take
                        in_count[d0:end] += T
                        continue
                    if sp.kind == "input":
                        s0 = int(sp.src_start)
                        in_count[d0:end] += seg_input_logical[
                            s0:s0 + take, 0, active_slice
                        ].sum(axis=1).astype(np.int64)
                        continue
                    s0 = int(sp.src_start)
                    src_start = max(latency - 1, 0)
                    src_end = max(latency + T - 1, 0)
                    in_count[d0:end] += core_buffer_spikes[int(sp.src_core)][
                        s0:s0 + take, 0, src_start:src_end
                    ].sum(axis=1).astype(np.int64)

                recorder_seg.cores.append(
                    CoreSpikeCounts(
                        core_index=core_idx,
                        n_in_used=used_ax,
                        n_out_used=used_neu,
                        core_latency=int(core.latency) if core.latency is not None else -1,
                        has_hardware_bias=getattr(core, "hardware_bias", None) is not None,
                        n_always_on_axons=n_always_on,
                        input_spike_count=in_count,
                        output_spike_count=out_count,
                    )
                )

        return seg_out_rates.astype(np.float32)

    def _run_neural_segment(
        self,
        seg: HardCoreMapping,
        seg_input_rates: np.ndarray,
        *,
        recorder_seg: SegmentSpikeRecord | None = None,
    ) -> np.ndarray:
        """Execute one neural segment with HCM-equivalent Lava core dynamics."""
        return self._run_neural_segment_scheduled(
            seg, seg_input_rates, recorder_seg=recorder_seg,
        )
